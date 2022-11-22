#!/usr/bin/env python
import logging
import math
import os
import sqlite3
import sys
from os.path import join
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Union

import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
import torch
import transformers
import typer
from optuna import Trial
from scipy.special import expit as sigmoid
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

import exmed_bert.models as MM
from exmed_bert.data import PatientDataset
from exmed_bert.data.dataset import ExtendedPatientDataset
from exmed_bert.utils import FineTuningTrainer
from exmed_bert.utils.helpers import ModelLoader, calculate_class_weights, read_id_file
from exmed_bert.utils.metrics import (
    compute_classification_report,
    compute_endpoint_metrics,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def prepare_dataset(
    dataset: PatientDataset,
    observations: Optional[pd.DataFrame] = None,
    iptw_df: Optional[pd.DataFrame] = None,
) -> ExtendedPatientDataset:
    combi_dataset = ExtendedPatientDataset.load_from_normal_dataset(dataset)
    if observations is not None:
        combi_dataset.load_observations(observations)
    if iptw_df is not None:
        combi_dataset.load_iptw(iptw_df)
    return combi_dataset


def make_subset(data: PatientDataset, ids: FrozenSet[int]) -> PatientDataset:
    idx = [i for i in range(len(data)) if data.get_patient(i)[0].patient_id in ids]
    return data.subset_by_ids(idx)


# Function to train code and substance model
def finetune(
    train: str,
    val: str,
    test: str,
    model_in: str,
    output_dir: str,
    metric: str = "eval_auc",
    iptw_df: Optional[str] = None,
    study_name: str = "training",
    endpoint: str = "arm",
    observation_df: Optional[str] = None,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    epochs: Optional[int] = None,
    max_steps: int = -1,
    learning_rate: float = 1e-5,
    gradient_accumulation_steps: int = 1,
    seed: int = 201214,
    num_workers: int = 0,
    warmup_steps: int = 100,
    warmup_ratio: Optional[float] = None,
    hyperopt: bool = False,
    classification_head: str = "ffn",
    greater_is_better: bool = True,
    stop_metric: Optional[str] = None,
    stop_greater_is_better: bool = False,
    num_trials: int = 30,
    reweigh_loss: bool = False,
    valid_ids: Optional[str] = None,
):

    # assertion check
    if not torch.cuda.is_available():
        raise Exception(
            f"Cuda is not available. {torch.cuda.is_available()}. {torch.cuda.device_count()}"
        )
    else:
        logger.info(
            f"Cuda is available. {torch.cuda.is_available()}. {torch.cuda.device_count()}"
        )

    # create directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_data_dir = os.path.join(output_dir, "data")
    Path(output_data_dir).mkdir(parents=True, exist_ok=True)
    model_dir = os.path.join(output_dir, "model")
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # set variables
    if stop_metric is None:
        stop_metric = metric
        stop_greater_is_better = greater_is_better

    transformers.set_seed(seed)
    mlflow.set_tracking_uri(f"sqlite:///{output_data_dir}/mlruns.db")

    # Data
    def read_file(x: Union[str, Path, None]) -> Union[pd.DataFrame, None]:
        return pd.read_csv(x, index_col="patient_id") if x is not None else None

    observations = read_file(observation_df)
    iptw = read_file(iptw_df)
    train_data = prepare_dataset(
        PatientDataset.load_dataset(train, to_ram=True), observations, iptw
    )
    val_data = prepare_dataset(
        PatientDataset.load_dataset(val, to_ram=True), observations, None
    )
    train_data.eval = False
    val_data.eval = False

    if valid_ids is not None:
        ids = read_id_file(valid_ids)
        train_data = make_subset(train_data, ids)
        val_data = make_subset(val_data, ids)
    else:
        ids = frozenset()

    # train_data.predict_substances = False
    # val_data.predict_substances = False

    train_data.endpoints([endpoint])
    val_data.endpoints([endpoint])
    logger.info(f" loaded train_dataset length is: {len(train_data)}")
    logger.info(f" loaded val_dataset length is: {len(val_data)}")

    # Model configuration
    config = MM.CombinedConfig.from_json_file(join(model_in, "config.json"))
    config.num_endpoints = config.num_labels = 1
    config.max_position_embedding = 512
    config.classification_head = classification_head

    if observations is not None:
        config.num_observations = observations.shape[1]
    else:
        config.num_observations = 0

    if reweigh_loss:
        class_weights = calculate_class_weights(train_data)
        config.bc_pos_weight = float(class_weights[train_data.endpoint_index])

    model_init = ModelLoader(
        model_in,
        config,
        code_embed=train_data.code_embed,
        extended=True if iptw_df is not None or observation_df is not None else False,
    )

    # Training
    if warmup_ratio is not None and epochs is not None:
        warmup_steps = int(warmup_ratio * (len(train_data) / train_batch_size) * epochs)
    elif warmup_ratio is not None and max_steps < 0:
        warmup_steps = int(warmup_ratio * max_steps)
    elif epochs is None and warmup_steps is None:
        raise Exception("Must specify epochs or steps")
    logger.info(f"Performing warmup for {warmup_steps} steps.")

    if epochs is None and max_steps > 0:
        epochs = int(max_steps / math.ceil((len(train_data) / train_batch_size)))

    if epochs is None:
        raise Exception

    training_config = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,  # type: ignore
        per_device_eval_batch_size=eval_batch_size,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        label_names=["endpoint_labels"],
        dataloader_num_workers=num_workers,
        logging_dir=f"{output_data_dir}/logs",
        fp16=True,
        save_steps=None,  # type: ignore
        load_best_model_at_end=True,
        save_strategy="epoch",
        metric_for_best_model=stop_metric,
        greater_is_better=stop_greater_is_better,
    )
    logger.info(f"Using {training_config.n_gpu} gpus.")

    logger.info("***** Initialize trainer instance *****")
    trainer: Trainer = FineTuningTrainer(
        model_init=model_init,
        args=training_config,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_endpoint_metrics,
        callbacks=[EarlyStoppingCallback(2)],
    )

    if hyperopt:
        # hyperparameter tuning
        def hyperparameter_space(trial: Trial) -> Dict[str, float]:
            hps = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-3),
                "per_device_train_batch_size": trial.suggest_categorical(
                    "per_device_train_batch_size", [8, 16, 32]
                ),
                "warmup_ratio": trial.suggest_categorical(
                    "warmup_ratio", [0.0, 0.05, 0.1]
                ),
                "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-1),
            }

            if classification_head != "ffn":
                num_layers = trial.suggest_int("rnn_num_layers", 1, 4)
                hps["rnn_num_layers"] = num_layers
            else:
                num_final_blocks = trial.suggest_int("num_final_blocks", 0, 6)
                hps["num_final_blocks"] = num_final_blocks

            return hps

        db_path = f"sqlite:///{output_data_dir}/optuna.db"
        best_run = trainer.hyperparameter_search(
            hp_space=hyperparameter_space,
            compute_objective=lambda x: x[metric],
            n_trials=num_trials,
            direction="maximize" if greater_is_better else "minimize",
            storage=db_path,
            study_name=study_name,
            load_if_exists=True,
            backend="optuna",
            pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=epochs),  # type: ignore
        )

        # set best parameters
        # get best epoch from optuna study
        with sqlite3.connect(database=db_path.replace("sqlite:///", "")) as db:
            cur = db.cursor()
            num_epochs = (
                int(
                    cur.execute(
                        f"""
            SELECT study_name, tiv.trial_id, step , MAX(intermediate_value)
            FROM trial_intermediate_values tiv
            JOIN trials tr
            ON tiv.trial_id=tr.trial_id
            JOIN studies st
            ON tr.study_id=st.study_id
            WHERE study_name = '{study_name}';
            """
                    ).fetchall()[0][2]
                )
                + 1
            )
            cur.close()

        best_training_arguments = training_config.to_dict()
        best_training_arguments = {
            key: value
            for key, value in best_training_arguments.items()
            if key[0] != "_" and value != -1
        }

        if "rnn_num_layers" in best_run.hyperparameters.keys():
            num_layers = best_run.hyperparameters.pop("rnn_num_layers")
            model_init.config.rnn_num_layers = num_layers

        if "num_final_blocks" in best_run.hyperparameters.keys():
            num_final_blocks = best_run.hyperparameters.pop("num_final_blocks")
            model_init.config.num_final_blocks = num_final_blocks

        for parameter, value in best_run.hyperparameters.items():
            best_training_arguments[parameter] = value

        best_training_arguments["num_train_epochs"] = num_epochs
        best_training_arguments["load_best_model_at_end"] = False
        best_training_arguments["evaluation_strategy"] = "no"

        logging.info(f"Best parameters are: {best_training_arguments}")

        # repeat training on entire training data
        logger.info("Start training on train+validation datasets.")
        best_training_arguments = TrainingArguments(**best_training_arguments)
        entire_training_data = train_data + val_data
        logger.info(f"Train on dataset with {len(entire_training_data)} samples.")
        trainer = FineTuningTrainer(
            model=model_init(),
            args=best_training_arguments,
            train_dataset=entire_training_data,
        )
        trainer.create_optimizer()
        max_steps = (
            math.ceil(
                len(entire_training_data) / best_training_arguments.train_batch_size
            )
            * epochs
            if max_steps < 0
            else max_steps
        )
        trainer.create_scheduler(
            num_training_steps=max_steps, optimizer=trainer.optimizer
        )
        trainer.train()
    else:
        trainer.train()

    # evaluate model
    del train_data
    del val_data
    test_data = prepare_dataset(
        PatientDataset.load_dataset(test, to_ram=True), observations, None
    )
    test_data.eval = False
    if valid_ids is not None:
        test_data = make_subset(test_data, ids)

    test_data.endpoints([endpoint])
    logger.info(f" loaded test_dataset length is: {len(test_data)}")

    logits, labels, metrics = trainer.predict(test_dataset=test_data)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(metrics.items()):  # type: ignore
            writer.write(f"{key} = {value}\n")

    compute_classification_report(logits, labels).to_csv(  # type: ignore
        join(output_data_dir, "classification_report.csv")
    )

    trainer.save_model(model_dir)

    #
    probs = sigmoid(logits)
    labels = labels.astype(bool)  # type: ignore
    threshold_scores = []
    for th in np.linspace(0, 1, num=1000):
        th_preds = np.where(probs > th, True, False)
        pr, rc, f1, _ = sklearn.metrics.precision_recall_fscore_support(  # type: ignore
            y_true=labels, y_pred=th_preds, average="binary", zero_division=0
        )
        threshold_scores.append(
            {"threshold": th, "f1": f1, "precision": pr, "recall": rc}
        )
    scores = pd.DataFrame(threshold_scores).melt(id_vars="threshold")
    scores.to_csv(os.path.join(output_data_dir, "scores.csv"))


if __name__ == "__main__":
    typer.run(finetune)
