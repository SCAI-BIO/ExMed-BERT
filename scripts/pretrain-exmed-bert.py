#!/usr/bin/env python
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import mlflow
import transformers
import typer
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

import exmed_bert.models as MM
from exmed_bert.data.dataset import PatientDataset
from exmed_bert.models.trainer import MedBertTrainer
from exmed_bert.utils.metrics import compute_metrics_merged_model


def pretrain(
    training_data: Path,
    validation_data: Path,
    output_dir: Path,
    output_data_dir: Path,
    model_dir: Optional[Path] = None,
    train_batch_size: int = 256,
    eval_batch_size: int = 256,
    num_attention_heads: int = 6,
    num_hidden_layers: int = 6,
    hidden_size: int = 288,
    intermediate_size: int = 512,
    epochs: Optional[int] = None,
    max_steps: int = -1,
    learning_rate: float = 3e-5,
    gradient_accumulation_steps: int = 1,
    max_seq_length: int = 512,
    seed: int = 201214,
    num_workers: int = 0,
    logging_steps: int = 100,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    warmup_steps: int = 10_000,
    warmup_ratio: Optional[float] = None,
    initialization: str = "orthogonal",
    dynamic_masking: bool = False,
    max_masked: Optional[int] = None,
    plos: bool = True,
):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    output_file_handler = logging.FileHandler(f"{output_dir}/output.log")
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)

    transformers.set_seed(seed)
    mlflow.set_tracking_uri(f"sqlite:///{output_data_dir}/mlruns.db")

    # Data
    Path(output_data_dir).mkdir(parents=True, exist_ok=True)

    train_data = PatientDataset.load_dataset(str(training_data))
    val_data = PatientDataset.load_dataset(str(validation_data))
    train_data.dynamic_masking = dynamic_masking
    val_data.dynamic_masking = dynamic_masking

    if max_masked is not None and dynamic_masking:
        train_data.max_masked = max_masked
        val_data.max_masked = max_masked

    logger.info(f" loaded train_dataset length is: {len(train_data)}")
    logger.info(f" loaded test_dataset length is: {len(val_data)}")

    code_embed = train_data.code_embed
    age_embed = train_data.age_embed
    sex_embed = train_data.sex_embed
    state_embed = train_data.state_embed

    # Model configuration
    config = MM.ExMedBertConfig(
        age_vocab_size=len(age_embed),
        code_vocab_size=len(code_embed),
        graph_heads=4,
        graph_hidden_size=int(hidden_size / 4),
        hidden_size=hidden_size,
        initialization=initialization,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_seq_length,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        number_of_codes=len(code_embed.entity_to_id),
        pad_token_id=code_embed("PAD"),  # type: ignore
        plos_weight=(0.25, 1.0),  # account for unbalanced dataset
        predict_codes=True,
        predict_los=plos,
        region_vocab_size=len(state_embed),
        sex_vocab_size=len(sex_embed),
    )

    if model_dir is not None:
        logger.info("Load pretrained model")
        model = MM.ExMedBertForMaskedLM.from_pretrained(
            model_dir, config=config, code_embed=code_embed
        )
    else:
        logger.info("Initialize new model")
        model = MM.ExMedBertForMaskedLM(config, code_embed=code_embed)

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

    train_data.eval = True

    training_config = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_eval_batch_size=eval_batch_size,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        evaluation_strategy=transformers.IntervalStrategy.STEPS,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        label_names=["code_labels", "plos_label"],
        dataloader_num_workers=num_workers,
        logging_dir=f"{output_data_dir}/logs",
        fp16=False,
        save_steps=save_steps,
        ignore_data_skip=True,
    )
    logger.info(f"Using {training_config.n_gpu} gpus.")

    logger.info("***** Initialize trainer instance *****")
    trainer = MedBertTrainer(
        model=model,  # type: ignore
        args=training_config,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics_merged_model,
        model_init=None,  # type: ignore
    )

    if get_last_checkpoint(output_dir) is not None:
        logger.info("***** continue training *****")
        logger.info(f"Last checkpoint is {get_last_checkpoint(output_dir)}")
        trainer.train(resume_from_checkpoint=get_last_checkpoint(output_dir))
    else:
        logger.info("***** start training *****")
        trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=val_data)

    # writes eval result to file which can be accessed later in s3 output
    with open(os.path.join(output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    if model_dir is not None:
        trainer.save_model(model_dir)
    else:
        trainer.save_model(output_data_dir / "pretrained-model")


if __name__ == "__main__":
    typer.run(pretrain)
