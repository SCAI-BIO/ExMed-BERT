#!/usr/bin/env python
import os
import re
import sys
from typing import FrozenSet, List, Optional, Tuple, Union

from exmed_bert.utils.helpers import read_id_file
from exmed_bert.utils.metrics import calc_auc20, calc_prauc

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import logging
import pickle
from collections import Counter
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
import seaborn as sns
import sklearn
import typer
import xgboost as xgb
from typer import Argument, Option

logger = logging.getLogger(__name__)


def prepare_df(
    df: pd.DataFrame,
    endpoint: str,
    available_endpoints: List[str],
    iptw_df: Optional[pd.DataFrame] = None,
    valid_ids: Optional[FrozenSet] = None,
) -> Tuple[xgb.DMatrix, npt.ArrayLike, npt.ArrayLike, Union[npt.ArrayLike, None]]:

    if valid_ids is not None:
        df = df[df.index.isin(valid_ids)]

    labels = df.pop(endpoint)
    _ = [df.pop(x) for x in available_endpoints if x != endpoint]
    to_remove = [i for i, col in enumerate(df.columns) if re.match("Unnamed.*", col)]
    if len(to_remove) > 0:
        df.drop(df.columns[to_remove], axis=1, inplace=True)  # type: ignore
    if iptw_df is not None:
        merged = df.copy()
        merged.reset_index()
        merged = pd.merge(
            merged, iptw_df.loc[:, ["patient_id", "iptw_score"]], on="patient_id"
        )
        iptw_score = merged.iptw_score.values
    else:
        iptw_score = None
    return (
        xgb.DMatrix(df.values, label=labels.values, weight=iptw_score),
        df.values,
        labels.values,
        iptw_score,
    )  # type: ignore


METRICS = {"auc20": calc_auc20, "prauc": calc_prauc}


def train(
    train_data: Path = Argument(..., help="Path of train dataset"),
    val_data: Path = Argument(..., help="Path of val dataset"),
    test_data: Path = Argument(..., help="Path of test dataset"),
    iptw_path: Path = Argument(..., help="Path of dataframe with IPTW"),
    output_dir: Path = Argument(..., help="Output directory"),
    valid_ids: Optional[Path] = Option(
        None, help="Path of text file with patient ids (one per line)"
    ),
    endpoint_column: str = Option("arm", help="Name of selected endpoint"),
    metric: str = Option("auc20", help="Metric for optimization (auc, auc_20)"),
    endpoints: List[str] = Option(["arm", "hospital"], help="Name of endpoint columns"),
    num_rounds: int = Option(150, help="Number of rounds for XGBoost"),
    n_trials: int = Option(30, help="Number of trials for HPO"),
    num_cores: int = Option(-1, help="Number of cores"),
    seed: int = Option(20210831),
    log_dir: Path = Option(Path("logs")),
):
    np.random.seed(seed)

    log_dir.mkdir(exist_ok=True, parents=True)

    handler = RotatingFileHandler(
        filename=f"logs/{datetime.now().strftime('%Y%m%d')}_xgboost-training.log",
        maxBytes=int(1e7),
        backupCount=5,
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    model_dir = output_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if iptw_path is not None:
        iptw_df = pd.read_csv(str(iptw_path))
    else:
        iptw_df = None

    ids = read_id_file(str(valid_ids)) if valid_ids is not None else None

    dtrain, x_train, y_train, train_iptw = prepare_df(
        pd.read_csv(train_data, index_col="patient_id"),
        endpoint_column,
        endpoints,
        iptw_df=iptw_df,
        valid_ids=ids,
    )
    dval, x_val, y_val, val_iptw = prepare_df(
        pd.read_csv(val_data, index_col="patient_id"),
        endpoint_column,
        endpoints,
        iptw_df=iptw_df,
        valid_ids=ids,
    )
    dtest, x_test, y_test, _ = prepare_df(
        pd.read_csv(test_data, index_col="patient_id"),
        endpoint_column,
        endpoints,
        iptw_df=iptw_df,
        valid_ids=ids,
    )

    def objective(trial):
        logger.debug("***** Start new trial *****")

        param = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1.0),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 14),
            "nthread": num_cores,
        }

        logger.debug(param)

        # Add a callback for pruning.
        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial,
            f"validation-{metric}",
        )
        early_stop = xgb.callback.EarlyStopping(
            rounds=40,
            metric_name="logloss",
            data_name="validation",
            maximize=False,
        )
        bst = xgb.train(
            param,
            dtrain,
            num_rounds,
            feval=METRICS[metric],
            evals=[(dval, "validation")],
            callbacks=[early_stop, pruning_callback],
        )
        preds = bst.predict(dval)
        logger.debug("***** Final score for round *****")
        return METRICS[metric](preds, dval)[1]

    study = optuna.create_study(
        storage=f"sqlite:///{str(output_dir)}/optuna.db",
        direction="maximize",
        study_name=f"xg{metric}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(max_resource=num_rounds),
    )
    trial_count = Counter([trial.state.name for trial in study.trials])
    already_trained = trial_count["COMPLETE"] + trial_count["PRUNED"]
    n_to_train = n_trials - already_trained
    logger.info(f"Train for {n_to_train} trials.")

    if n_to_train > 0:
        study.optimize(objective, n_trials=n_to_train)

    best_param = study.best_params
    best_param["objective"] = "binary:logistic"
    best_param["booster"] = "gbtree"
    best_param["eval_metric"] = "logloss"
    print(best_param)
    intermediate_values = list(study.best_trial.intermediate_values.values())
    boost_rounds = np.argmax(intermediate_values)

    whole_training = xgb.DMatrix(
        np.concatenate([x_train, x_val]),
        label=np.concatenate([y_train, y_val]),
        weight=np.concatenate([train_iptw, val_iptw])
        if train_iptw is not None and val_iptw is not None
        else None,
    )
    clf = xgb.train(
        best_param,
        whole_training,
        num_boost_round=int(boost_rounds),
    )
    logger.info("Trained final model")
    probs = clf.predict(dtest)
    auc = sklearn.metrics.roc_auc_score(dtest.get_label(), probs)
    preds = np.rint(probs)
    labels = dtest.get_label().astype(bool)
    rprt = sklearn.metrics.classification_report(labels, preds)

    threshold_scores = []
    for th in np.linspace(0, 1, num=1000):
        th_preds = np.where(probs > th, True, False)
        pr, rc, f1, _ = sklearn.metrics.precision_recall_fscore_support(
            y_true=labels, y_pred=th_preds, average="binary", zero_division=0
        )
        threshold_scores.append(
            {"threshold": th, "f1": f1, "precision": pr, "recall": rc}
        )
    scores = pd.DataFrame(threshold_scores).melt(id_vars="threshold")
    p = sns.lineplot(data=scores, x="threshold", y="value", hue="variable")
    plt.savefig(str(output_dir / "scores.png"), dpi=200)
    scores.to_csv(str(output_dir / "scores.csv"))

    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    clf.save_model(os.path.join(model_dir, "model.xgb"))
    with open(os.path.join(model_dir, "config.json"), "wb") as f:
        clf.save_config()

    with open(str(output_dir / "report.txt"), "w+") as f:
        f.write(rprt)

    with open(str(output_dir / "auc.txt"), "w+") as f:
        f.write(str(auc))


if __name__ == "__main__":
    typer.run(train)
