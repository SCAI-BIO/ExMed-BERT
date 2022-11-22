#!/usr/bin/env python
import os
import re
import sys

from exmed_bert.utils.helpers import read_id_file
from exmed_bert.utils.metrics import calc_auc20, calc_prauc

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import logging
import os
import pickle
from collections import Counter
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import FrozenSet, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.metrics as M
import typer
from typer import Argument, Option

logger = logging.getLogger(__name__)

METRICS = {"auc20": calc_auc20, "prauc": calc_prauc}


def prepare_df(
    df: pd.DataFrame,
    endpoint: str,
    available_endpoints: List[str],
    iptw_df: Optional[pd.DataFrame] = None,
    valid_ids: Optional[FrozenSet] = None,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, Union[npt.ArrayLike, None]]:
    if valid_ids is not None:
        df = df[df.index.isin(valid_ids)]

    labels = df.pop(endpoint)
    _ = [df.pop(x) for x in available_endpoints if x != endpoint]
    to_remove = [
        int(i) for i, col in enumerate(df.columns) if re.match("Unnamed.*", col)
    ]
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

    return (df.values, labels.values, iptw_score)  # type: ignore


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

    x_train, y_train, train_iptw = prepare_df(
        pd.read_csv(train_data, index_col="patient_id"),
        endpoint_column,
        endpoints,
        iptw_df=iptw_df,
        valid_ids=ids,
    )
    x_val, y_val, val_iptw = prepare_df(
        pd.read_csv(val_data, index_col="patient_id"),
        endpoint_column,
        endpoints,
        iptw_df=iptw_df,
        valid_ids=ids,
    )
    x_test, y_test, _ = prepare_df(
        pd.read_csv(test_data, index_col="patient_id"),
        endpoint_column,
        endpoints,
        iptw_df=iptw_df,
        valid_ids=ids,
    )

    def objective(trial):
        logger.debug("***** Start new trial *****")

        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 2, 200),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 150),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

        logger.debug(param)

        clf = sklearn.ensemble.RandomForestClassifier(
            random_state=seed, n_jobs=num_cores, **param
        )
        clf.fit(x_train, y_train, train_iptw)
        pred = clf.predict_proba(x_val)
        return METRICS[metric](pred[:, 1], y_val.astype(bool), False)[1]

    study = optuna.create_study(
        storage=f"sqlite:///{str(output_dir)}/optuna.db",
        direction="maximize",
        study_name=f"rf{metric}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(max_resource=num_rounds),
    )
    trial_count = Counter([trial.state.name for trial in study.trials])
    already_trained = trial_count["COMPLETE"] + trial_count["PRUNED"]
    n_to_train = n_trials - already_trained
    print(f"Train for {n_to_train} trials")
    if n_to_train > 0:
        study.optimize(objective, n_trials=n_to_train)

    best_param = study.best_params

    x_whole_training = np.concatenate([x_train, x_val])
    y_whole_training = np.concatenate([y_train, y_val])
    iptw_whole_training = (
        np.concatenate([train_iptw, val_iptw])
        if train_iptw is not None and val_iptw is not None
        else None
    )

    clf = sklearn.ensemble.RandomForestClassifier(
        random_state=seed, n_jobs=-1, **best_param
    )
    clf.fit(x_whole_training, y_whole_training, sample_weight=iptw_whole_training)
    probs = clf.predict_proba(x_test)[:, 1]

    auc = sklearn.metrics.roc_auc_score(y_test, probs)
    preds = np.rint(probs)
    labels = y_test.astype(bool)
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

    fpr, tpr, _ = M.roc_curve(y_test, probs)
    roc_auc = M.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver operating characteristic; {endpoint_column}")
    plt.legend(loc="lower right")
    plt.savefig(
        f"{output_dir}/{datetime.now().strftime('%Y%m%d')}_rf-roc.png",
        dpi=300,
    )

    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    with open(str(output_dir / "report.txt"), "w+") as f:
        f.write(rprt)

    with open(str(output_dir / "auc.txt"), "w+") as f:
        f.write(str(auc))


if __name__ == "__main__":
    typer.run(train)
