# imports ----------------------------------------------------------------------
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from multipledispatch import dispatch
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import EvalPrediction

# global vars ------------------------------------------------------------------

logger = logging.getLogger(__name__)

# functions --------------------------------------------------------------------


def compute_classification_report(
    model_output: np.ndarray, labels: np.ndarray, is_logit: bool = True
) -> pd.DataFrame:
    """Run sklearn classification report

    Args:
        model_output (np.array): Model output (logits or probabilities)
        labels (np.array): Labels
        is_logit (bool, optional): Flag if model_output are logits. Defaults to True.

    Returns:
        [type]: [description]
    """
    logger.info(f"***** Classification report *****")
    if is_logit:
        probs = sigmoid(model_output)
    else:
        probs = model_output
    predictions = np.where(probs > 0.5, 1, 0)
    results = classification_report(
        y_pred=predictions,
        y_true=labels.astype(bool),
        zero_division=0,  # type: ignore
        output_dict=True,
    )
    return pd.DataFrame(results)


def compute_metrics_merged_model(pred: EvalPrediction) -> Dict[str, Any]:
    """Compute the metrics for MLM+PLOS training with merged input"""
    code_labels, plos_labels = pred.label_ids  # type: ignore
    code_predictions, plos_predictions = pred.predictions  # type: ignore
    precision, recall, f1, _ = precision_recall_fscore_support(
        plos_labels.flatten(),
        plos_predictions.flatten(),
        zero_division=0,  # type: ignore
        average="binary",
    )  # type: ignore

    return {
        "mlm_accuracy": calc_accuracy(code_predictions, code_labels),
        "plos_accuracy": calc_accuracy(plos_predictions, plos_labels),
        "plos_f1": f1,
        "plos_precision": precision,
        "plos_recall": recall,
    }


def calc_accuracy(logits, labels):
    labels = labels.flatten()
    predictions = logits.flatten()

    idx = np.where(labels != -100)
    labels = labels[idx]
    predictions = predictions[idx]

    return accuracy_score(labels, predictions)


def compute_endpoint_metrics(pred: EvalPrediction) -> Dict[str, Any]:
    """Compute the metrics for fine-tuning

    Args:
        pred (EvalPrediction)

    Returns:
        Dict[str, float]: Dictionary with metrics
    """
    if pred.label_ids.shape[1] > 1:  # type: ignore
        raise Exception("Only implemented for binary case")

    logits = pred.predictions.reshape(-1)  # type: ignore
    labels = pred.label_ids.astype(bool).reshape(-1)  # type: ignore

    probs = sigmoid(logits)
    predictions = np.where(probs > 0.5, 1, 0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=predictions, average="binary", zero_division=0  # type: ignore
    )

    rocauc = roc_auc_score(y_true=labels, y_score=probs)
    rocauc20 = roc_auc_score(y_true=labels, y_score=probs, max_fpr=0.2)
    precision_c, recall_c, _ = precision_recall_curve(labels, probs)
    prauc = auc(recall_c, precision_c).astype(float)  # type: ignore

    return {
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "auc": rocauc,
        "auc20": rocauc20,
        "prauc": prauc,
    }  # type: ignore


@dispatch(np.ndarray, np.ndarray, bool)  # type: ignore
def calc_auc20(prediction: np.ndarray, labels: np.ndarray, apply_sigmoid=True):  # type: ignore
    """AUC20 score metric"""
    y = labels.astype(bool)
    logger.debug(f"Min {prediction.min()}, Max {prediction.max()}, Shape {prediction.shape}")  # type: ignore
    if apply_sigmoid:
        prediction = scipy.special.expit(prediction)
    score = sklearn.metrics.roc_auc_score(y, prediction.reshape(-1), max_fpr=0.2)
    logger.debug(f"Calculated AUC of {score}")
    return "auc20", score


@dispatch(np.ndarray, xgboost.DMatrix)  # type: ignore
def calc_auc20(prediction: np.ndarray, data: xgboost.DMatrix, apply_sigmoid=True):  # type: ignore
    """AUC20 score metric"""
    labels = data.get_label()
    y = labels.astype(bool)  # type: ignore
    logger.debug(f"Min {prediction.min()}, Max {prediction.max()}, Shape {prediction.shape}")  # type: ignore
    if apply_sigmoid:
        prediction = scipy.special.expit(prediction)
    score = sklearn.metrics.roc_auc_score(y, prediction.reshape(-1), max_fpr=0.2)
    logger.debug(f"Calculated AUC of {score}")
    return "auc20", score


@dispatch(np.ndarray, np.ndarray, bool)  # type: ignore
def calc_prauc(prediction: np.ndarray, labels: np.ndarray, apply_sigmoid=True):  # type: ignore
    """AUC20 score metric"""
    if len(set(labels)) == 1:
        logger.warning(f"Only one label specified")
    logger.debug(f"Min {prediction.min()}, Max {prediction.max()}, Shape {prediction.shape}")  # type: ignore
    if apply_sigmoid:
        logger.debug("Sigmoid transformation")
        prediction = scipy.special.expit(prediction)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(  # type: ignore
        labels, prediction.reshape(-1), pos_label=1
    )
    if len(precision) == 2:
        raise Exception("Metric not reliable due to one class prediction")
    score = auc(recall, precision)
    logger.debug(f"Calculated PRAUC of {score}")
    return "prauc", score
