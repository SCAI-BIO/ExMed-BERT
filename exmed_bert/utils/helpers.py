"""Various helper functions"""

# imports ----------------------------------------------------------------------
from typing import FrozenSet, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import xgboost as xgb
from psmpy import PsmPy
from psmpy_mod import PsmPy
from sklearn.model_selection import train_test_split
from transformers import PretrainedConfig
from xgboost import DMatrix

from exmed_bert.data.data_exceptions import UndefinedEndpoint
from exmed_bert.data.encoding import CodeDict
from exmed_bert.models import (
    CombinedModelForSequenceClassification,
    ExMedBertConfig,
    ExMedBertForSequenceClassification,
)
from exmed_bert.models.config import CombinedConfig

# functions --------------------------------------------------------------------


def calculate_iptw_values(
    dataframe: pd.DataFrame,
    endpoint: str,
    index: str,
    exclude: List[str] = [],
    psm: Optional[PsmPy] = None,
) -> Tuple[pd.DataFrame, PsmPy]:
    """Calculate iptw values for given data

    Args:
        dataframe (pd.DataFrame): Dataframe with necessary variables
        endpoint (str): endpoint columns
        index (str): index column
        exclude (List[str], optional): Columns to exclude. Defaults to [].
        psm (PsmPy, optional): PsmPy instance with fitted model

    Returns:
        pd.DataFrame: _description_
    """
    # FIXME: Use custom psmpy implementation
    if psm is None:
        trained_psm = PsmPy(dataframe, treatment=endpoint, indx=index, exclude=exclude)
        trained_psm.logistic_ps(balance=False)
        out = trained_psm.predicted_data
    else:
        trained_psm = psm
        out = trained_psm.apply_fitted_model(dataframe)
    iptw_scores = []
    out = psm.predicted_data
    for i, row in out.iterrows():
        positive = row[endpoint] == 1
        if positive:
            iptw_scores.append(1 / row["propensity_score"])
        else:
            iptw_scores.append(1 / (1 - row["propensity_score"]))
    out["iptw_score"] = iptw_scores
    return out, trained_psm


def read_id_file(path: str) -> FrozenSet[int]:
    """Read file with patient ids

    Args:
        path (str): path to input file with one patient_id per line

    Returns:
        FrozenSet[int]: Set with patient ids
    """
    with open(path, "r") as f:
        return frozenset([int(x) for x in f.read().splitlines()])


def calculate_class_weights(dataset: "PatientDataset") -> npt.NDArray:  # type: ignore
    """Calculates class weights for the different endpoints

    Args:
        dataset (PatientDataset):

    Returns:
        npt.NDArray: Array with class weights per endpoint
    """

    def _get_labels(index: int) -> torch.LongTensor:
        patient, path = dataset.get_patient(index)
        if patient.endpoint_labels is not None:
            return patient.endpoint_labels
        else:
            raise UndefinedEndpoint

    num_true_classes = torch.stack(
        [_get_labels(i) for i in range(len(dataset))],
        dim=0,
    ).sum(dim=0)
    num_neg_classes = (
        torch.tensor([len(dataset)] * len(num_true_classes)) - num_true_classes
    )
    class_weights = (num_neg_classes / num_true_classes).numpy()
    return class_weights


def prepare_df(
    df: pd.DataFrame, endpoint: str
) -> tuple[DMatrix, npt.NDArray[np.int32], npt.NDArray[np.int32],]:
    """Prepare data for tree based methods

    Args:
        df (pd.DataFrame): Preprocessed data for tree-based classifier
        endpoint (str): Name of endpoint for prediction

    Returns:
        Tuple[xgb.DMatrix, pd.DataFrame, pd.DataFrame]
    """
    labels = df.pop(endpoint)
    # TODO: find better solution for endpoint definition
    _ = [df.pop(x) for x in ["arm", "hospitalization"] if x != endpoint]

    return xgb.DMatrix(df.values, label=labels.values), df.values, labels.values  # type: ignore


def get_splits(
    test_size: float,
    val_size: float,
    endpoint_labels: npt.NDArray,
    num_samples: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Calculate stratified train/val/test splits

    Args:
        test_size (float): Ratio for test set
        val_size (float): Ratio for validation set
        endpoint_labels (np.array): Endpoint labels
        num_samples (int): Number of samples

    Returns:
        Tuple[List[int], List[int], List[int]]: Indices for the three splits
    """
    patient_idx = list(range(num_samples))

    train_patients, test_patients = train_test_split(
        patient_idx, test_size=test_size, stratify=endpoint_labels
    )
    num_val = int(val_size * num_samples)
    ratio_of_test = num_val / len(train_patients)
    train_patients, val_patients = train_test_split(
        train_patients, test_size=ratio_of_test
    )

    return train_patients, val_patients, test_patients


# class definitions ------------------------------------------------------------


class ModelLoader(object):
    """Helper to load model during hyperparameter optimization"""

    def __init__(
        self,
        path: str,
        config: Union[ExMedBertConfig, PretrainedConfig, CombinedConfig],
        code_embed: Optional[CodeDict] = None,
        extended: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.path = path
        self.code_embed = code_embed
        self.model_func = (
            ExMedBertForSequenceClassification
            if not extended
            else CombinedModelForSequenceClassification
        )

    def __call__(
        self,
    ) -> Union[
        ExMedBertForSequenceClassification, CombinedModelForSequenceClassification
    ]:
        return self.model_func.from_pretrained(
            self.path,
            config=self.config,
            code_embed=self.code_embed,
        )  # type: ignore
