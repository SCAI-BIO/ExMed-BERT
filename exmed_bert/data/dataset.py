# Code for IBM dataset class

# imports ----------------------------------------------------------------------

import copy
import logging
import math
import os
import random
import sys
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import joblib  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from psmpy_mod import PsmPy
from torch import Tensor
from torch.utils.data import Dataset

import exmed_bert
from exmed_bert.data.data_exceptions import (
    MisconfiguredDataset,
    NonePatient,
    UndefinedEndpoint,
    UndefinedPatients,
)
from exmed_bert.data.encoding import AgeDict, CodeDict, EndpointDict, SexDict, StateDict
from exmed_bert.data.patient import Patient
from exmed_bert.utils.helpers import calculate_iptw_values, get_splits

# global vars ------------------------------------------------------------------

logger = logging.getLogger(__name__)
# TODO: remove in final version
sys.modules["src"] = exmed_bert
sys.modules["medbert_int"] = exmed_bert

# functions --------------------------------------------------------------------


# class definitions ------------------------------------------------------------


class PatientDataset(Dataset):  # type: ignore
    """Pytorch Dataset with the processed patient data"""

    def __init__(
        self,
        code_embed: CodeDict,
        age_embed: AgeDict,
        sex_embed: SexDict,
        state_embed: StateDict,
        endpoint_dict: EndpointDict,
        patient_paths: Optional[List[str]] = None,
        max_length: int = 32,
        do_eval: bool = True,
        mask_substances: bool = True,
        dataset_path: Optional[str] = None,
        patients: Optional[List[Patient]] = None,
        dynamic_masking: bool = False,
        min_unmasked: int = 1,
        max_masked: int = 20,
        masked_lm_prob: float = 0.15,
    ) -> None:
        """Initialize PatientDataset

        Args:
            code_embed (CodeDict): CodeDict instance
            age_embed (AgeDict): AgeDict instance
            sex_embed (SexDict): SexDict instance
            state_embed (StateDict): StateDict instance
            endpoint_dict (EndpointDict): EndpointDict instance
            patient_paths (Optional[List[str]], optional): List of patient paths. Only required if patients are not provided. Defaults to None.
            max_length (int, optional): maximum sequence length. Defaults to 32.
            do_eval (bool, optional): Indicate if the dataset is meant for evaluation. This affects the patient output. Defaults to True.
            mask_substances (bool, optional): Indicate whether substances should be masked as well. Otherwise only diagnoses are masked. Defaults to True.
            dataset_path (Optional[str], optional): Path of dataset. Required if patients in RAM should be saved. Defaults to None.
            patients (Optional[List[Patient]], optional): List of Patient instances. Defaults to None.
            dynamic_masking (bool, optional): Indicate whether masking should happen dynamically or statically. Defaults to False.
            min_unmasked (int, optional): Minimum number of unmasked tokens. Defaults to 1.
            max_masked (int, optional): Maximum number of masked tokens. Defaults to 20.
            masked_lm_prob (float, optional): Probability of masking. Defaults to 0.15.

        Raises:
            Exception: Indicates misconfiguration of dataset (e.g., no paths and patients)
        """
        super().__init__()
        if patient_paths is None and patients is None:
            raise Exception

        # Dataset and relative location to patient files
        self.dataset_path = dataset_path
        self.patient_paths = patient_paths
        self.patients = patients
        self.loaded_to_ram = False if patients is None else True

        # Embeddings
        self.code_embed = code_embed
        self.age_embed = age_embed
        self.sex_embed = sex_embed
        self.state_embed = state_embed

        # Finetuning related
        self.endpoint_dict = endpoint_dict
        self.endpoint_index = None
        self.selected_endpoint = None

        # Configuration
        self.max_length = max_length
        self.eval = do_eval
        self.mask_substances = mask_substances
        self.dynamic_masking = dynamic_masking
        self.min_unmasked: int = min_unmasked
        self.max_masked: int = max_masked
        self.masked_lm_prob = masked_lm_prob
        self.psm: Optional[PsmPy] = None
        logger.info("PatientDataset initialization finished")

    def __add__(self, other: "PatientDataset") -> "PatientDataset":  # type: ignore
        """Combine two datasets

        Args:
            other (PatientDataset): Dataset to be added.

        Raises:
            MisconfiguredDataset

        Returns:
            PatientDataset: Combined dataset
        """
        if not self.loaded_to_ram and self.dataset_path != other.dataset_path:
            raise MisconfiguredDataset("Must be in the same directory or loaded in RAM")

        assert self.code_embed == other.code_embed
        assert self.age_embed == other.age_embed
        assert self.sex_embed == other.sex_embed
        assert self.state_embed == other.state_embed
        assert self.endpoint_dict == other.endpoint_dict

        if self.patient_paths is not None and other.patient_paths is not None:
            self.patient_paths.extend(other.patient_paths)
        elif other.patient_paths is not None and self.patient_paths is None:
            self.patient_paths = other.patient_paths

        if self.patients is not None and other.patients is not None:
            self.patients.extend(other.patients)  # type: ignore
        elif other.patients is not None and self.patients is None:
            self.patients = other.patients

        return self

    @property
    def stats(self) -> Dict[str, Union[float, int]]:
        """Generate a dictionary with dataset stats

        Raises:
            UndefinedEndpoint

        Returns:
            Dict[str, Union[float, int]]: Dataset statistics (sequence length, number of patients, endpoint)
        """

        # gather data
        endpoints: List[Tensor] = []
        sequence_lengths: List[int] = []
        for i in range(len(self)):  # type: ignore
            p = (self.get_patient(i))[0]
            if p.endpoint_labels is None or p.unpadded_length is None:
                raise UndefinedEndpoint(
                    "Endpoint labels have not been provided for patients"
                )
            endpoints.append(p.endpoint_labels)
            sequence_lengths.append(p.unpadded_length)

        # process endpoints
        endpoint_matrix = torch.stack(endpoints)
        column_sum = torch.sum(endpoint_matrix, dim=0)
        endpoint_dict = {
            endpoint: float(count)
            for endpoint, count in zip(self.endpoint_dict.endpoints, column_sum)
        }

        # stats for sequence length
        sequence_lengths_array: npt.NDArray[np.int32] = np.array(sequence_lengths)  # type: ignore
        sequence_dict = {
            "sl_min": float(np.min(sequence_lengths_array)),  # type: ignore
            "sl_q1": float(np.quantile(sequence_lengths_array, 0.25)),  # type: ignore
            "sl_median": float(np.quantile(sequence_lengths_array, 0.5)),  # type: ignore
            "sl_mean": float(np.round(np.mean(sequence_lengths_array), 0)),  # type: ignore
            "sl_q3": float(np.quantile(sequence_lengths_array, 0.75)),  # type: ignore
            "sl_max": float(np.max(sequence_lengths_array)),  # type: ignore
        }
        return {"num_patients": len(self), **endpoint_dict, **sequence_dict}

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Loads patient and returns its data

        Args:
            index (int): Index of patient

        Raises:
            UndefinedEndpoint

        Returns:
            Dict[str, Tensor]: Patient data
        """
        logger.debug(f"Get data for patient {index}")
        if not self.eval and self.endpoint_index is None:
            raise UndefinedEndpoint("Set endpoint index first.")

        # FIXME: replace evaluate with pretrain
        patient_data = (self.get_patient(index))[0].get_patient_data(
            evaluate=self.eval,
            endpoint_index=self.endpoint_index,  # type: ignore
            mask_dynamically=self.dynamic_masking,
            min_unmasked=self.min_unmasked,
            max_masked=self.max_masked,
            masked_lm_prob=self.masked_lm_prob,
            code_embed=self.code_embed,
            mask_drugs=self.mask_substances,
        )
        return patient_data

    def __len__(self) -> int:
        """Get the number of Patients in the dataset

        Returns:
            int: Size of the dataset
        """
        return len(self.patients) if self.loaded_to_ram else len(self.patient_paths)  # type: ignore

    def endpoints(self, selected_endpoints: List[str]):
        """Sets the endpoints for fine-tuning

        This method can be used to set the endpoint label which should currently
        be returned by __getitem__.


        Args:
            selected_endpoints (List[str]): List of endpoint names
        """

        available_endpoints = self.endpoint_dict.endpoints
        endpoint_index = torch.tensor(
            [i for i, x in enumerate(available_endpoints) if x in selected_endpoints]
        )
        available_endpoints = list(available_endpoints)[endpoint_index]
        if len(endpoint_index.view(-1)) == 1:
            self.selected_endpoint = available_endpoints
        else:
            raise UndefinedEndpoint()
        self.endpoint_index = endpoint_index

    def patient_to_df(self, idx: int) -> pd.DataFrame:
        """Return a dataframe with the representations for a certain patient

        Args:
            idx (int): Index of the patient in the dataset (NOT patient_id)

        Raises:
            MisconfiguredDataset

        Returns:
            pd.DataFrame: DataFrame with patient data
        """

        if not self.loaded_to_ram and self.dataset_path is None:
            logger.warning("Dataset path must be set")
            raise MisconfiguredDataset("Dataset path is not set")

        if not self.loaded_to_ram:
            patient: Patient = joblib.load(
                os.path.join(self.dataset_path, self.patient_paths[idx])  # type: ignore
            )
        else:
            patient = self.patients[idx]  # type: ignore

        return patient.to_df(
            code_embed=self.code_embed,
            age_embed=self.age_embed,
            sex_embed=self.sex_embed,
            state_embed=self.state_embed,
            dynamic_masking=self.dynamic_masking,
            min_unmasked=self.min_unmasked,
            max_masked=self.max_masked,
            mask_drugs=self.mask_substances,
            masked_lm_prob=self.masked_lm_prob,
        )

    def get_patient(self, idx: int, with_path: bool = False) -> Tuple[Patient, str]:
        """Gets the patient instance

        Depending on the location data - either in memory or saved in a joblib file -
        this method gets the instance and returns it.

        Args:
            idx (int): Index of the patient in the dataset (NOT patient_id)
            with_path (bool, optional): Flag if the path should be returned. Defaults to False.

        Raises:
            MisconfiguredDataset
            NonePatient

        Returns:
            Union[Patient, Tuple[Patient, str]]: Patient instance with optional path
        """
        if not self.loaded_to_ram and self.dataset_path is None:
            logger.warning("Dataset path must be set")
            raise MisconfiguredDataset("Dataset path is not set")

        if self.loaded_to_ram and not with_path:
            patient_instance: Patient = self.patients[idx]  # type: ignore
            if patient_instance is not None:
                return patient_instance, "na"
            else:
                raise NonePatient("Unexpectedly got None instead of patient instance.")

        patient_path: str = os.path.join(self.dataset_path, self.patient_paths[idx])  # type: ignore
        if self.loaded_to_ram and with_path:
            patient_instance: Patient = self.patients[idx]  # type: ignore
            return patient_instance, patient_path
        elif not with_path:
            return joblib.load(patient_path), "na"
        else:
            return joblib.load(patient_path), patient_path

    def train_val_test_split(
        self, test_size: float = 0.2, val_size: float = 0.1
    ) -> Tuple["PatientDataset", "PatientDataset", "PatientDataset"]:
        """Splits the dataset into train, val, and test set

        Args:
            test_size (float, optional): Split ratio for test set. Defaults to 0.2.
            val_size (float, optional): Split ratio for validation set. Defaults to 0.1.

        Raises:
            UndefinedEndpoint

        Returns:
            Tuple["PatientDataset", "PatientDataset", "PatientDataset"]: Tuple with PatientDatasets
        """

        def _get_ep_labels(index: int) -> npt.ArrayLike:
            p = (self.get_patient(index))[0]
            if p.endpoint_labels is None:
                raise UndefinedEndpoint("Endpoint labels have not been defined")
            return p.endpoint_labels.numpy()

        endpoint_labels = np.stack([_get_ep_labels(i) for i in range(len(self))])
        num_samples = len(self)
        train_patients, val_patients, test_patients = get_splits(
            test_size=test_size,
            val_size=val_size,
            endpoint_labels=endpoint_labels,
            num_samples=num_samples,
        )

        train_data = self.subset_by_ids(train_patients)
        val_data = self.subset_by_ids(val_patients)
        test_data = self.subset_by_ids(test_patients)
        logger.info(
            f"Train/Val/Test datasets were generated ({len(train_data)}, {len(val_data)}, {len(test_data)})"
        )
        return train_data, val_data, test_data

    def subset_by_ids(self, idx: List[int]) -> "PatientDataset":
        """
        Subset by a list of indices

        Args:
            idx (List[int]): List of integers, indicating which patients should be included

        Returns: Subset of the PatientDataset

        """
        return PatientDataset(
            patient_paths=[self.patient_paths[i] for i in idx]
            if self.patient_paths is not None
            else None,
            code_embed=self.code_embed,
            age_embed=self.age_embed,
            sex_embed=self.sex_embed,
            state_embed=self.state_embed,
            endpoint_dict=self.endpoint_dict,
            max_length=self.max_length,
            do_eval=self.eval,
            dataset_path=self.dataset_path,
            patients=[self.patients[i] for i in idx]
            if self.patients is not None
            else None,
            dynamic_masking=self.dynamic_masking,
            min_unmasked=self.min_unmasked,
            max_masked=self.max_masked,
            masked_lm_prob=self.masked_lm_prob,
            mask_substances=self.mask_substances,
        )

    def subset(
        self, validation_ratio: float
    ) -> Tuple["PatientDataset", "PatientDataset"]:
        """Generates a training and validation dataset.

        Args:
            validation_ratio (float):

        Raises:
            UndefinedPatients

        Returns:
            Tuple[PatientDataset, PatientDataset] = training_set, validation_set
        """
        if self.patient_paths is None:
            raise UndefinedPatients("Patient paths not defined")

        patient_paths, val_patients = self._get_validation_patients(validation_ratio)

        return (
            PatientDataset(
                patient_paths=patient_paths,
                code_embed=self.code_embed,
                age_embed=self.age_embed,
                sex_embed=self.sex_embed,
                state_embed=self.state_embed,
                max_length=self.max_length,
                do_eval=self.eval,
                dataset_path=self.dataset_path,
                dynamic_masking=self.dynamic_masking,
                endpoint_dict=self.endpoint_dict,
                mask_substances=self.mask_substances,
                min_unmasked=self.min_unmasked,
                max_masked=self.max_masked,
                masked_lm_prob=self.masked_lm_prob,
            ),
            PatientDataset(
                patient_paths=val_patients,
                code_embed=self.code_embed,
                age_embed=self.age_embed,
                sex_embed=self.sex_embed,
                state_embed=self.state_embed,
                max_length=self.max_length,
                do_eval=self.eval,
                dataset_path=self.dataset_path,
                dynamic_masking=self.dynamic_masking,
                mask_substances=self.mask_substances,
                endpoint_dict=self.endpoint_dict,
                min_unmasked=self.min_unmasked,
                max_masked=self.max_masked,
                masked_lm_prob=self.masked_lm_prob,
            ),
        )

    def _get_validation_patients(
        self, validation_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """Helper method to split data into training and validation set.

        Args:
            validation_ratio (float): Ratio of patients used for validation.

        Raises:
            MisconfiguredDataset

        Returns:
            Tuple[List[str], List[str]]: Paths of training and validation paths
        """
        num_validation = math.ceil(validation_ratio * len(self.patient_paths))  # type: ignore
        if self.patient_paths is None:
            raise MisconfiguredDataset("Please set patient paths before")

        patient_paths = copy.deepcopy(self.patient_paths)
        val_patients: List[str] = []
        while len(val_patients) < num_validation:
            val_patients.append(
                patient_paths.pop(random.randint(0, len(patient_paths) - 1))  # type: ignore
            )
        return patient_paths, val_patients

    def save_dataset(
        self, path: str, with_patients: bool = False, do_copy: bool = True
    ):
        """Save dataset to pytorch file

        Args:
            path (str): Path where dataset should be saved
            with_patients (bool, optional): If Patients should be saved in memory or as joblib files. Defaults to False.
            do_copy (bool, optional): Defaults to True.
        """
        dataset = copy.deepcopy(self) if do_copy else self
        dataset.dataset_path = "/".join(path.split("/")[0:-1])
        # Remove loaded patients from dataset to save space; still stored in patient_paths
        if not with_patients:
            if self.patient_paths is None:
                dataset._save_patients(f"{path}/patients")
            dataset.patients = None
            dataset.loaded_to_ram = False
        else:
            dataset.patient_paths = None
            dataset.loaded_to_ram = True
        dataset.endpoint_index = None
        torch.save(dataset, path)

    @classmethod
    def load_dataset(cls, path: str, to_ram: bool = False) -> "PatientDataset":
        """Load dataset from pytorch file

        Returns:
            PatientDataset
        """
        dataset = torch.load(path)
        dataset.dataset_path = "/".join(path.split("/")[0:-1])

        if dataset.patients is not None:
            logger.info("Patients are loaded in RAM")
        elif to_ram:
            dataset.load_to_ram()
        else:
            dataset.loaded_to_ram = False

        return dataset

    def _save_patients(self, path: str):
        """Helper function to save patient instances as joblib file

        Args:
            path (str): Path where patients should be saved

        Raises:
            Exception
            MisconfiguredDataset
        """
        if self.patient_paths is not None:
            raise Exception("Patients were already saved")

        if self.patients is None:
            raise MisconfiguredDataset("No patients available")

        self.patient_paths = []
        Path(path).mkdir(parents=True, exist_ok=True)
        for patient in self.patients:
            existing_files = glob(f"{path}/{patient.patient_id}_*.joblib")
            num = 0
            if len(existing_files) > 0:
                num = (
                    max(
                        [
                            int(file.split("_")[-1].split(".")[0])
                            for file in existing_files
                        ]
                    )
                    + 1
                )

            pat_path = f"{path}/{patient.patient_id}_{num}.joblib"
            joblib.dump(patient, pat_path)
            self.patient_paths.append(pat_path)

    def load_to_ram(self, threads: int = 100):
        """Helper function to load joblib files in memory

        Args:
            threads (int, optional): Number of threads used to load the data into RAM. Defaults to 100.
        """
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(threads) as p:
            self.patients = [
                p
                for p in p.map(lambda x: self.get_patient(x)[0], list(range(len(self))))
            ]

        self.loaded_to_ram = True


class ExtendedPatientDataset(PatientDataset):
    """Combined Dataset for quantitative clinical data and normal medical history"""

    def __init__(
        self,
        code_embed: CodeDict,
        age_embed: AgeDict,
        sex_embed: SexDict,
        state_embed: StateDict,
        endpoint_dict: EndpointDict,
        patient_paths: Optional[List[str]] = None,
        max_length: int = 32,
        do_eval: bool = True,
        mask_substances: bool = True,
        dataset_path: Optional[str] = None,
        patients: Optional[List[Patient]] = None,
        dynamic_masking: bool = False,
        min_unmasked: int = 1,
        max_masked: int = 20,
        masked_lm_prob: float = 0.15,
        obs_df: Optional[pd.DataFrame] = None,
        iptw_df: Optional[pd.DataFrame] = None,
        selected_endpoint: Optional[str] = None,
    ) -> None:
        """Initialize Dataset

        Args:
             code_embed (CodeDict): CodeDict instance
            age_embed (AgeDict): AgeDict instance
            sex_embed (SexDict): SexDict instance
            state_embed (StateDict): StateDict instance
            endpoint_dict (EndpointDict): EndpointDict instance
            patient_paths (Optional[List[str]], optional): List of patient paths. Only required if patients are not provided. Defaults to None.
            max_length (int, optional): maximum sequence length. Defaults to 32.
            do_eval (bool, optional): Indicate if the dataset is meant for evaluation. This affects the patient output. Defaults to True.
            mask_substances (bool, optional): Indicate whether substances should be masked as well. Otherwise only diagnoses are masked. Defaults to True.
            dataset_path (Optional[str], optional): Path of dataset. Required if patients in RAM should be saved. Defaults to None.
            patients (Optional[List[Patient]], optional): List of Patient instances. Defaults to None.
            dynamic_masking (bool, optional): Indicate whether masking should happen dynamically or statically. Defaults to False.
            min_unmasked (int, optional): Minimum number of unmasked tokens. Defaults to 1.
            max_masked (int, optional): Maximum number of masked tokens. Defaults to 20.
            masked_lm_prob (float, optional): Probability of masking. Defaults to 0.15.
            obs_df (Optional[pd.DataFrame], optional): Dataframe with quantitative clinical data. Defaults to None.
            iptw_df (Optional[pd.DataFrame], optional): Dataframe with IPTWs. Defaults to None.
            selected_endpoint (Optional[str], optional): String indicating selected endpoint. Defaults to None.
        """
        super().__init__(
            code_embed=code_embed,
            age_embed=age_embed,
            sex_embed=sex_embed,
            state_embed=state_embed,
            endpoint_dict=endpoint_dict,
            patient_paths=patient_paths,
            max_length=max_length,
            do_eval=do_eval,
            mask_substances=mask_substances,
            dataset_path=dataset_path,
            patients=patients,
            dynamic_masking=dynamic_masking,
            min_unmasked=min_unmasked,
            max_masked=max_masked,
            masked_lm_prob=masked_lm_prob,
        )
        self.obs_df = obs_df
        self.iptw_df = iptw_df
        if selected_endpoint is not None:
            self.endpoints([selected_endpoint])

    def subset_by_ids(self, idx: List[int]) -> "ExtendedPatientDataset":
        """
        Subset by a list of indices

        Args:
            idx (List[int]): List of integers, indicating which patients should be included

        Returns: Subset of the ExtendedPatientDataset

        """
        selection = ExtendedPatientDataset(
            patient_paths=[self.patient_paths[i] for i in idx]
            if self.patient_paths is not None
            else None,
            code_embed=self.code_embed,
            age_embed=self.age_embed,
            sex_embed=self.sex_embed,
            state_embed=self.state_embed,
            endpoint_dict=self.endpoint_dict,
            max_length=self.max_length,
            do_eval=self.eval,
            dataset_path=self.dataset_path,
            patients=[self.patients[i] for i in idx]
            if self.patients is not None
            else None,
            dynamic_masking=self.dynamic_masking,
            min_unmasked=self.min_unmasked,
            max_masked=self.max_masked,
            masked_lm_prob=self.masked_lm_prob,
            mask_substances=self.mask_substances,
            obs_df=self.obs_df,
            iptw_df=self.iptw_df,
            selected_endpoint=self.selected_endpoint,
        )
        return selection

    @classmethod
    def load_from_normal_dataset(
        cls, dataset: PatientDataset
    ) -> "ExtendedPatientDataset":
        """Initializes and ExtendedPatientDataset from a PatientDataset

        Args:
            dataset (PatientDataset)

        Returns:
            ExtendedPatientDataset
        """
        return ExtendedPatientDataset(
            patient_paths=dataset.patient_paths,
            patients=dataset.patients,
            code_embed=dataset.code_embed,
            age_embed=dataset.age_embed,
            sex_embed=dataset.sex_embed,
            state_embed=dataset.state_embed,
            max_length=dataset.max_length,
            do_eval=dataset.eval,
            dataset_path=dataset.dataset_path,
            dynamic_masking=dataset.dynamic_masking,
            endpoint_dict=dataset.endpoint_dict,
            mask_substances=dataset.mask_substances,
            min_unmasked=dataset.min_unmasked,
            max_masked=dataset.max_masked,
            masked_lm_prob=dataset.masked_lm_prob,
            selected_endpoint=dataset.selected_endpoint,
        )

    def load_observations(self, obs_df: pd.DataFrame):
        """Loads quantitative clinical data

        Args:
            obs_df (pd.DataFrame): Dataframe with quantitative clinical data
        """
        self.obs_df = obs_df

    def load_iptw(self, iptw_df: pd.DataFrame):
        """Loads dataframe with IPTWs

        Args:
            iptw_df (pd.DataFrame): Dataframe with IPTWs
        """
        self.iptw_df = iptw_df

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Loads patient and returns its data

        Args:
            index (int): Index of patient

        Raises:
            UndefinedEndpoint

        Returns:
            Dict[str, Tensor]: Patient data
        """
        logger.debug(f"Get data for patient {index}")
        if self.endpoint_index is None:
            raise UndefinedEndpoint("Endpoint index not defined")

        patient = self.get_patient(index)[0]
        patient_data = patient.get_patient_data(
            evaluate=self.eval,
            endpoint_index=self.endpoint_index,
            mask_dynamically=self.dynamic_masking,
            min_unmasked=self.min_unmasked,
            max_masked=self.max_masked,
            masked_lm_prob=self.masked_lm_prob,
            code_embed=self.code_embed,
            mask_drugs=self.mask_substances,
        )
        if self.obs_df is not None:
            patient_data["observation_input"] = torch.Tensor(
                self.obs_df.loc[patient.patient_id, :].values
            )
        if self.iptw_df is not None:
            assert len(self.endpoint_index.view(-1)) == 1, "Set endpoint first"
            patient_data["iptw_score"] = torch.tensor(
                self.iptw_df.loc[patient.patient_id, "iptw_score"]
            )
        return patient_data

    def subset(
        self, validation_ratio: float
    ) -> Tuple["ExtendedPatientDataset", "ExtendedPatientDataset"]:
        """Generates a training and validation dataset.

        Args:
            validation_ratio (float):

        Raises:
            UndefinedPatients

        Returns:
            Tuple[ExtendedPatientDataset, ExtendedPatientDataset] = training_set, validation_set
        """
        if self.patient_paths is None:
            raise UndefinedPatients("No patient paths available")

        patient_paths, val_patients = self._get_validation_patients(validation_ratio)

        return (
            ExtendedPatientDataset(
                patient_paths=patient_paths,
                code_embed=self.code_embed,
                age_embed=self.age_embed,
                sex_embed=self.sex_embed,
                state_embed=self.state_embed,
                max_length=self.max_length,
                do_eval=self.eval,
                dataset_path=self.dataset_path,
                dynamic_masking=self.dynamic_masking,
                endpoint_dict=self.endpoint_dict,
                mask_substances=self.mask_substances,
                min_unmasked=self.min_unmasked,
                max_masked=self.max_masked,
                masked_lm_prob=self.masked_lm_prob,
                obs_df=self.obs_df,
                iptw_df=self.iptw_df,
                selected_endpoint=self.selected_endpoint,
            ),
            ExtendedPatientDataset(
                patient_paths=val_patients,
                code_embed=self.code_embed,
                age_embed=self.age_embed,
                sex_embed=self.sex_embed,
                state_embed=self.state_embed,
                max_length=self.max_length,
                do_eval=self.eval,
                dataset_path=self.dataset_path,
                dynamic_masking=self.dynamic_masking,
                mask_substances=self.mask_substances,
                endpoint_dict=self.endpoint_dict,
                min_unmasked=self.min_unmasked,
                max_masked=self.max_masked,
                masked_lm_prob=self.masked_lm_prob,
                obs_df=self.obs_df,
                iptw_df=self.iptw_df,
                selected_endpoint=self.selected_endpoint,
            ),
        )

    def _patient_generator(self) -> Iterable[Patient]:
        """Helper function

        Yields:
            Patient
        """
        for i in range(len(self)):
            yield (self.get_patient(i))[0]

    def calculate_iptw(self, psm: Optional[PsmPy] = None) -> None:
        """Calculate IPTW values for dataset

        Args:
            psm (PsmPy, optional): PsmPy instance from training data

        Raises:
            UndefinedEndpoint
            MisconfiguredDataset
        """

        if self.endpoint_index is None:
            raise UndefinedEndpoint("Endpoint index must be provided in advance")

        if len(self.endpoint_index.view(-1)) != 1:
            raise MisconfiguredDataset(
                "Can only be calculated for one endpoint at a time"
            )

        logger.warning("This should only be used for the training dataset")
        ps_relevant_list = []
        for patient in self._patient_generator():
            if patient.endpoint_labels is None:
                raise UndefinedEndpoint("Endpoint labels have not been defined")

            ps_relevant_list.append(
                {
                    "birth_year": patient.birth_year,
                    "is_male": patient.sex == "Male",
                    "patient_id": patient.patient_id,
                    "endpoint": patient.endpoint_labels.index_select(
                        dim=-1, index=self.endpoint_index
                    ),
                }
            )

        ps_relevant_df = pd.DataFrame(ps_relevant_list)
        self.iptw_df, self.psm = calculate_iptw_values(
            ps_relevant_df, endpoint="endpoint", index="patient_id", psm=psm
        )
        self.iptw_df = self.iptw_df.set_index("patient_id")  # type: ignore
        logger.info("IPTW values calculated for dataset")
