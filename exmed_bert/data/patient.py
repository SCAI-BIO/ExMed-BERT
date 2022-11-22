# code for patient and dataset class
# imports

import copy
import logging
import random
from collections import deque
from dataclasses import InitVar, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from exmed_bert.data.data_exceptions import InputException, PatientNotValid
from exmed_bert.data.encoding import (
    AgeDict,
    CodeDict,
    EndpointDict,
    SexDict,
    StateDict,
    icd_to_phewas,
    rxcui_to_atc4,
)
from exmed_bert.utils.sequence_prep import (
    create_masked_lm_predictions,
    create_mlm_mask,
    mask_token_ids,
    position_idx,
    seq_padding,
)

# global vars

logger = logging.getLogger(__name__)


# functions


# class definitions


@dataclass
class Patient(object):
    """Class for the data that corresponds to one patient/enroll id"""

    patient_id: int
    diagnoses: List[str]
    drugs: List[str]
    diagnosis_dates: List[date]
    prescription_dates: List[date]
    birth_year: int
    sex: str
    patient_state: str

    # Init vars (matched by order)
    max_length: InitVar[int]
    code_embed: InitVar[CodeDict]
    sex_embed: InitVar[SexDict]
    age_embed: InitVar[AgeDict]
    state_embed: InitVar[StateDict]
    mask_drugs: InitVar[bool]
    delete_temporary_variables: InitVar[bool] = True
    split_sequence: InitVar[bool] = True
    drop_duplicates: InitVar[bool] = True
    # TODO: Remove converted_codes for clarity
    converted_codes: InitVar[bool] = False
    convert_icd_to_phewas: InitVar[bool] = True
    convert_rxcui_to_atc: InitVar[bool] = True
    keep_min_unmasked: InitVar[int] = 1
    max_masked_tokens: InitVar[int] = 20
    masked_lm_prob: InitVar[float] = 0.15
    truncate: InitVar[str] = "right"

    index_date: Optional[datetime] = None
    had_plos: Optional[bool] = None
    endpoint_labels: Optional[torch.LongTensor] = None

    dynamic_masking: bool = True
    min_observations: int = 5
    age_usage: str = "months"
    use_cls: bool = False
    use_sep: bool = False
    valid_patient: bool = True

    num_visits: Optional[int] = None
    combined_length: Optional[int] = None
    unpadded_length: Optional[int] = None

    # code related
    code_entities = None
    code_tokens = None
    remaining_data: Optional[Dict[str, List[Any]]] = None
    code_labels = None
    age_sequence = None
    sex_sequence = None
    state_sequence = None
    attention_mask = None
    positions = None
    model_input: Optional[Dict[str, Any]] = None

    def __post_init__(
        self,
        max_length: int,
        code_embed: CodeDict,
        sex_embed: SexDict,
        age_embed: AgeDict,
        state_embed: StateDict,
        mask_drugs: bool,
        delete_temporary_variables: bool,
        split_sequence: bool,
        drop_duplicates: bool,
        converted_codes: bool,
        convert_icd_to_phewas: bool,
        convert_rxcui_to_atc: bool,
        keep_min_unmasked: int,
        max_masked_tokens: int,
        masked_lm_prob: float,
        truncate: str,
    ):
        # assertions
        assert (
            self.endpoint_labels is not None or mask_drugs is not None
        ), "Specify whether drugs should be masked or not"

        if self.endpoint_labels is not None and not isinstance(
            self.endpoint_labels, torch.Tensor
        ):
            raise InputException("Endpoint labels must be provided as a tensor")

        if len(self.diagnoses) != len(self.diagnosis_dates) or len(self.drugs) != len(
            self.prescription_dates
        ):
            logger.debug(
                f"Diagnosis: {len(self.diagnoses)}, {len(self.diagnosis_dates)}"
            )
            logger.debug(f"Drugs: {len(self.drugs)}, {len(self.prescription_dates)}")
            raise InputException("Dates and codes do not match. Check input data.")

        # Test if patient has enough observations for diagnoses and substances
        combined_length = len(self.diagnosis_dates) + len(self.prescription_dates)
        if combined_length < self.min_observations:
            logger.debug(
                f"""
                Not enough diagnoses/drugs for patient {self.patient_id}
                [{len(self.diagnosis_dates)} /
                {len(self.prescription_dates) if self.prescription_dates is not None else '-'}]
                """
            )
            self.valid_patient = False
            raise PatientNotValid

        # infer optional embeddings -
        if age_embed is not None:
            self.use_age = True
        else:
            self.use_age = False

        if state_embed is not None:
            self.use_state = True
        else:
            self.use_state = False

        if self.patient_state is None:
            self.patient_state = "UNK"

        # Determine number of time points
        all_time_points = set(self.diagnosis_dates + self.prescription_dates)
        self.num_visits = len(all_time_points)

        # Sort time series data
        diag_idx = np.argsort(np.array(self.diagnosis_dates))
        self.diagnosis_dates = [self.diagnosis_dates[i] for i in diag_idx]
        self.diagnoses = [self.diagnoses[i] for i in diag_idx]

        drug_idx = np.argsort(np.array(self.prescription_dates))
        self.prescription_dates = [self.prescription_dates[i] for i in drug_idx]
        self.drugs = [self.drugs[i] for i in drug_idx]

        # prepare the patient data (either dynamic or static)
        if (
            not self.dynamic_masking
            and self.endpoint_labels is None
            and (keep_min_unmasked is None or max_masked_tokens is None)
        ):
            raise InputException(
                "Supply keep_min_unmasked and max_masked_tokens in case of static masking"
            )
        self.prepare_patient(
            max_length=max_length,
            code_embed=code_embed,
            age_embed=age_embed,
            sex_embed=sex_embed,
            state_embed=state_embed,
            split_sequence=split_sequence,
            drop_duplicates=drop_duplicates,
            converted_codes=converted_codes,
            convert_icd_to_phewas=convert_icd_to_phewas,
            convert_rxcui_to_atc=convert_rxcui_to_atc,
            dynamic_masking=self.dynamic_masking,
            keep_min_unmasked=keep_min_unmasked,
            max_masked_tokens=max_masked_tokens,
            masked_lm_prob=masked_lm_prob,
            mask_drugs=mask_drugs,
            truncate=truncate,
        )

        # delete data which is no longer needed
        if delete_temporary_variables:
            tokens_to_delete = {
                "diagnoses",
                "diagnosis_dates",
                "attention_mask",
                "age_embed",
                "age_sequence",
                "delete_temporary_variables",
                "diagnosis_tokens",
                "positions",
                "sex_sequence",
                "state_embed",
                "state_sequence",
                "time_embed",
            }

            for ele in tokens_to_delete:
                # noinspection PyBroadException
                try:
                    delattr(self, ele)
                except:
                    logger.debug(f"{ele} could not be deleted from patient.")

    def __repr__(self) -> str:
        return f"Patient: {self.patient_id}; Valid: {self.valid_patient}; Visits: {self.num_visits}"

    @staticmethod
    def combine_substance_and_code(
        diagnosis_dates: List[date],
        diagnoses: List[str],
        prescription_dates: List[date],
        drugs: List[str],
        code_embed: CodeDict,
        drop_duplicates: bool = True,
        converted_codes: bool = False,
        convert_icd_to_phewas: bool = True,
        convert_rxcui_to_atc: bool = True,
    ) -> Tuple[List[str], List[datetime]]:
        """Combine atc and phewas codes to one sequence

        Args:
            diagnosis_dates (List[date]): [description]
            diagnoses (List[str]): [description]
            prescription_dates (List[date]): [description]
            drugs (List[str]): [description]
            code_embed (CodeDict): [description]
            drop_duplicates (bool, optional): [description]. Defaults to True.
            converted_codes (bool, optional): [description]. Defaults to False.
            convert_icd_to_phewas (bool, optional): Converts icd9/icd10 codes to phecodes. Defaults to True.
            convert_rxcui_to_atc (bool, optional): Converts rxnorm ids to atc ids. Defaults to True.

        Raises:
            PatientNotValid: Validation error

        Returns:
            Tuple[List[int], List[datetime]]: List of codes, List of time points
        """

        if prescription_dates is None and diagnosis_dates is None:
            logger.error("Both date sequences are none.")
            raise PatientNotValid

        codes = []
        time_points = []
        try:
            unique_time_points = sorted(
                set(
                    np.concatenate(
                        (np.array(diagnosis_dates), np.array(prescription_dates))
                    )
                )
            )
        except:
            logger.error("Unique time points could not be found")
            logger.error(diagnosis_dates)
            logger.error(prescription_dates)
            raise PatientNotValid

        for time_point in unique_time_points:
            # processing of diagnoses --
            diagnosis_index = [
                i
                for i, diagnosis_date in enumerate(diagnosis_dates)
                if diagnosis_date == time_point
            ]
            diagnoses_at_time = [diagnoses[idx] for idx in diagnosis_index]
            if (
                not converted_codes
                and convert_icd_to_phewas
                and len(diagnoses_at_time) > 0
            ):
                # diagnoses are coded in icd => convert to phewas
                if code_embed is None:
                    raise Exception("Code dictionary is required")
                diagnoses_at_time = icd_to_phewas(
                    diagnoses_at_time, icd_to_pw=code_embed.icd_phewas_map  # type: ignore
                )

            # processing of drugs
            drug_index = [
                i
                for i, prescription_date in enumerate(prescription_dates)
                if prescription_date == time_point
            ]
            drugs_at_time = [drugs[idx] for idx in drug_index]
            if not converted_codes and convert_rxcui_to_atc and len(drugs_at_time) > 0:
                drugs_at_time = rxcui_to_atc4(
                    drugs_at_time, rx_to_atc_map=code_embed.rx_atc_map  # type: ignore
                )

            if drop_duplicates:
                diagnoses_at_time = list(set(diagnoses_at_time))
                drugs_at_time = list(set(drugs_at_time))

            codes_at_time = diagnoses_at_time + drugs_at_time
            random.shuffle(codes_at_time)
            codes.extend(codes_at_time)
            time_points.extend([time_point] * len(codes_at_time))

        return codes, time_points

    def prepare_patient(
        self,
        max_length: int,
        code_embed: CodeDict,
        age_embed: AgeDict,
        sex_embed: SexDict,
        state_embed: StateDict,
        split_sequence: bool = True,
        drop_duplicates: bool = True,
        converted_codes: bool = False,
        convert_icd_to_phewas: bool = True,
        convert_rxcui_to_atc: bool = True,
        dynamic_masking: bool = True,
        keep_min_unmasked: int = 1,
        max_masked_tokens: int = 20,
        masked_lm_prob: float = 0.15,
        mask_drugs: bool = True,
        truncate: str = "right",
    ):
        """Prepare all patient information without masking

        Args:
            max_length (int): maximum sequence length
            code_embed (CodeDict): CodeDict instance
            age_embed (AgeDict): AgeDict instance
            sex_embed (SexDict): SexDict instance
            state_embed (StateDict): StateDict instance
            split_sequence (bool, optional): Indicate whether sequences should be split. Defaults to True.
            drop_duplicates (bool, optional): Indicate whether duplicate codes should be dropped. Defaults to True.
            converted_codes (bool, optional): Indicate whether codes need conversion. Defaults to False.
            convert_icd_to_phewas (bool, optional): Indicate whether icd codes are present. Defaults to True.
            convert_rxcui_to_atc (bool, optional): Indicate whether atc codes are present. Defaults to True.
            dynamic_masking (bool, optional): Indicate whether masking should be performed dynamically. Defaults to True.
            keep_min_unmasked (int, optional): Minimum unmasked tokens. Defaults to 1.
            max_masked_tokens (int, optional): Maximum number of masked tokens in sequence. Defaults to 20.
            masked_lm_prob (float, optional): Probability of masking. Defaults to 0.15.
            mask_drugs (bool, optional): Indicate whether drugs (ATC) should be masked. Defaults to True.
            truncate (str, optional): Specify with "right" or "left" where sequences should be truncated. Defaults to "right".

        Raises:
            PatientNotValid
            Exception
        """
        # TODO: simplify conversion part in function arguments

        try:
            (self.codes, self.time_points,) = self.combine_substance_and_code(
                diagnosis_dates=self.diagnosis_dates,
                diagnoses=self.diagnoses,
                prescription_dates=self.prescription_dates,
                drugs=self.drugs,
                code_embed=code_embed,
                drop_duplicates=drop_duplicates,
                converted_codes=converted_codes,
                convert_icd_to_phewas=convert_icd_to_phewas,
                convert_rxcui_to_atc=convert_rxcui_to_atc,
            )
            unique_dates = sorted(set(self.time_points))
            self.num_visits = len(unique_dates)
            self.valid_patient = True
        except PatientNotValid:
            self.valid_patient = False
            logger.warning(f"Patient {self.patient_id} is not valid")
            raise

        if len(self.codes) < self.min_observations:
            logger.debug(f"Dates: {self.time_points}")
            logger.debug(f"Codes: {self.codes}")
            logger.debug(f"Min observations not given for ({self.patient_id})")
            # a patient has to have a minimum number of observations
            self.valid_patient = False
            raise PatientNotValid

        segment_information: List[int] = []
        code_sequence: List[str] = []
        age_sequence: List[Union[int, str]] = []
        self.combined_length = len(self.time_points)

        if self.use_cls:
            # CLS token is used in BEHRT, but not in Med-BERT (optional)

            code_sequence = ["CLS"]

            if self.use_age:
                # Exchanged with min(age) at a later point
                age_sequence = ["UNK"]

            self.combined_length = len(unique_dates) + 1 + len(self.codes)

        # Remaining data can be stored to generate multiple patient instances if
        # splitting is activated
        self.remaining_data = {}

        # Prepare the time series data
        logger.debug(f"***** Start processing time series ({self.patient_id})*****")
        segment_wise_information = []
        for time in unique_dates:

            # get events at the given time point
            idx = [i for i, x in enumerate(self.time_points) if x == time]
            codes_at_time = [self.codes[i] for i in idx]

            logger.debug(f"Process time point {time} ({self.patient_id})")
            # Add events to input sequence if max_length is not exceeded

            code_sequence.extend(codes_at_time)
            segment_wise_information.append(codes_at_time)

            if self.use_sep:
                # SEP is used in BEHRT, but not in Med-BERT
                code_sequence.append("SEP")
                length_to_add = len(codes_at_time) + 1
            else:
                length_to_add = len(codes_at_time)

            if self.use_cls and len(segment_information) == 0:
                segment_information.append(length_to_add + 1)
            else:
                segment_information.append(length_to_add)

            if self.use_age:
                if self.birth_year is not None and self.birth_year != "UNK":
                    days_alive = (time - date(self.birth_year, 1, 1)).days
                    if self.age_usage == "decimal":
                        # age in years, but with decimal precision
                        patient_age = round(days_alive / 365.24, 1)
                    elif self.age_usage == "months":
                        # age in months since birth year
                        # 365.24 / 12 = 30.44
                        patient_age = int(days_alive / 30.44)
                    elif self.age_usage == "year":
                        # integer representation
                        patient_age = int(days_alive / 365.24)
                    else:
                        raise Exception(
                            "age usage must either be decimal, months or year"
                        )
                    age_list = [patient_age] * length_to_add
                    age_sequence.extend(age_list)  # type: ignore
                else:
                    age_sequence.extend(["UNK"] * length_to_add)

        if len(code_sequence) > max_length:
            # needs truncation and split
            codes_that_remain: Deque[str] = deque([])
            dates_that_remain: Deque[datetime] = deque([])
            dates_to_keep: Deque[datetime] = deque([])
            new_segment_information: List[int] = []
            total_length = 0
            done_truncating = False

            assert len(segment_information) == len(segment_wise_information)
            assert len(segment_wise_information) == len(unique_dates)

            if truncate == "right":
                # remove from right side
                for segment, codes_for_segment, time_point in zip(
                    reversed(segment_information),
                    reversed(segment_wise_information),
                    reversed(unique_dates),
                ):
                    if total_length + segment <= max_length and not done_truncating:
                        total_length += segment
                        new_segment_information.insert(0, segment)
                        dates_to_keep.extendleft([time_point] * len(codes_for_segment))
                    else:
                        done_truncating = True
                        codes_that_remain.extendleft(reversed(codes_for_segment))
                        dates_that_remain.extendleft(
                            [time_point] * len(codes_for_segment)
                        )

                code_sequence = code_sequence[len(code_sequence) - total_length :]
                if self.use_age:
                    age_sequence = age_sequence[len(age_sequence) - total_length :]
                segment_information = new_segment_information
            elif truncate == "left":
                # remove from left side
                for segment, codes_for_segment, time_point in zip(
                    segment_information, segment_wise_information, unique_dates
                ):
                    if total_length + segment <= max_length and not done_truncating:
                        total_length += segment
                        new_segment_information.append(segment)
                        dates_to_keep.extend([time_point] * len(codes_for_segment))
                    else:
                        done_truncating = True
                        codes_that_remain.extend(codes_for_segment)
                        dates_that_remain.extend([time_point] * len(codes_for_segment))

                code_sequence = code_sequence[:total_length]
                if self.use_age:
                    age_sequence = age_sequence[:total_length]
                segment_information = new_segment_information
            else:
                raise Exception(
                    f"Sequence is to long ({len(code_sequence)}). Please use truncation"
                )

            if split_sequence:
                self.remaining_data = code_embed.split_codes_and_dates(
                    list(codes_that_remain), list(dates_that_remain)
                )

            del segment_wise_information

        if len(code_sequence) < self.min_observations:
            logger.debug(f"Dates: {self.time_points}")
            logger.debug(f"Codes: {self.codes}")
            logger.debug(f"Code sequence: {code_sequence}")
            logger.debug(f"Min observations not given for ({self.patient_id})")
            # a patient has to have a minium number of observations
            self.valid_patient = False
            raise PatientNotValid

        if self.use_cls and self.use_age:
            age_sequence[0] = min(age_sequence[1:])

        # Padding
        self.code_tokens, self.code_entities = seq_padding(
            tokens=code_sequence,
            max_len=max_length,
            embed=code_embed,
            return_entities=True,
        )
        if self.endpoint_labels is None and not dynamic_masking:
            input_ids, code_labels = self.mask_code_ids(
                code_ids=self.code_tokens
                if isinstance(self.code_tokens, torch.LongTensor)
                else torch.LongTensor(self.code_tokens),
                code_embed=code_embed,
                min_unmasked=keep_min_unmasked,
                max_masked=max_masked_tokens,
                mask_drugs=mask_drugs,
                masked_lm_prob=masked_lm_prob,
            )
            self.code_labels = code_labels
            self.code_tokens = input_ids

        # Prepare the static data
        logger.debug(f"preparing static data ({self.patient_id})")
        self.unpadded_length = len(code_sequence)

        if self.use_age:
            self.age_sequence = np.zeros(max_length)
            self.age_sequence[: self.unpadded_length] = age_embed(age_sequence)

        self.sex_sequence = np.zeros(max_length)
        self.sex_sequence[: self.unpadded_length] = sex_embed(self.sex)

        if self.use_state:
            self.state_sequence = np.zeros(max_length)
            self.state_sequence[: self.unpadded_length] = (
                state_embed(self.patient_state)
                if self.patient_state is not None
                else state_embed("UNK")
            )

        self.attention_mask = np.ones(max_length)
        self.attention_mask[self.unpadded_length :] = 0

        self.positions = position_idx(
            visit_length=segment_information, max_length=max_length
        )

        # prepare dictionary as model input
        self.model_input = {
            "input_ids": torch.LongTensor(self.code_tokens),
            "entity_ids": torch.LongTensor(self.code_entities.tolist()),
            "sex_ids": torch.LongTensor(self.sex_sequence),
            "attention_mask": torch.LongTensor(self.attention_mask),
            "position_ids": torch.LongTensor(self.positions),
        }

        if self.use_state:
            self.model_input["state_ids"] = torch.LongTensor(self.state_sequence)

        if self.use_age:
            self.model_input["age_ids"] = torch.LongTensor(self.age_sequence)

        # if not dynamic_masking:

        logger.debug(f"Finished processing ({self.patient_id})")

    def randomly_mask_data(
        self,
        code_embed: CodeDict,
        max_masked_tokens: int = 20,
        keep_min_unmasked: int = 1,
        masked_lm_prob: float = 0.15,
    ):
        """Mask phecodes and atc codes randomly

        Args:
            code_embed (CodeDict): CodeDict instance
            keep_min_unmasked (int, optional): Minimum unmasked tokens. Defaults to 1.
            max_masked_tokens (int, optional): Maximum number of masked tokens in sequence. Defaults to 20.
            masked_lm_prob (float, optional): Probability of masking. Defaults to 0.15.

        Raises:
            Exception
        """

        if self.model_input is None:
            raise Exception("Process patient data before executing this function")

        self.max_masked_tokens = max_masked_tokens  # type: ignore
        self.keep_min_unmasked = keep_min_unmasked  # type: ignore
        self.masked_lm_prob = masked_lm_prob  # type: ignore
        length = len(self.model_input["input_ids"])

        # Masks codes
        logger.debug(f"Masking diagnoses ({self.patient_id})")
        replace_phewas = {
            i: lbl for i, lbl in enumerate(self.code_labels) if lbl != -100  # type: ignore
        }
        input_ids = self.model_input["input_ids"]
        for pos, lbl in replace_phewas.items():
            input_ids[pos] = lbl
        input_ids = input_ids[input_ids != 0]
        code_sequence = code_embed.decode(input_ids)
        if isinstance(code_sequence, tuple):
            code_sequence = code_sequence[0]

        (
            masked_code_tokens,
            masked_code_positions,
            masked_code_labels,
        ) = create_masked_lm_predictions(
            code_sequence,
            masked_lm_prob=masked_lm_prob,
            max_predictions_per_seq=max_masked_tokens,
            vocab_words=code_embed.vocab,
            keep_min_unmasked=keep_min_unmasked,
        )
        self.code_labels = create_mlm_mask(  # type: ignore
            masked_code_positions,
            masked_code_labels,
            length,
            code_embed,
        )
        masked_code_tokens = seq_padding(masked_code_tokens, length, code_embed)  # type: ignore
        self.model_input["input_ids"] = torch.LongTensor(masked_code_tokens)

    def get_remaining_data(self) -> Union[Dict[str, List[Any]], None]:
        """Gets the remaining data and deletes attribute to save space"""
        remaining_data = self.remaining_data
        delattr(self, "remaining_data")
        return remaining_data

    def get_patient_data(
        self,
        evaluate: bool = False,
        endpoint_index: Optional[torch.Tensor] = None,
        code_embed: Optional[CodeDict] = None,
        mask_dynamically: bool = False,
        min_unmasked: int = 1,
        max_masked: int = 20,
        masked_lm_prob: float = 0.15,
        mask_drugs: bool = True,
    ) -> Dict[str, Tensor]:
        """Returns a dictionary with the encoded diagnoses and time points

        Args:
            evaluate (bool, optional): Specifies if labels should be returned. Defaults to True.
            endpoint_index (Optional[torch.Tensor], optional): Index for endpoint label. Defaults to None.
            code_embed (Optional[CodeDict], optional): CodeDict instance. Defaults to None.
            mask_dynamically (bool, optional): Indicate whether input should be masked dynamically. Defaults to False.
            min_unmasked (int): Minimum number of unmasked tokens
            max_masked (int): Maximum number of masked tokens
            masked_lm_prob (float, optional): Probability of masking. Defaults to 0.15.
            mask_drugs (bool, optional): Indicate whether drugs should be masked. Defaults to True.

        Returns:
            Dict[str, Tensor]: Model input
        """
        if self.model_input is None:
            raise Exception("Process data before executing this function")
        patient_data = copy.deepcopy(self.model_input)

        if not evaluate and self.endpoint_labels is not None and endpoint_index is None:
            # Finetuning scenario --> only endpoint label required
            patient_data["endpoint_labels"] = self.endpoint_labels
        elif (
            not evaluate
            and self.endpoint_labels is not None
            and endpoint_index is not None
        ):
            # Finetuning scenario --> only specific endpoint labels required
            patient_data["endpoint_labels"] = self.endpoint_labels.index_select(
                dim=-1, index=endpoint_index
            )
        elif evaluate:
            # Pretraining scenario --> mlm labels required
            if self.had_plos is not None:
                patient_data["plos_label"] = (
                    torch.LongTensor(1) if self.had_plos else torch.LongTensor(0)
                )

            if mask_dynamically:
                logger.debug("Mask input dynamically")
                if code_embed is None:
                    raise Exception("code_embed is required for dynamic masking.")
                input_ids, code_labels = self.mask_code_ids(
                    code_ids=self.model_input["input_ids"],
                    code_embed=code_embed,
                    min_unmasked=min_unmasked,
                    max_masked=max_masked,
                    mask_drugs=mask_drugs,
                    masked_lm_prob=masked_lm_prob,
                )
                patient_data["code_labels"] = code_labels
                patient_data["input_ids"] = input_ids

            else:
                patient_data["code_labels"] = self.code_labels
                patient_data["input_ids"] = self.code_tokens

        return patient_data

    @staticmethod
    def mask_code_ids(
        code_ids: torch.LongTensor,
        code_embed: CodeDict,
        min_unmasked: int,
        max_masked: int,
        mask_drugs: bool,
        masked_lm_prob: float = 0.15,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """helper function to mask codes

        Args:
            code_ids (torch.LongTensor): Tensor with code_ids
            code_embed (CodeDict): Dictionary class for codes
            min_unmasked (int): Minimum number of unmasked tokens
            max_masked (int): Maximum number of masked tokens
            mask_drugs (bool): Boolean if drugs should be masked
            masked_lm_prob (float): Probability for masking

        Returns:
            Tuple[torch.Tensor, torch.TensorType]: Tensors with masked input and labels
        """
        codes_to_ignore = {
            i for i, entity in enumerate(code_embed.entities) if entity == "default"
        }

        if not mask_drugs:
            for i, entity in enumerate(code_embed.entities):
                if entity == "atc":
                    codes_to_ignore.add(i)

        codes_to_use = set(list(range(len(code_embed)))) - codes_to_ignore

        assert len(codes_to_ignore) + len(codes_to_use) == len(code_embed)

        input_ids, code_labels = mask_token_ids(
            token_ids=code_ids,
            masked_lm_prob=masked_lm_prob,
            max_predictions_per_seq=max_masked,
            codes_to_use=list(codes_to_use),
            keep_min_unmasked=min_unmasked,
            mask_id=code_embed.labels_to_id["MASK"],
            codes_to_ignore=codes_to_ignore,
        )

        return input_ids, code_labels

    def to_df(
        self,
        code_embed: CodeDict,
        age_embed: AgeDict,
        sex_embed: SexDict,
        state_embed: StateDict,
        dynamic_masking: bool = False,
        mask_drugs: bool = True,
        min_unmasked: int = 1,
        max_masked: int = 20,
        masked_lm_prob: float = 0.15,
    ) -> pd.DataFrame:
        """Returns a dataframe object of the patient representation

        Args:
            code_embed (CodeDict): [description]
            age_embed (AgeDict): [description]
            sex_embed (SexDict): [description]
            state_embed (StateDict): [description]
            mask_drugs (bool, optional): [description]. Defaults to True.
            dynamic_masking (bool, optional): [description]. Defaults to False.
            min_unmasked ([type], optional): [description]. Defaults to None.
            max_masked ([type], optional): [description]. Defaults to None.
            masked_lm_prob (float, optional): Probability for masking. Defaults to 0.15

        Returns:
            pd.DataFrame: [description]
        """
        representation = self.get_patient_data(
            evaluate=True,
            mask_dynamically=dynamic_masking,
            code_embed=code_embed,
            min_unmasked=min_unmasked,
            max_masked=max_masked,
            masked_lm_prob=masked_lm_prob,
            mask_drugs=mask_drugs,
        )

        position = [
            x if x != 0 else "PAD" for x in representation["position_ids"].numpy()
        ]

        data = {
            "codes": code_embed.decode(representation["input_ids"]),
            "sex": sex_embed.decode(representation["sex_ids"]),
            "position": position,
        }

        if "entity_ids" in representation:
            data["entity_ids"] = [
                x
                for x in map(
                    code_embed.ids_to_entity.get, representation["entity_ids"].numpy()
                )
            ]

        if "code_labels" in representation:
            data["code_label"] = code_embed.decode(representation["code_labels"])

        if self.use_state:
            data["state"] = state_embed.decode(representation["state_ids"])

        if self.use_age:
            data["age"] = age_embed.decode(representation["age_ids"])

        if self.had_plos is not None:
            data["plos"] = [self.had_plos] * len(data["codes"])

        return pd.DataFrame(data)

    @classmethod
    def generate_demo(
        cls,
        dynamic_masking: bool,
        max_length: int,
        code_embed: CodeDict,
        sex_embed: SexDict,
        age_embed: Optional[AgeDict] = None,
        state_embed: Optional[StateDict] = None,
        use_cls=False,
        use_sep=False,
        age_usage="months",
        start: date = date(2010, 1, 1),
        end: date = date(2020, 12, 31),
        delete_temporary_variables: bool = True,
        mask_drugs: bool = True,
        num_endpoints: Optional[int] = None,
        truncate: str = "right",
    ) -> "Patient":
        """Function to generate a demo patient

        Args:
            dynamic_masking (bool): Apply dynamic masking
            max_length (int): Maximum sequence length
            code_embed (CodeDict): CodeDict instance
            sex_embed (SexDict): SexDict instance
            age_embed (Optional[AgeDict], optional): AgeDict instance. Defaults to None.
            state_embed (Optional[StateDict], optional): StateDict instance. Defaults to None.
            use_cls (bool, optional): Indicate whether the CLS token should be used. Defaults to False.
            use_sep (bool, optional): Indicate whether the sep token should be used. Defaults to False.
            age_usage (str, optional): Indicate whether age should be used. Defaults to "months".
            start (date, optional): Start date of medical history. Defaults to date(2010, 1, 1).
            end (date, optional): End date of medical history. Defaults to date(2020, 12, 31).
            delete_temporary_variables (bool, optional): Only set to false for debugging. Defaults to True.
            mask_drugs (bool, optional): Indicate whether drugs should be masked. Defaults to True.
            num_endpoints (Optional[int], optional): Specify the number of endpoints. Defaults to None.
            truncate (str, optional): "right" or "left". Defaults to "right".

        Returns:
            Patient: Random Patient instance
        """

        patient_id = random.randint(0, 500_000)
        target_length = max_length + random.randint(
            -int(0.25 * max_length), int(0.25 * max_length)
        )
        n_diag = int(0.25 * target_length) + int(random.random() * (target_length / 2))
        n_drug = target_length - n_diag

        diagnoses = random.choices(list(code_embed.phewas_codes), k=n_diag)
        drugs = random.choices(list(code_embed.atc_codes), k=n_drug)
        diagnosis_dates = cls.generate_time_series(start, end, n_diag)
        prescription_dates = cls.generate_time_series(start, end, n_drug)

        Patient.randomly_combine_time_series(diagnosis_dates, prescription_dates)

        sex = random.choice(sex_embed.vocab[3:])
        birth_year = random.randint(1920, 1950)
        state = None if state_embed is None else random.choice(state_embed.vocab[3:])

        if num_endpoints is not None:
            endpoints = [0] * num_endpoints
            for i in range(num_endpoints):
                if random.uniform(0.0, 1.0) < 0.2:
                    endpoints[i] = 1
        else:
            endpoints = None

        return Patient(
            converted_codes=True,
            dynamic_masking=dynamic_masking,
            mask_drugs=mask_drugs,
            patient_id=patient_id,
            diagnoses=diagnoses,
            diagnosis_dates=diagnosis_dates,
            prescription_dates=prescription_dates,
            drugs=drugs,
            birth_year=birth_year,
            sex=sex,
            patient_state=state,  # type: ignore
            max_length=max_length,
            code_embed=code_embed,
            sex_embed=sex_embed,
            age_embed=age_embed,  # type: ignore
            state_embed=state_embed,  # type: ignore
            use_cls=use_cls,
            use_sep=use_sep,
            age_usage=age_usage,
            delete_temporary_variables=delete_temporary_variables,
            drop_duplicates=True,
            keep_min_unmasked=1,
            max_masked_tokens=20,
            masked_lm_prob=0.33,
            endpoint_labels=torch.LongTensor(endpoints)
            if num_endpoints is not None
            else None,
            truncate=truncate,
        )

    @staticmethod
    def generate_time_series(
        start=date(2010, 1, 1), end=date(2020, 12, 31), length=512
    ) -> List[date]:
        """Helper to generate a demo patient's time series

        Args:
            start (_type_, optional): Start of time series. Defaults to date(2010, 1, 1).
            end (_type_, optional): End of time series. Defaults to date(2020, 12, 31).
            length (int, optional): Length of sequence. Defaults to 512.

        Returns:
            List[date]: List of random dates
        """

        number_of_days = (end - start).days
        difference_in_days = sorted(
            [random.randint(0, number_of_days) for _ in range(length)]
        )

        return [start + timedelta(days=d) for d in difference_in_days]

    @staticmethod
    def randomly_combine_time_series(a, b, threshold: float = 0.3):
        """Helper to create a more realistic time series representation"""

        if len(a) <= len(b):
            selected = a
            other = b
        else:
            selected = b
            other = a

        for i, prob in enumerate([random.random() for _ in range(len(selected))]):
            if prob > threshold:
                selected[i] = other[i]

    def to_dict(self, endpoint_dict: Optional[EndpointDict] = None) -> Dict:
        """Creates dictionary from Patient instance

        Returns:
            Dict
        """
        return {
            "patient_id": self.patient_id,
            "diagnoses": self.diagnoses,
            "diagnosis_dates": [x.strftime("%Y%m%d") for x in self.diagnosis_dates],
            "drugs": self.drugs,
            "prescription_dates": [
                x.strftime("%Y%m%d") for x in self.prescription_dates
            ],
            "birth_year": self.birth_year,
            "sex": self.sex,
            "patient_state": self.patient_state,
            "endpoint_labels": [
                ep
                for val, ep in zip(self.endpoint_labels, endpoint_dict.endpoints)
                if val == 1
            ]
            if endpoint_dict is not None and self.endpoint_labels is not None
            else self.endpoint_labels,
        }

    @classmethod
    def from_dict(
        cls,
        patient_dict: Dict,
        code_embed: CodeDict,
        sex_embed: SexDict,
        age_embed: AgeDict,
        state_embed: Optional[StateDict] = None,
        max_length: int = 512,
        **kwargs,
    ) -> "Patient":
        """Create Patient instance from dictionary

        Returns:
            Patient
        """
        if "state" not in patient_dict.keys():
            patient_dict["state"] = None

        return cls(
            **patient_dict,
            code_embed=code_embed,
            sex_embed=sex_embed,
            age_embed=age_embed,
            state_embed=state_embed,  # type: ignore
            max_length=max_length,
            **kwargs,
        )
