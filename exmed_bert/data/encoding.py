# Code for Embedding dictionaries

# imports ----------------------------------------------------------------------

import logging
import pickle
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from exmed_bert.data.data_exceptions import MappingError

# global vars ------------------------------------------------------------------

DICT_DEFAULTS = ["PAD", "UNK", "CLS", "SEP", "MASK", "NA"]
logger = logging.getLogger(__name__)
UNKNOWN_ICD: Set[str] = set()
UNKNOWN_RX: Set[str] = set()

# functions --------------------------------------------------------------------


def icd_to_phewas(
    icd: Union[str, List[str], List[int], int], icd_to_pw: Dict[str, str]
) -> List[str]:
    """Convert ICD codes to phewas

    Args:
        icd (List[int]): List of ICD codes
        icd_to_pw (Dict[str, str]): Dictionary with ICD and Phewas codes

    Returns:
        str: Phecode
    """

    def _map_to_phewas(icd_code: Union[str, int], mapping: Dict[str, str]) -> str:
        """Helper function to map icd to phewas"""
        if icd_code == "UNK":
            raise MappingError()
        if isinstance(icd_code, str) and icd_code in DICT_DEFAULTS:
            return icd_code

        code = mapping.get(icd_code, "UNK")  # type: ignore
        if code == "UNK":
            # noinspection PyBroadException
            try:
                global UNKNOWN_ICD
                UNKNOWN_ICD.add(icd_code)  # type: ignore
            except:
                logger.info("Could not add code to unknowns")

        return code

    if isinstance(icd, list):
        phewas_codes = [_map_to_phewas(code, icd_to_pw) for code in icd]
    elif isinstance(icd, str) or isinstance(icd, int):
        phewas_codes = [_map_to_phewas(icd, icd_to_pw)]
    else:
        raise MappingError("Error in icd to phewas conversion")
    return phewas_codes


def rxcui_to_atc4(
    rxcui: Union[str, List[str], List[int], int], rx_to_atc_map: Dict[str, str]
) -> List[str]:
    """RXCUI to ATC conversion

    Args:
        rxcui (List[int]): List of rxcuis
        rx_to_atc_map (Dict[str, str]): Dictionary with rxcui and atc codes

    Returns:
        List[str]: mapped atc codes
    """

    def _map_to_atc(rxcode: Union[str, int], mapping: Dict[str, str]) -> str:
        """Helper function to map rxnorm codes to atc"""
        if isinstance(rxcode, str) and rxcode in DICT_DEFAULTS:
            return rxcode
        elif isinstance(rxcode, str):
            rxcode = int(rxcode)

        code = mapping.get(rxcode, "UNK")  # type: ignore
        if code == "UNK":
            # noinspection PyBroadException
            try:
                global UNKNOWN_RX
                UNKNOWN_RX.add(rxcode)  # type: ignore
            except:
                logger.info("Could not add code to unknowns")

        return code

    if isinstance(rxcui, Iterable):
        atc_codes = [code for rx in rxcui for code in _map_to_atc(rx, rx_to_atc_map)]
    elif isinstance(rxcui, str) or isinstance(rxcui, int):
        atc_codes = [code for code in _map_to_atc(rxcui, rx_to_atc_map)]
    else:
        raise MappingError("Error in rx to atc conversion")
    return atc_codes


# class definitions ------------------------------------------------------------
class EndpointDict(object):
    """Class to store endpoint labels"""

    def __init__(self, endpoint_labels: List[str]):
        """Initialize EndpointDict

        Args:
            endpoint_labels (List[str]): List of possible endpoints
        """
        super().__init__()
        self.endpoints = endpoint_labels
        self.num_endpoints = len(endpoint_labels)
        self._id_to_endpoint = {i: ep for i, ep in enumerate(endpoint_labels)}
        self._endpoint_to_id = {ep: i for i, ep in self._id_to_endpoint.items()}

    def encode(self, patient_endpoints: List[Union[str, int]]) -> List[int]:
        """Encode patient labels as integers

        Args:
            patient_endpoints (List[Union[str, int]]): Endpoints per patient

        Returns:
            List[int]: Vector representation of endpoint status
        """
        enc = [0] * self.num_endpoints
        for i, endpoint in enumerate(self.endpoints):
            if endpoint in patient_endpoints:
                enc[i] = 1
        return enc

    def create_one_hot_tensor(
        self, patient_ids: List[int], endpoint_dict: Dict[str, List[int]]
    ) -> Dict[int, torch.Tensor]:
        """Creates one hot encodings for every patient

        Args:
            patient_ids (List[int]): List of patient ids
            endpoint_dict (Dict[str, List[int]]): Dictionary with list of patient ids per endpoint

        Returns:
            Dict[int, torch.tensor]: Dictionary with one hot encodings per patient
        """
        assert list(endpoint_dict.keys()) == self.endpoints, "Not the same labels"

        patient_one_hot: Dict[int, torch.Tensor] = {}
        for pid in patient_ids:
            labels = torch.zeros(self.num_endpoints)
            for ep, i in self._endpoint_to_id.items():
                if pid in endpoint_dict[ep]:
                    labels[i] = 1
            patient_one_hot[pid] = labels.int()

        return patient_one_hot

    def decode(self, patient_encoding: List[int]) -> List[str]:
        """Decode patient labels from integers

        Args:
            patient_encoding (List[int]): Vector representation of endpoint status

        Returns:
            List[str]: Endpoints per patient
        """
        endpoints: List[str] = []
        for i, val in enumerate(patient_encoding):
            if val == 1:
                endpoints.append(self.endpoints[i])

        return endpoints

    def __eq__(self, other: "EndpointDict") -> bool:
        if (
            self._id_to_endpoint == other._id_to_endpoint
            and self._endpoint_to_id == other._endpoint_to_id
        ):
            return True
        else:
            return False


class EmbedDict(object):
    """Base class for embedding dictionary"""

    def __init__(self, defaults: List[str] = DICT_DEFAULTS) -> None:
        """Initialize EmbedDIct

        Args:
            defaults (List[str], optional): Default values which should be present each time (e.g., CLS, PAD, MASK). Defaults to DICT_DEFAULTS.
        """
        super().__init__()

        self.vocab: List[str] = []
        self.entities: List[str] = []

        self.entity_to_id: Dict[str, int] = {}
        self.ids_to_entity: Dict[int, str] = {}
        self.labels_to_id: Dict[str, int] = {}
        self.ids_to_label: Dict[int, str] = {}
        self._add_labels_to_dict(defaults)

    def _add_labels_to_dict(self, labels: List[str], entity: str = "default"):
        """
        Adds a list of labels to the vocab. It can process one entity type at a time.

        Args:
            labels (List[str]): List of labels that are embedded
            entity (str): Entity type
        """
        if labels is None:
            return None

        unique_labels = []
        for label in labels:
            if label not in self.vocab and label not in unique_labels:
                unique_labels.append(label)

        start = len(self.ids_to_label)
        for _num, label in enumerate(unique_labels):
            num = start + _num

            self.labels_to_id[label] = num
            self.ids_to_label[num] = label
            self.entities.append(entity)

        if entity not in self.entity_to_id.keys():
            num = len(self.entity_to_id)
            self.entity_to_id[entity] = num
            self.ids_to_entity[num] = entity

        self.vocab.extend(unique_labels)

    def __call__(
        self,
        tokens: Union[Union[str, int], List[Union[str, int]]],
        return_entities: bool = False,
    ) -> Union[List[int], Tuple[List[int], List[str]]]:
        """
        Encode the input according to the vocab

        Args:
            tokens (Union[str, List[str]]): key(s) from the vocab
            return_entities (bool): return entity type

        Returns:
            Union[List[int], Tuple[List[int], List[str]]]: Integer representation for the respective key
        """
        return self.encode(tokens, return_entities)

    def write_to_txt(self, file_name: str):
        """
        Save vocab to txt file.

        Args:
            file_name (str): output file name
        """
        with open(file_name, "w+") as f:
            for code in self.ids_to_label.values():
                f.write(code + "\n")

    @classmethod
    def read_from_txt(cls, file_name: str):
        """
        Read the saved vocab from a text file.

        Args:
            file_name (str): Path to a vocab file.

        Returns:
            EmbedDict:
        """
        with open(file_name, "r") as f:
            labels = [code.replace("\n", "") for code in f.readlines()]

        return cls(labels)

    def encode(
        self, keys: Union[str, List[str]], return_entities: bool = False
    ) -> Union[List[int], Tuple[List[int], List[str]]]:
        """
        Args:
            keys (List[str]): List of keys from the vocab
            return_entities (bool): indicate if entity types should be returned

        Returns:
            List: List of integer representations
        """

        def convert_tokens_to_ids(
            token_list: Union[str, List[str]]
        ) -> Union[int, List[int]]:
            if len(token_list) == 1:
                return self.labels_to_id.get(token_list[0], -1)
            else:
                return list(map(lambda x: self.labels_to_id.get(x, -1), token_list))

        if isinstance(keys, set):
            keys = list(keys)
        elif not isinstance(keys, list):
            keys = [keys]

        tokens = []
        for key in keys:
            if key in self.vocab:
                tokens.append(key)
            elif str(key) in self.vocab:
                tokens.append(str(key))
            else:
                tokens.append("UNK")

        ids: List[int] = convert_tokens_to_ids(tokens)  # type: ignore
        if return_entities:
            entities = (
                [self.entities[ids]]
                if isinstance(ids, int)
                else [self.entities[i] for i in ids]
            )
            entities = list(map(self.entity_to_id.get, entities))
            return ids, entities
        else:
            return ids

    def decode(
        self, int_reps: Tensor, return_entities: bool = False
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """
        Args:
            int_reps (Tensor): Tensor with integer representations
            return_entities (bool): Boolean if entity types should be returned

        Returns:
            List[str]: List of keys from the vocab
            List[str]: List of entity names if return_entities is true
        """
        if isinstance(int_reps, Tensor):
            int_reps = int_reps.cpu().detach().numpy()

        if return_entities:
            return list(map(self.ids_to_label.get, int_reps)), [  # type: ignore
                self.entities[i] for i in int_reps
            ]
        else:
            return list(map(self.ids_to_label.get, int_reps))  # type: ignore

    def __len__(self) -> int:
        """Vocab size

        Returns:
            int: Number of codes/words in vocab
        """
        return len(self.vocab)

    def save(self, file_name: str):
        """Saves vocab to pickle file

        Args:
            file_name (str): Output path
        """
        pickle.dump(self, open(file_name, "wb"))

    @classmethod
    def load(cls, file_name: str) -> "EmbedDict":
        """Loads dictionary from pickle file

        Args:
            file_name (str): input path

        Returns:
            EmbedDict: Loaded dictionary instance
        """
        with open(file_name, "rb") as f:
            obj = pickle.load(f)

        return obj

    def __eq__(self, other: "EmbedDict") -> bool:
        """Check if dictionaries are identical"""
        if (
            self.ids_to_label == other.ids_to_label
            and self.labels_to_id == other.labels_to_id
        ):
            return True
        else:
            return False


class CodeDict(EmbedDict):
    """Dictionary for drug and diagnosis codes"""

    def __init__(
        self,
        atc_codes: List[str],
        phewas_codes: List[str],
        rx_to_atc_map: Optional[Dict[str, str]] = None,
        icd_to_phewas_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initializes a code dict for ATC and Phewas codes

        Args:
            atc_codes (List[str]): List of atc codes
            phewas_codes (List[str]): List of phecodes
            rx_to_atc_map (Dict[str, str], optional): Mapping from rxcuis to atc. Defaults to None.
            icd_to_phewas_map (Dict[str, str], optional): Mapping from icd to phecodes. Defaults to None.
        """
        super().__init__(defaults=DICT_DEFAULTS)
        assert (
            len(set(atc_codes).intersection(set(phewas_codes))) == 0
        ), "Codes are not unique"

        if atc_codes is not None:
            self._add_labels_to_dict(atc_codes, entity="atc")

        if phewas_codes is not None:
            self._add_labels_to_dict(phewas_codes, entity="phewas")

        self.atc_codes = frozenset(atc_codes)
        self.phewas_codes = frozenset(phewas_codes)
        self.rx_atc_map = rx_to_atc_map
        self.icd_phewas_map = icd_to_phewas_map

    def __eq__(self, other: "CodeDict") -> bool:
        """Check if two dictionaries match"""
        if (
            super().__eq__(other)
            and self.rx_atc_map == other.rx_atc_map
            and self.icd_phewas_map == other.icd_phewas_map
        ):
            return True
        else:
            return False

    def split_codes_and_dates(
        self, codes: List[str], dates: List[datetime]
    ) -> Dict[str, List[Any]]:
        """Generate dictionary with split codes and dates

        Args:
            codes (List[str]): ATC and Phecodes
            dates (List[datetime]): Timestamps for each of the codes

        Raises:
            Exception

        Returns:
            Dict[str, List[Any]]: Dictionary with ATC and Phewas codes
        """
        assert len(codes) == len(dates)

        output = {
            "atc_codes": [],
            "atc_dates": [],
            "phewas_codes": [],
            "phewas_dates": [],
        }
        for code, date in zip(codes, dates):
            if code in self.atc_codes:
                output["atc_codes"].append(code)
                output["atc_dates"].append(date)
            elif code in self.phewas_codes:
                output["phewas_codes"].append(code)
                output["phewas_dates"].append(date)
            elif code in DICT_DEFAULTS:
                pass
            else:
                logger.error(f"Could not process{code} correctly")
                raise Exception("Codes and dates could not be split.")

        return output

    @classmethod
    def copy(cls, other_dict: "CodeDict") -> "CodeDict":
        """Copies content from another CodeDict"""
        cd = CodeDict([], [])
        cd.atc_codes = other_dict.atc_codes
        cd.icd_phewas_map = other_dict.icd_phewas_map
        cd.phewas_codes = other_dict.phewas_codes
        cd.rx_atc_map = other_dict.rx_atc_map
        cd.ids_to_entity = other_dict.ids_to_entity
        cd.entity_to_id = other_dict.entity_to_id
        cd.ids_to_label = other_dict.ids_to_label
        cd.labels_to_id = other_dict.labels_to_id
        cd.vocab = other_dict.vocab
        cd.entities = other_dict.entities
        return cd


class AgeDict(EmbedDict):
    """
    Dictionary for the age embedding
    """

    def __init__(
        self, max_age: int = 110, min_age: int = 0, binsize: float = 0.1
    ) -> None:
        """
        Initialize a dictionary for age embeddings

        Args:
            max_age (int, optional): Maximum encoded age. Defaults to 110.
            min_age (int, optional): Minimum encoded age. Defaults to 0.
            binsize (float, optional): Numeric resolution of age encodings (E.g.
                                       79.1 years old). Defaults to 0.1.
        """
        super().__init__(defaults=["PAD", "UNK"])

        ages = list(np.round(np.arange(min_age, max_age + 1, binsize), 1))
        ages = [round(age, 1) for age in ages]
        self._add_labels_to_dict(ages, entity="age")


class SexDict(EmbedDict):
    """
    Dictionary for the sex embedding

    It is used for the UNK token in case we do not have the sex information.
    """

    def __init__(self, sex: List[str] = ["MALE", "FEMALE"]) -> None:
        """Initialize SexDict

        Args:
            sex (List[str], optional): Usually, the is no need for modification. Defaults to ["MALE", "FEMALE"].
        """
        super().__init__(defaults=["PAD", "UNK"])

        self._add_labels_to_dict(sex, entity="gender")


class StateDict(EmbedDict):
    """
    Dictionary for the state embedding

    It is used for the UNK token in case we do not have the state information.
    """

    def __init__(self, states: List[str]) -> None:
        """Initializes a StateDict

        Args:
            states (List[str]): List of states/regions/cities
        """
        super().__init__(defaults=["PAD", "UNK"])

        self._add_labels_to_dict(states, entity="state")
