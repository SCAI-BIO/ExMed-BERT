# Data related exceptions


class PatientNotValid(Exception):
    """Raise when patient does not fulfill the predefined criteria"""


class NonePatient(Exception):
    """Raise when None got returned instead of patient"""


class MisconfiguredDataset(Exception):
    """Indicate when dataset configuration is wrong"""


class UndefinedPatients(Exception):
    """Occurs when neither patient paths not patient instances are provided"""


class UndefinedEndpoint(Exception):
    """Endpoint is undefined although it must not be"""


class MappingError(Exception):
    """Error occurring during ICD->Phewas or RXnorm->ATC mapping"""


class InputException(Exception):
    """Wrong input for Patient class"""
