# imports ----------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import logging

# global vars ------------------------------------------------------------------

logger = logging.get_logger(__name__)

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
else:
    _is_native_amp_available = False

# class definitions ------------------------------------------------------------


class MedBertTrainer(Trainer):
    """
    Custom Trainer with prediction step to handle Med-BERT pretraining
    """

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is not None:
            raise NotImplementedError

        with torch.no_grad():
            # if self.use_amp:  # type: ignore
            #     with autocast():
            #         outputs = model(**inputs)
            # else:
            outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None and "labels" in inputs:
                    logger.error(
                        "Label_smoother is not implemented in the  \
                        inherited method"
                    )
                    raise NotImplementedError
                else:
                    loss = (outputs["loss"]).mean().detach()
                    logits = tuple(
                        v.argmax(2) for k, v in outputs.items() if k in ["logits"]
                    )
                    if "plos_logits" in outputs.keys():
                        logits += (outputs["plos_logits"].argmax(1),)
            else:
                loss = None
                logits = tuple(
                    v.argmax(2) for k, v in outputs.items() if k in ["logits"]
                )
                if "plos_logits" in outputs.keys():
                    logits += (outputs["plos_logits"].argmax(1),)
            if self.args.past_index >= 0:
                logger.error("Past index has been removed in this implementation")
                raise NotImplementedError

        if prediction_loss_only:
            return loss, None, None

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = tuple(inputs.get(name) for name in self.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return loss, logits, labels  # type: ignore
