# Script for modified Trainer

# imports ----------------------------------------------------------------------

from typing import Any, Dict, Optional, Union

import optuna
import torch
from transformers import Trainer, get_scheduler
from transformers.trainer_utils import HPSearchBackend
from transformers.utils import logging

# global vars ------------------------------------------------------------------

logger = logging.get_logger(__name__)

# functions --------------------------------------------------------------------


# class definitions ------------------------------------------------------------


class FineTuningTrainer(Trainer):
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Set up the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to take.
            optimizer (torch.optim.Optimizer, optional):

        """
        if self.lr_scheduler is None:
            print(num_training_steps)
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,  # type: ignore
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]):
        """HP search setup code"""
        self._trial = trial

        if self.hp_search_backend is None or trial is None:
            return
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            params: optuna.Trial = self.hp_space(trial)  # type: ignore
        elif self.hp_search_backend == HPSearchBackend.RAY:
            params: Dict[str, Any] = trial  # type: ignore
            params.pop("wandb", None)  # type: ignore
        elif self.hp_search_backend == HPSearchBackend.SIGOPT:
            params = {
                k: int(v) if isinstance(v, str) else v
                for k, v in trial.assignments.items()  # type: ignore
            }
        else:
            raise Exception()

        # Modification for RNN models

        if "rnn_num_layers" in params.keys():
            num_layers = params.pop("rnn_num_layers")
            self.model_init.config.rnn_num_layers = num_layers
        elif "num_final_blocks" in params.keys():
            num_final_blocks = params.pop("num_final_blocks")
            self.model_init.config.num_final_blocks = num_final_blocks

        # Modification end

        for key, value in params.items():
            if not hasattr(self.args, key):
                logger.warning(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in `TrainingArguments`."
                )
                continue
            old_attr = getattr(self.args, key, None)
            # Casting value to the proper type
            if old_attr is not None:
                value = type(old_attr)(value)
            setattr(self.args, key, value)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            logger.info("Trial:", trial.params)  # type: ignore
        elif self.hp_search_backend == HPSearchBackend.SIGOPT:
            logger.info(f"SigOpt Assignments: {trial.assignments}")  # type: ignore
        if self.args.deepspeed:
            # Rebuild the deepspeed config to reflect the updated training parameters
            from transformers.deepspeed import HfDeepSpeedConfig

            self.args.hf_deepspeed_config = HfDeepSpeedConfig(self.args)
