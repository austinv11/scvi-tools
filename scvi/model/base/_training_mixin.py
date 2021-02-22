from typing import Optional, Union

import numpy as np
import torch

from scvi.dataloaders import DataSplitter
from scvi.lightning import TrainingPlan

from ._trainrunner import TrainRunner


class UnsupervisedTrainingMixin:
    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        plan_kwargs
            Keyword args for model-specific Pytorch Lightning task. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.lightning.Trainer`.
        """
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        if use_gpu is None or use_gpu is True:
            use_gpu = torch.cuda.current_device() if torch.cuda.is_available() else 0
        elif use_gpu is False:
            use_gpu = 0
        elif isinstance(use_gpu, int):
            use_gpu = [use_gpu]

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )
        training_plan = TrainingPlan(
            self.module, len(data_splitter.train_idx), **plan_kwargs
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            gpus=use_gpu,
            **trainer_kwargs,
        )
        return runner()
