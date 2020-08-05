"""
This file defines the DPN model.
"""
# pylint: disable=too-many-ancestors

import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from custom_pytorch_implementation.dataset.dataset import ReplayBuffer, RLDataset


class DPNLightning(pl.LightningModule):
    """
    Lightning module that represents the DPN model.
    """

    def __init__(self, hparams: argparse.Namespace):
        """
        Constructor for :class:`DPNLightning`.

        :param hparams: Hyperparameters defined as command line arguments
        """

        # Needed to record hyperparameters to tensorboard
        super().__init__()
        self.hparams = hparams

        self._replay_buffer = ReplayBuffer(self.hparams.batch_size)

    def forward(self, *args, **kwargs):
        # TODO: Implement PL forward() method
        pass

    def train_dataloader(self) -> DataLoader:
        """
        Returns :class:`DataLoader`.

        :return: dataloader.
        """

        return self.__dataloader()

    def __dataloader(self):
        """
        Initialise and return the :class:`ReplayBuffer` dataset used for retrieving
        experiences.

        :return: An initialised :class:`ReplayBuffer` dataset
        """

        # TODO: Check what should be passed to RLDataset instead of max_episode_length
        dataset = RLDataset(self._replay_buffer, self.hparams.max_episode_length)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)

        return dataloader
