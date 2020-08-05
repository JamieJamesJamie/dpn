"""
This module contains code to handle the continually growing dataset used to store
experience during reinforcement learning. Adapted from https://bit.ly/3hoo9xa.
"""
# pylint: disable=abstract-method,too-few-public-methods

from collections import namedtuple, deque

import numpy as np
from torch.utils.data.dataset import IterableDataset

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


class ReplayBuffer:
    """
    Replay buffer for storing experiences allowing the agent to learn from them.
    """

    def __init__(self, capacity: int):
        """
        Constructor for :class:`ReplayBuffer`.

        :param capacity: Maximum size of the replay buffer.
        """
        self._buffer = deque(maxlen=capacity)

    def __len__(self):
        """
        Returns the length of the :class:`ReplayBuffer`

        :return: Length of the :class:`ReplayBuffer`
        """
        return len(self._buffer)

    def append(self, experience: Experience):
        """
        Add experience to the replay buffer.

        :param experience: Experience to add to the replay buffer
        """
        self._buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Returns :param:`batch_size` samples from the replay buffer.

        :param batch_size: Number of samples to include in the sampled batch.
        :return: A batch of samples from the replay buffer
        """

        indices = np.random.choice(len(self._buffer), batch_size, replace=False)

        states, actions, rewards, dones, next_states = zip(
            *[self._buffer[index] for index in indices]
        )

        samples = (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )

        return samples


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the :class:`ReplayBuffer` which will be updated with
    new experiences during training.
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200):
        """
        Constructor for :class:`RLDataset`.

        :param buffer: Replay buffer
        :param sample_size: Number of experiences to sample at a time
        """
        self._buffer = buffer
        self._sample_size = sample_size

    def __iter__(self):
        states, actions, rewards, dones, new_states = self._buffer.sample(
            self._sample_size
        )

        for i, _ in enumerate(dones):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
