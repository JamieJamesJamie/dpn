"""
This module contains all the command line arguments used in the program.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Hparams:
    """
    Hparams for DPN. Most of the fields in this dataclass are copied from the
    original implementation.
    """

    # TODO: Fix Pylint's "too-many-instance-attributes" problem in Hparams dataclass

    inner_horizon: int = 5  # Length of RNN rollout horizon
    outer_horizon: int = 5  # Length of BC loss horizon
    sampling_max_horizon: int = 0  # Max length of BC loss horizon for sampling
    num_plan_updates: int = 8  # Number of planning update steps before BC loss
    n_hidden: int = 1  # Number of hidden layers to encode after conv
    obs_latent_dim: int = 128  # Obs latent space dim
    act_latent_dim: int = 128  # Act latent space dim

    meta_gradient_clip_value: float = 25.0  # Meta gradient clip value

    batch_size: int = 128  # Batch size
    test_batch_size: int = 128  # Test batch size

    il_lr_0: float = 0.5  # il_lr_0
    il_lr: float = 0.25  # il_lr
    ol_lr: float = 0.0035  # ol_lr

    num_batch_updates: int = 100000  # Number of minibatch updates
    testing_frequency: int = 500  # How frequently to get stats for test data

    log_frequency: int = 1000  # How frequently to log the data
    log_file: str = "log"  # Name of log file to dump test data stats
    log_dir: Path = "dpn_lightning_logs"  # Path to store logs

    huber_loss: bool = True  # Whether to use Huber loss
    huber_delta: float = 1.0  # Delta coefficient in Huber loss

    img_height: int = 84  # Image height
    img_width: int = 84  # Image width
    img_channels: int = 3  # Number of channels in input image

    task: str = "kinova_reach"  # Which task to train on

    act_scale_coeff: float = 1.0  # Scaling factor for actions

    act_dim: int = 2  # Dimensionality of action space
    joint_dim: int = 4  # Dimensionality of joint space

    num_train: int = 5000  # Number of rollouts for training
    num_test: int = 1000  # Number of rollouts for testing

    spatial_softmax: bool = False  # Whether to use spatial softmax

    bias_transform: bool = False  # Whether to use bias transform
    bt_num_units: int = 10  # Number of dimensions in bias transform

    nonlinearity: str = "swish"  # Which nonlinearity for dynamics and fully connected

    decay: str = ""  # Decay the learning rate in the outer loop

    n_iter_init: int = 50000  # No. of iterations of running the initial learning rate

    learn_lr: bool = False  # Learn the il_lr

    test: bool = False  # Restore the model for testing or not

    d_t: int = 1  # No. of time steps between the initial image, and the goal image

    restore_iter: int = -1  # Restore the checkpoint at which iteration

    date: str = "False"  # Date of the checkpoint created
    training_tf_record: Path = "None"  # Path to the tfrecord file for training
    validation_tf_record: Path = "None"  # Path to the tfrecord file for validation

    n_hidden_act: int = 1  # Number of hidden layers for action

    beta: float = 1.0  # Beta for beta-vae

    seed: int = 0  # Random seed for reproducibility


@dataclass
class DPNConfig:
    """
    Config for DPN that contains the available command line args.

    The pl_trainer field is a dictionary that can have command line args for the
    Pytorch Lightning trainer added to it using e.g. `+pl_trainer.gpus=-1`.

    The hparams field contains most of the command line args used in the original DPN
    implementation, with some minor additions. Help text for each of these fields is
    available as a comment in the :class:`Hparams` dataclass definition.
    """

    # Pytorch Lightning args - refer to PL docs for specific args
    pl_trainer: dict = field(default_factory=dict)

    # Hparams for DPN
    hparams: Hparams = Hparams()
