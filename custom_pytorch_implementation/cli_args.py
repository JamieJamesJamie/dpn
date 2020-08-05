"""
This module contains all the command line arguments used in the program.
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl


def _add_pl_trainer_args():
    """
    Adds command line arguments specific to Pytorch Lightning to a parser. The parser
    is then returned.

    :return: Parser
    """

    # noinspection PyTypeChecker
    trainer_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )

    trainer_parser = pl.Trainer.add_argparse_args(trainer_parser)
    return trainer_parser


def _add_custom_args(parser):

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--logdir",
        type=lambda path: str(Path(path)),
        default=str(Path("lightning_logs")),
        help="Path to store logs",
    )

    return parser


def parse_args():
    """
    Adds command line arguments to a parser and returns the parsed command line
    arguments.

    :return: Parsed command line arguments
    """

    trainer_parser = _add_pl_trainer_args()

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        parents=[trainer_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--inner-horizon", type=int, default=5, help="length of RNN rollout horizon"
    )
    parser.add_argument(
        "--outer-horizon", type=int, default=5, help="length of BC loss horizon"
    )
    parser.add_argument(
        "--sampling-max-horizon",
        type=int,
        default=0,
        help="max length of BC loss horizon for sampling",
    )
    parser.add_argument(
        "--num-plan-updates",
        type=int,
        default=8,
        help="number of planning update steps before BC loss",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=1,
        help="number of hidden layers to encode after conv",
    )
    parser.add_argument(
        "--obs-latent-dim", type=int, default=128, help="obs latent space dim"
    )
    parser.add_argument(
        "--act-latent-dim", type=int, default=128, help="act latent space dim"
    )
    parser.add_argument(
        "--meta-gradient-clip-value",
        type=float,
        default=25.0,
        help="meta gradient clip value",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--test-batch-size", type=int, default=128, help="test batch size"
    )
    parser.add_argument("--il-lr-0", type=float, default=0.5, help="il_lr_0")
    parser.add_argument("--il-lr", type=float, default=0.25, help="il_lr")
    parser.add_argument("--ol-lr", type=float, default=0.0035, help="ol_lr")
    parser.add_argument(
        "--num-batch-updates",
        type=int,
        default=100000,
        help="number of minibatch updates",
    )
    parser.add_argument(
        "--testing-frequency",
        type=int,
        default=500,
        help="how frequently to get stats for test data",
    )
    parser.add_argument(
        "--log-frequency", type=int, default=1000, help="how frequently to log the data"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="log",
        help="name of log file to dump test data stats",
    )
    # parser.add_argument(
    #     "--log-directory",
    #     type=str,
    #     default="/scr/kevin/unsupervised_upn/",
    #     help="name of log directory to dump checkpoints",
    # )
    parser.add_argument(
        "--huber",
        dest="huber_loss",
        action="store_true",
        help="whether to use Huber Loss",
    )
    parser.add_argument(
        "--no-huber",
        dest="huber_loss",
        action="store_false",
        help="whether not to use Huber Loss",
    )
    parser.set_defaults(huber_loss=True)
    parser.add_argument(
        "--huber-delta", type=float, default=1.0, help="delta coefficient in Huber Loss"
    )
    parser.add_argument(
        "--img-c", type=int, default=3, help="number of channels in input"
    )
    parser.add_argument(
        "--task", type=str, default="pointmass", help="which task to train on"
    )
    parser.add_argument(
        "--act-scale-coeff", type=float, default=1.0, help="scaling factor for actions"
    )
    parser.add_argument(
        "--act-dim", type=int, default=2, help="dimensionality of action space"
    )
    parser.add_argument(
        "--joint-dim", type=int, default=4, help="dimensionality of joint space"
    )
    parser.add_argument("--img-h", type=int, default=84, help="image height")
    parser.add_argument("--img-w", type=int, default=84, help="image width")
    parser.add_argument(
        "--num-train", type=int, default=5000, help="number of rollouts for training"
    )
    parser.add_argument(
        "--num-test", type=int, default=1000, help="number of rollouts for test"
    )
    parser.add_argument(
        "--spatial-softmax",
        dest="spatial_softmax",
        action="store_true",
        help="whether to use spatial softmax",
    )
    parser.add_argument(
        "--bias-transform",
        dest="bias_transform",
        action="store_true",
        help="whether to use bias transform",
    )
    parser.add_argument(
        "--bt-num-units",
        type=int,
        default=10,
        help="number of dimensions in bias transform",
    )
    parser.add_argument(
        "--nonlinearity",
        type=str,
        default="swish",
        help="which nonlinearity for dynamics and fully connected",
    )
    parser.add_argument(
        "--decay",
        type=str,
        default="None",
        help="decay the learning rate in the outer loop",
    )
    parser.add_argument(
        "--niter-init",
        type=int,
        default=50000,
        help="number of iterations of running the initial learning rate",
    )
    parser.add_argument("--learn-lr", type=str, default="False", help="learn the il-lr")
    parser.add_argument(
        "--test", type=str, default="False", help="restore the model for testing or not"
    )
    parser.add_argument(
        "--dt",
        type=int,
        default=1,
        help="number of time steps between the initial image and the final goal image",
    )
    parser.add_argument(
        "--restore-iter",
        type=int,
        default=-1,
        help="restore the checkpoint at which iteration",
    )
    parser.add_argument(
        "--date", type=str, default="False", help="date of the checkpoint created"
    )
    parser.add_argument(
        "--training-tfrecord",
        type=str,
        default="None",
        help="path to the tfrecord file for training",
    )
    parser.add_argument(
        "--validation-tfrecord",
        type=str,
        default="None",
        help="path to the tfrecord file for validation",
    )
    parser.add_argument(
        "--n-hidden-act", type=int, default=1, help="number of hidden layers for action"
    )
    parser.add_argument("--beta", type=float, default=1.0, help="beta for beta-vae")

    parser = _add_custom_args(parser)

    args = parser.parse_args()
    return args
