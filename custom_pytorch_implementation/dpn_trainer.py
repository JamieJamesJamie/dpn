"""
Main module to run from.
"""
import time

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from custom_pytorch_implementation.cli_args import parse_args


def main(hparams):
    """
    Main method.

    :param hparams: Command line arguments
    """

    pl.seed_everything(hparams.seed)

    logger = TensorBoardLogger(
        save_dir=hparams.logdir, name=_experiment_dir_name(hparams)
    )

    trainer = pl.Trainer.from_argparse_args(hparams, logger=logger)

    # TODO: Define model (i.e. DPNLightning class) to be passed to trainer.fit()
    # trainer.fit(model)


def _experiment_dir_name(hparams) -> str:
    """
    Returns logging directory name for an experiment based on :param:`hparams`.

    :param hparams: Hyperparameters to generate an experiment logging directory from
    :return: Experiment logging directory name
    """

    dirname = (
        hparams.task
        + "_latent_planning_ol_lr"
        + str(hparams.ol_lr)
        + "_il_lr"
        + str(hparams.il_lr)
        + "_num_plan_updates_"
        + str(hparams.num_plan_updates)
        + "_horizon_"
        + str(hparams.inner_horizon)
        + "_num_train_"
        + str(hparams.num_train)
        + "_"
        + hparams.decay
        + "_"
        + hparams.learn_lr
        + "_clip"
        + str(hparams.meta_gradient_clip_value)
        + "_n_hidden_"
        + str(hparams.n_hidden)
        + "_latent_dim_"
        + str(hparams.obs_latent_dim)
        + "_dt_"
        + str(hparams.dt)
    )

    if hparams.spatial_softmax:
        dirname += "_fp"

    if hparams.bias_transform:
        dirname += "_bt"

    dirname += (
        "_n_act_%d" % hparams.n_hidden_act
        + "_act_latent_dim_"
        + str(hparams.act_latent_dim)
    )

    dirname += "_beta_" + str(hparams.beta)

    dirname += "_date_"

    if not hparams.test:
        dirname += time.strftime("%d-%m-%Y_%H-%M-%S")
    else:
        dirname += hparams.date

    return dirname


if __name__ == "__main__":
    main(parse_args())
