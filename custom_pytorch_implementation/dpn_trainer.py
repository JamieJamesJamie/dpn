"""
Main module to run from.
"""

import time

import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from pytorch_lightning.loggers import TensorBoardLogger

from custom_pytorch_implementation.cli_args import DPNConfig, Hparams

cs = ConfigStore.instance()
cs.store(name="config", node=DPNConfig)


@hydra.main(config_name="config")
def main(config: DPNConfig) -> None:
    """
    Main method.

    :param config: Command line arguments
    """

    pl.seed_everything(config.hparams.seed)

    logger = TensorBoardLogger(
        save_dir=str(config.hparams.log_dir), name=_experiment_dir_name(config.hparams)
    )

    trainer = pl.Trainer(**config.pl_trainer, logger=logger)

    # TODO: Define model (i.e. DPNLightning class) to be passed to trainer.fit()
    # trainer.fit(model)


def _experiment_dir_name(hparams: Hparams) -> str:
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
        + str(hparams.learn_lr)
        + "_clip"
        + str(hparams.meta_gradient_clip_value)
        + "_n_hidden_"
        + str(hparams.n_hidden)
        + "_latent_dim_"
        + str(hparams.obs_latent_dim)
        + "_dt_"
        + str(hparams.d_t)
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
    main()
