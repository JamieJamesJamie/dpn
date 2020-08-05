"""
Main module to run from.
"""

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from custom_pytorch_implementation.cli_args import parse_args


def main(hparams):
    """
    Main method.

    :param hparams: Command line arguments
    """

    pl.seed_everything(hparams.seed)

    logger = TensorBoardLogger(save_dir=hparams.logdir, name=hparams.task)

    trainer = pl.Trainer.from_argparse_args(hparams, logger=logger)

    # TODO: Define model (i.e. DPNLightning class) to be passed to trainer.fit()
    # trainer.fit(model)


if __name__ == "__main__":
    main(parse_args())
