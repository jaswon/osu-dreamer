
from typing import Optional

import sys

import click

from pytorch_lightning.cli import LightningCLI

from osu_dreamer.lm.data.dataset import Data
from osu_dreamer.lm.model import Model

file_option_type = click.Path(exists=True, dir_okay=False)

default_config_path = './osu_dreamer/lm/model.yml'

class MyLightningCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        # Manually link the vocab from the data module to the model.
        # This is a workaround for a limitation in LightningCLI's link_arguments with nested class arguments.
        self.config.model.vocab = self.config.data.vocab

@click.command()
@click.option('-c', '--config', type=file_option_type, default=default_config_path, help='config file')
@click.option(   '--ckpt-path', type=file_option_type, help='if provided, checkpoint from which to resume training')
def fit_lm(config: str, ckpt_path: Optional[str]):
    """begin a training run for the language model."""

    sys.argv.pop(0) # pop subcommand
    args = ['--config', config]

    cli = MyLightningCLI(Model, Data, save_config_callback=None, args=args, run=False)
    cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt_path)
    cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=ckpt_path)
