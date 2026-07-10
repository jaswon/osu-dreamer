
from typing import Union, Optional

import warnings
import click

from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning import Trainer

from osu_dreamer.data.module import BeatmapDataModule
from osu_dreamer.models.style.train import StyleTrainer

file_option_type = click.Path(exists=True, dir_okay=False)

default_config_path = './osu_dreamer/models/style/model.yml'

@click.command()
@click.option('-c', '--config', type=file_option_type, default=default_config_path, help='config file')
@click.option(   '--ckpt-path', type=file_option_type, help='if provided, checkpoint from which to resume training')
def fit_style(config: str, ckpt_path: Optional[str]):
    """begin a training run for the style model."""

    parser = LightningArgumentParser()
    parser.add_argument('seed_everything', type=Union[int, bool])
    parser.add_lightning_class_args(StyleTrainer, 'model')
    parser.add_lightning_class_args(BeatmapDataModule, 'data')
    parser.add_lightning_class_args(Trainer, 'trainer')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cli = LightningCLI(StyleTrainer, BeatmapDataModule, args=parser.parse_path(config), run=False)
    cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=ckpt_path)
