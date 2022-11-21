from pytorch_lightning.cli import LightningCLI

from osu_dreamer.model import Model

cli = LightningCLI(Model)