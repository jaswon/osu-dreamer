from pytorch_lightning.cli import LightningCLI

from osu_dreamer.model.vqvae import Model
from osu_dreamer.data import Data

cli = LightningCLI(Model, Data)