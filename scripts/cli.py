from pytorch_lightning.cli import LightningCLI

from osu_dreamer.model import Model
from osu_dreamer.data import Data

if __name__ == "__main__":
    cli = LightningCLI(Model, Data)