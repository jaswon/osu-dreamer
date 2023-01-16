from pytorch_lightning.cli import LightningCLI

from data import Data
from model import Model

if __name__ == "__main__":
    cli = LightningCLI(Model, Data)