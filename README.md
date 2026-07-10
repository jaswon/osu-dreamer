# osu!dreamer - an ML model for generating maps from raw audio

[![image](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/ZewBWhjxsR)

osu!dreamer is a generative model for osu! beatmaps based on diffusion

-   [sample generated mapset](https://osu.ppy.sh/beatmapsets/1888586#osu/3889513)
-   [video of a generated map](https://streamable.com/ijp1jj)

## Quick start

[colab notebook (no installation required)](https://colab.research.google.com/drive/1Th6v5OOrY5vcTWvIH3NKZsuj_RMnAEM5#sandboxMode=true)

## Installation for development

### Required dependencies
- FFmpeg
- python 3.9
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 

Clone this repo, then run:

```
uv sync [--group dev]
```

This will install `osu-dreamer`'s dependencies

## Model training

### Generate dataset

first you must generate a dataset, using eg. your `osu!/Songs` directory.
This step only needs to be done once (unless you delete the generated dataset directory).

```
$ uv run python -m osu_dreamer.model generate-data [MAPS_DIR]
```

where `[MAPS_DIR]` is the path to eg. your `osu!/Songs` directory

### Training

after the dataset generation completes, you can start training. Training occurs in two stages:

#### Latent Model

```
$ uv run python -m osu_dreamer.model fit-latent
```

See `osu_dreamer/latent_model/model.yml` for all latent model training parameters.

At the end of every epoch, the model parameters will be checkpointed to `runs/latent/version_{NUM}/checkpoints/epoch={EPOCH}-step={STEP}.ckpt`. You can resume training from a saved checkpoint by adding `--ckpt-path [PATH TO CHECKPOINT]` to the `fit-latent` command.

run `tensorboard --logdir=runs/latent` in a new window to track training progress in Tensorboard

After training, copy/link the final checkpoint to the repo root:

```
ln runs/latent/version_{NUM}/checkpoints/epoch={EPOCH}-step={STEP}.ckpt latent.ckpt
```

Afterwards, proceed to the next training stage

#### Flow Model

```
$ uv run python -m osu_dreamer.model fit-denoiser
```

See `osu_dreamer/diffusion_model/model.yml` for all latent model training parameters.

At the end of every epoch, the model parameters will be checkpointed to `runs/denoiser/version_{NUM}/checkpoints/epoch={EPOCH}-step={STEP}.ckpt`. You can resume training from a saved checkpoint by adding `--ckpt-path [PATH TO CHECKPOINT]` to the `fit-denoiser` command.

run `tensorboard --logdir=runs/denoiser` in a new window to track training progress in Tensorboard

After training, copy/link the final checkpoint to the repo root:

```
ln runs/denoiser/version_{NUM}/checkpoints/epoch={EPOCH}-step={STEP}.ckpt denoiser.ckpt
```

Afterwards, proceed to the next training stage

#### Style Model

```
$ uv run python -m osu_dreamer.model fit-style
```

See `osu_dreamer/models/style/model.yml` for all style model training parameters.

At the end of every epoch, the model parameters will be checkpointed to `runs/style/version_{NUM}/checkpoints/epoch={EPOCH}-step={STEP}.ckpt`. You can resume training from a saved checkpoint by adding `--ckpt-path [PATH TO CHECKPOINT]` to the `fit-style` command.

run `tensorboard --logdir=runs/style` in a new window to track training progress in Tensorboard

After training, copy/link the final checkpoint to the repo root:

```
ln runs/style/version_{NUM}/checkpoints/epoch={EPOCH}-step={STEP}.ckpt style.ckpt
```

After obtaining all three training checkpoints, you must create an inference artifcat

## Model Inference

### Inference artifact

```
# uv run python -m osu_dreamer export-inference
```

this will create an `inference.pt` file in the repo root.

### Generate mapset

```
$ uv run python -m osu_dreamer predict --help
Usage: python -m osu_dreamer predict [OPTIONS]

  generate osu!std maps from raw audio.

Options:
  --model-path FILE               inference artifact (.pt)  [required]
  --audio-file FILE               audio file to map  [required]
  --diff <FLOAT FLOAT FLOAT FLOAT FLOAT>...
                                  difficulty conditioning (sr, ar, od, cs, hp)
  --sample-steps INTEGER          number of diffusion steps to sample
  --title TEXT                    Song title - required if it cannot be
                                  determined from the audio metadata
  --artist TEXT                   Song artist - required if it cannot be
                                  determined from the audio metadata
  --help                          Show this message and exit.
```

you may specify `--diff` multiple times to generate multiple diffs at once. 


### visual validation

![image](https://user-images.githubusercontent.com/943003/203165744-68da33fa-967f-45a7-956e-f0fe0114f9cc.png)

The training process will generate one plot at the end of every epoch, using a sample from the validation set

-   the first row is the spectrogram of the audio file
-   the second row is the actual map associated with the audio file in its signal representation
-   the third and fourth rows are signal representations of the maps produced by the model

## 💻 Windows Batch Setup

> ⚠️ Support for training/evaluating the model locally on Windows is highly experimental and provided as-is

### Requirements
-   🐍 Python 3.9 (via Microsoft Store, or python.org)

### Installation

Install the source code directly through github, or with the git clone command:

`git clone https://github.com/jaswon/osu-dreamer`

### Usage

Setup from this point is pretty simple, navigate into the osu-dreamer directory and then into the `windows_scripts` folder, this is where all the batch scripts are stored.

First, you will need to run `! Install.bat`, this will install osu-dreamer and all of its dependencies. Optionally you can install tensorboard and mathplotlib to view training statistics.

Now you're ready to begin training your own model! Here is a list of all the scripts and their functionality

-   Install
    -   Installs osu-dreamer and all of its dependencies.
-   Run Training
    -   Compiles the given songs directory and begins training a model
-   Resume Training
    -   Resumes training the given checkpoint
-   Generate Beatmap
    -   Generates a beatmap with the given information (requires a trained model and song)
-   Tensorboard
    -   Hosts tensorboard for tracking training statistics
