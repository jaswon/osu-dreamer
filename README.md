# osu!dreamer - an ML model for generating maps from raw audio

osu!dreamer is a generative model for osu! beatmaps based on diffusion

- [sample generated mapset](https://osu.ppy.sh/beatmapsets/1888586#osu/3889513)
- [video of a generated map](https://streamable.com/ijp1jj)

## Installation

Clone this repo, then run:

```
pip install ./osu-dreamer
```

This will install `osu-dreamer` as well as all dependencies

## Generate your own maps

[colab notebook](https://colab.research.google.com/drive/1Th6v5OOrY5vcTWvIH3NKZsuj_RMnAEM5?usp=sharing)

### locally

```
python scripts/pred.py -S SAMPLE_STEPS -N NUM_SAMPLES AUDIO_FILE MODEL_PATH
```

- `SAMPLE_STEPS`: number of diffusion steps to sample
- `NUM_SAMPLES`: number of maps to generate
- `AUDIO_FILE`: path to audio file
- `MODEL_PATH`: path to trained model

## Model training

```
python scripts/cli.py fit -c config.yml --model.src_path SONGS_DIR
```

replace `SONGS_DIR` with the path to the osu! Songs directory (or a directory with the same structure).
other model parameters are in `config.yml`
