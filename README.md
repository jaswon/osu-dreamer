# osu!dreamer server (api)

(official osu!dreamer discord)
[![image](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/ZewBWhjxsR)

osu!dreamer server is a api based on a machine learning (AI) generative model for osu! beatmaps

## Installation & startup

Dependencys needed for installation & running

- FFmpeg
- Git
- Python 3.9

To clone this repo, run:

```
git clone https://github.com/seanmcbroom/osu-dreamer-server
```

To install dependencies, run:

```
pip install ./osu-dreamer-server
```

## Running server

**To start the server, run**

```
python scripts/server.py [PATH_TO_CHECKPOINT] --port [PORT_NUMBER]
```

**Command Usage**

```
usage: server.py [-h] [--port PORT] MODEL_PATH

generate osu!std maps from raw audio

positional arguments:
  MODEL_PATH   trained model (.ckpt)

optional arguments:
  -h, --help   show this help message and exit

server arguments:
  --port PORT  port to run the server on
```
