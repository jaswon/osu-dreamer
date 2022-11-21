from setuptools import setup

setup(
    name="osu-dreamer",
    version="1.0",
    packages=["osu_dreamer"],
    install_requires=[
        "bezier",
        "librosa",
        "tqdm",
        "torch",
        "torchaudio",
        "einops",
        "pytorch-lightning",
    ],
    extras_require={
        "dev": ["matplotlib"],
    },
)