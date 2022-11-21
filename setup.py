from setuptools import setup

setup(
    name="osu-dreamer",
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