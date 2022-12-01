from setuptools import setup

setup(
    name="osu-dreamer",
    version="3.0",
    python_requires='>=3.7.15',
    packages=[
        "osu_dreamer",
        "osu_dreamer.model",
        "osu_dreamer.osu",
        "osu_dreamer.signal",
    ],
    install_requires=[
        "bezier",
        "librosa",
        "tqdm",
        "torch",
        "torchaudio",
        "einops",
        "pytorch-lightning",
        "jsonargparse[signatures]",
        "mutagen",
    ],
    extras_require={
        "dev": ["matplotlib"],
    },
)
