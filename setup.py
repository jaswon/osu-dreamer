from setuptools import setup

setup(
    name="osu-dreamer",
    version="3.1",
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
        "einops",
        "pytorch-lightning",
        "jsonargparse[signatures]",
        "mutagen",
        "requests",
        "flask",
        "flask_cors",
        "unidecode",
        "fake_useragent"
    ],
    extras_require={
        "dev": ["matplotlib"],
    },
)
