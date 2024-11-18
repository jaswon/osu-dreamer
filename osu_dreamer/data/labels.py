
from jaxtyping import Float

import numpy as np

from osu_dreamer.osu.beatmap import Beatmap

NUM_LABELS = 5

def get_labels(bm: Beatmap) -> Float[np.ndarray, f'{NUM_LABELS}']:
    return np.array([bm.sr, bm.ar, bm.od, bm.cs, bm.hp])