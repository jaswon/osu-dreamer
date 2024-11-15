
from jaxtyping import Float

from pathlib import Path

import numpy as np
import rosu_pp_py as rosu

from osu_dreamer.osu.beatmap import Beatmap

NUM_LABELS = 5

perf = rosu.Performance()

def get_labels(map_file: Path, bm: Beatmap) -> Float[np.ndarray, f'{NUM_LABELS}']:
    diff_attrs = perf.calculate(rosu.Beatmap(path=str(map_file))).difficulty
    diff_labels = np.array([diff_attrs.stars, bm.ar, bm.od, bm.cs, bm.hp])
    return diff_labels