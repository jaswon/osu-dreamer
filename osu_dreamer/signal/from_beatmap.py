
import numpy as np

from osu_dreamer.osu.hit_objects import Circle, Slider, Spinner

from .util import smooth_hit

__all__ = ['from_beatmap', 'MAP_SIGNAL_DIM', 'timing_signal']

MAP_SIGNAL_DIM = 6
    
def timing_signal(beatmap, frame_times: "L,") -> ",L":
    pass


def hit_signal(beatmap, frame_times: "L,") -> "4,L":
    """
    returns an array encoding the hits occurring at the times represented by `frames`
    - [0] represents hits
    - [1] represents slider holds
    - [2] represents spinner holds
    - [3] represents new combos

    - `frame_times`: array of times at each frame in ms
    """

    sig = np.zeros((4, len(frame_times)))
    for ho in beatmap.hit_objects:
        if isinstance(ho, Circle):
            sig[0] += smooth_hit(frame_times, ho.t)
        elif isinstance(ho, Slider):
            sig[1] += smooth_hit(frame_times, (ho.t, ho.t+int(beatmap.slider_duration(ho))))
        else: # Spinner
            sig[2] += smooth_hit(frame_times, (ho.t, ho.u))

        if ho.new_combo:
            sig[3] += smooth_hit(frame_times, ho.t)

    return sig


def cursor_signal(beatmap, frame_times: "L,") -> "2,L":
    """
    return [2,L] where [{0,1},i] is the {x,y} position at the times represented by `frames`

    - `frame_times`: array of times at each frame in ms
    """
    return np.array([ beatmap.cursor(t)[0] for t in frame_times ]).T


def from_beatmap(beatmap, frame_times: "L,") -> "6,L":
    """
    returns a [6,L] scaled to [-1,1]

    - `frame_times`: array of times at each frame in ms
    """
    hits: "4,L" = hit_signal(beatmap, frame_times)
    cursor: "2,L" = cursor_signal(beatmap, frame_times) / np.array([[512],[384]])

    return np.concatenate([hits, cursor], axis=0) * 2 - 1
    