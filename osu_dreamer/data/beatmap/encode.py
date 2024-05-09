
from osu_dreamer.osu.beatmap import Beatmap
from ..load_audio import FrameTimes

from .cursor import cursor_signal, CursorSignal, CURSOR_DIM
from .hit import hit_signal, HitSignal, HIT_DIM

def encode_beatmap(bm: Beatmap, frame_times: FrameTimes) -> tuple[HitSignal, CursorSignal]:
    return (
        hit_signal(bm, frame_times),
        cursor_signal(bm, frame_times),
    )