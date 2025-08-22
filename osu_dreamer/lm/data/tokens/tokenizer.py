
from itertools import chain
from typing import Generator
import numpy as np

from .intermediate import IntermediateBeatmap
from .timed import *

# Vocabulary Definition

SPECIAL_TOKENS = [
    '<pad>',
    '<bos>',
    '<eos>',
]

TIME_SHIFT_BINS = 1000
X_BINS = 512
Y_BINS = 384
DIFFICULTY_BINS = 101 # 0-100 for 0.0-10.0
SLIDER_TICK_RATE_BINS = 8 # 1-8
VELOCITY_BINS = 100 # 0.1-10.0 -> 1-100
DURATION_BINS = 5000 # 0-5000ms
SLIDES_BINS = 16 # 0-15
DEVIATION_BINS = 64 # -pi to pi

EVENT_TOKENS = [
    'BREAK',
    'HIT_CIRCLE',
    'SPINNER',
    'SLIDER', 
    'PERFECT',
    'BEZIER',
    'LINE',
    'CUBIC',
]

FLAG_TOKENS = [
    'NEW_COMBO',
    'WHISTLE',
    'FINISH',
    'CLAP',
]

ID_TO_TOKEN = list(chain(
    SPECIAL_TOKENS,
    EVENT_TOKENS,
    FLAG_TOKENS,
    [f'TIME_SHIFT_{i}' for i in range(TIME_SHIFT_BINS)],
    [f'X_{i}' for i in range(X_BINS)],
    [f'Y_{i}' for i in range(Y_BINS)],
    # Map-level features
    [f'HP_DRAIN_RATE_{i}' for i in range(DIFFICULTY_BINS)],
    [f'CIRCLE_SIZE_{i}' for i in range(DIFFICULTY_BINS)],
    [f'OVERALL_DIFFICULTY_{i}' for i in range(DIFFICULTY_BINS)],
    [f'APPROACH_RATE_{i}' for i in range(DIFFICULTY_BINS)],
    [f'SLIDER_TICK_RATE_{i}' for i in range(1, SLIDER_TICK_RATE_BINS + 1)],
    # Event-level values
    [f'BEAT_LEN_{i}' for i in range(40, 1001)], # ms_per_beat
    [f'SLIDER_VEL_{i}' for i in range(1, VELOCITY_BINS + 1)],
    [f'DURATION_{i}' for i in range(DURATION_BINS)],
    [f'SLIDES_{i}' for i in range(SLIDES_BINS)],
    [f'DEVIATION_{i}' for i in range(DEVIATION_BINS)],
))
TOKEN_TO_ID = {tok: i for i, tok in enumerate(ID_TO_TOKEN)}

def quantize(value, min_val, max_val, bins):
    return int(np.clip(((value - min_val) / (max_val - min_val)), 0, 1) * (bins - 1))

class Tokenizer:
    def __init__(self):
        self.token_to_id = TOKEN_TO_ID
        self.id_to_token = ID_TO_TOKEN

    def _tokenize_map_features(self, bm: IntermediateBeatmap) -> Generator[str]:
        yield f'HP_DRAIN_RATE_{int(round(bm.hp_drain_rate * 10))}'
        yield f'CIRCLE_SIZE_{int(round(bm.circle_size * 10))}'
        yield f'OVERALL_DIFFICULTY_{int(round(bm.overall_difficulty * 10))}'
        yield f'APPROACH_RATE_{int(round(bm.approach_rate * 10))}'
        yield f'SLIDER_TICK_RATE_{int(bm.slider_tick_rate)}'
    
    def _tokenize_coordinate(self, p: tuple[int, int]) -> Generator[str]:
        yield f'X_{p[0]}'
        yield f'Y_{p[1]}'

    def _tokenize_duration(self, d: int) -> Generator[str]:
        yield f'DURATION_{min(d, DURATION_BINS - 1)}'
    
    def _tokenize_hit_object(self, event: HitObject) -> Generator[str]:
        if event.new_combo: yield 'NEW_COMBO'
        if event.whistle: yield 'WHISTLE'
        if event.finish: yield 'FINISH'
        if event.clap: yield 'CLAP'

        match event:
            case Spinner():
                yield 'SPINNER'
                yield from self._tokenize_duration(event.duration)

            case HitCircle():
                yield 'HIT_CIRCLE'
                yield from self._tokenize_coordinate(event.p)

            case Slider():
                yield 'SLIDER'
                yield from self._tokenize_coordinate(event.head)
                yield from self._tokenize_duration(event.duration)
                yield f'SLIDES_{min(event.slides, SLIDES_BINS - 1)}'
                match event:
                    case PerfectSlider():
                        yield 'PERFECT'
                        yield from self._tokenize_coordinate(event.tail)
                        yield f'DEVIATION_{quantize(event.deviation, -np.pi, np.pi, DEVIATION_BINS)}'

                    case BezierSlider():
                        yield 'BEZIER'
                        for seg in event.segments:
                            if isinstance(seg, LineSegment):
                                yield 'LINE'
                                yield from self._tokenize_coordinate(seg.q)
                            else:
                                yield 'CUBIC'
                                yield from self._tokenize_coordinate(seg.pc)
                                yield from self._tokenize_coordinate(seg.qc)
                                yield from self._tokenize_coordinate(seg.q)
    
    def _tokenize_timed_objects(self, bm: IntermediateBeatmap) -> Generator[str]:
        last_time = 0
        for t, event in bm.timed:
            time_shift = t - last_time
            if time_shift > 0:
                yield f'TIME_SHIFT_{min(time_shift, TIME_SHIFT_BINS - 1)}'
                last_time = t

            match event:
                case BeatLen():
                    yield f'BEAT_LEN_{int(np.clip(event.ms, 40, 1000))}'

                case SliderVel():
                    yield f'SLIDER_VEL_{int(np.clip(event.vel * 10, 1, VELOCITY_BINS))}'

                case Break():
                    yield 'BREAK'
                    yield from self._tokenize_duration(event.duration)
                    
                case HitObject():
                    yield from self._tokenize_hit_object(event)
                            

    def encode(self, bm: IntermediateBeatmap) -> tuple[
        list[int], # context prelude
        list[int], # beatmap tokens
    ]:
        return (
            [ self.token_to_id[tok] for tok in self._tokenize_map_features(bm) ],
            [ self.token_to_id[tok] for tok in self._tokenize_timed_objects(bm) ],
        )

    def decode(
        self, 
        context_prelude_token_ids: list[int],
        beatmap_token_ids: list[int],
    ) -> IntermediateBeatmap:
        # TODO: Implement decoding logic
        ...