
from typing import Generator

import numpy as np

from .decoder import Decoder
from .intermediate import IntermediateBeatmap
from .timed import *
from .tokens import Token, TokenType, VocabConfig, make_vocab

class Tokenizer:
    def __init__(self, config: VocabConfig):
        self.config = config
        self.id_to_token = make_vocab(config)
        self.token_to_id = { token: i for i, token in enumerate(self.id_to_token) }

    def _tokenize_map_features(self, bm: IntermediateBeatmap) -> Generator[Token]:
        yield Token(TokenType.HP_DRAIN_RATE, round(bm.hp_drain_rate, 1))
        yield Token(TokenType.CIRCLE_SIZE, round(bm.circle_size, 1))
        yield Token(TokenType.OVERALL_DIFFICULTY, round(bm.overall_difficulty, 1))
        yield Token(TokenType.APPROACH_RATE, round(bm.approach_rate, 1))
        yield Token(TokenType.SLIDER_TICK_RATE, bm.slider_tick_rate)
    
    def _tokenize_coordinate(self, p: tuple[int, int]) -> Generator[Token]:
        yield Token(TokenType.X, p[0])
        yield Token(TokenType.Y, p[1])

    def _tokenize_time_shift(self, ms: int) -> Generator[Token]:
        if ms >= 1000:
            s, ms = divmod(ms, 1000)
            yield Token(TokenType.TIME_SHIFT_S, s)
        if ms > 0:
            yield Token(TokenType.TIME_SHIFT_MS, ms)

    def _tokenize_duration(self, ms: int) -> Generator[Token]:
        yield from self._tokenize_time_shift(ms)
        yield Token(TokenType.RELEASE)
    
    def _tokenize_hit_object(self, event: HitObject) -> Generator[Token]:
        if event.new_combo: yield Token(TokenType.NEW_COMBO)
        if event.whistle: yield Token(TokenType.WHISTLE)
        if event.finish: yield Token(TokenType.FINISH)
        if event.clap: yield Token(TokenType.CLAP)

        match event:
            case Spinner():
                yield Token(TokenType.SPINNER)
                yield from self._tokenize_duration(event.duration)

            case HitCircle():
                yield Token(TokenType.HIT_CIRCLE)
                yield from self._tokenize_coordinate(event.p)

            case Slider():
                yield Token(TokenType.SLIDER)
                yield from self._tokenize_coordinate(event.head)
                yield from self._tokenize_duration(event.duration)
                yield Token(TokenType.SLIDES, min(event.slides, self.config.SLIDES_BINS - 1))
                match event:
                    case PerfectSlider():
                        yield Token(TokenType.PERFECT)
                        yield from self._tokenize_coordinate(event.tail)
                        deviation_bin = 1+int(abs(event.deviation)*self.config.DEVIATION_BINS/np.pi)
                        yield Token(TokenType.DEVIATION, deviation_bin if event.deviation > 0 else -deviation_bin)

                    case BezierSlider():
                        yield Token(TokenType.BEZIER)
                        for seg in event.segments:
                            if isinstance(seg, LineSegment):
                                yield Token(TokenType.LINE)
                                yield from self._tokenize_coordinate(seg.q)
                            else:
                                yield Token(TokenType.CUBIC)
                                yield from self._tokenize_coordinate(seg.pc)
                                yield from self._tokenize_coordinate(seg.qc)
                                yield from self._tokenize_coordinate(seg.q)
    
    def _tokenize_timed_objects(self, bm: IntermediateBeatmap) -> Generator[Token]:
        last_time = 0
        for t, event in bm.timed:
            time_shift = t - last_time
            if time_shift > 0:
                yield from self._tokenize_time_shift(time_shift)
                last_time = t

            match event:
                case BeatLen():
                    yield Token(TokenType.BEAT_LEN, round(60_000/round(60_000/event.ms),2))

                case Break():
                    yield Token(TokenType.BREAK)
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
        return Decoder(
            self.config,
            [
                self.id_to_token[tid]
                for tid in (*context_prelude_token_ids, *beatmap_token_ids)
            ]
        ).parse_intermediate_beatmap()