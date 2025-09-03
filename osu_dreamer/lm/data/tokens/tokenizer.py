
from typing import Iterator

import numpy as np

from .decoder import Decoder
from ..parse.beatmap import BeatmapEvents
from ..timed import *
from .tokens import Token, TokenType, VocabConfig, make_vocab

cross = lambda a,b: a[0]*b[1] - a[1]*b[0]

class Tokenizer:
    def __init__(self, config: VocabConfig):
        self.config = config
        self.id_to_token = make_vocab(config)
        self.token_to_id = { token: i for i, token in enumerate(self.id_to_token) }
        self.t = 0
                            
    def encode(self, bm: BeatmapEvents) -> tuple[
        list[int], # beatmap tokens
        list[int], # timestamp @ token
    ]:
        ts, toks = [0], []
        for tok in self._tokenize_timed_objects(bm):
            try:
                toks.append(self.token_to_id[tok])
            except KeyError as e:
                raise Exception(self.t) from e
            ts.append(self.t)
        ts.pop()
        return toks, ts

    def decode(self, beatmap_token_ids: list[int]) -> BeatmapEvents:
        return Decoder(
            self.config,
            [ self.id_to_token[tid] for tid in beatmap_token_ids ]
        ).parse_beatmap_events()
    
    def _tokenize_coordinate(self, p: tuple[int, int]) -> Iterator[Token]:
        assert self.config.x_min <= p[0] < self.config.x_max, p[0]
        assert self.config.y_min <= p[1] < self.config.y_max, p[1]

        coarse_x_bin_size = (self.config.x_max - self.config.x_min) // self.config.coarse_x_bins
        coarse_y_bin_size = (self.config.y_max - self.config.y_min) // self.config.coarse_y_bins

        coarse_x_bin, fine_x = divmod(p[0] - self.config.x_min, coarse_x_bin_size)
        coarse_y_bin, fine_y = divmod(p[1] - self.config.y_min, coarse_y_bin_size)
        yield Token(TokenType.POS_COARSE, (coarse_x_bin, coarse_y_bin))

        fine_x_bin, _ = divmod(fine_x, coarse_x_bin_size // self.config.fine_x_bins)
        fine_y_bin, _ = divmod(fine_y, coarse_y_bin_size // self.config.fine_y_bins)
        yield Token(TokenType.POS_FINE, (fine_x_bin, fine_y_bin))

    def _tokenize_time_shift(self, ms: int) -> Iterator[Token]:
        s, ms = divmod(ms, 1000)
        m, s = divmod(s, 60)
        for _ in range(m):
            self.t += 60*1000
            yield Token(TokenType.TIME_SHIFT_S, 60)
        if s > 0:
            self.t += s*1000
            yield Token(TokenType.TIME_SHIFT_S, s)
        if ms > 0:
            self.t += ms
            yield Token(TokenType.TIME_SHIFT_MS, ms)

    def _tokenize_duration(self, ms: int) -> Iterator[Token]:
        yield from self._tokenize_time_shift(ms)
        yield Token(TokenType.RELEASE)

    def _tokenize_deviation(self, d: float) -> Iterator[Token]:
        assert -np.pi <= d <= np.pi, d
        if d == np.pi: d = -np.pi
        deviation_bin = round( (d+np.pi)/2/np.pi * self.config.DEVIATION_BINS ) % self.config.DEVIATION_BINS
        yield Token(TokenType.DEVIATION, deviation_bin)
    
    def _tokenize_hit_object(self, event: HitObject) -> Iterator[Token]:
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
                yield from self._tokenize_duration(event.duration)
                yield Token(TokenType.SLIDES, min(event.slides, self.config.SLIDES_BINS - 1))
                yield from self._tokenize_coordinate(event.head)
                match event:
                    case PerfectSlider():
                        yield Token(TokenType.PERFECT)
                        yield from self._tokenize_coordinate(event.tail)
                        yield from self._tokenize_deviation(event.deviation)

                    case PolyLineSlider(vertices=vertices):
                        yield Token(TokenType.POLYLINE)
                        for v in vertices:
                            yield from self._tokenize_coordinate(v)

                    case BezierSlider():
                        yield Token(TokenType.BEZIER)
                        head = event.head
                        for seg in event.segments:
                            try:
                                match seg:
                                    case LineSegment():
                                        yield from self._tokenize_linear_segment(head, seg)
                                    case QuadraticSegment():
                                        yield from self._tokenize_quad_segment(head, seg)
                                    case CubicSegment():
                                        yield from self._tokenize_cubic_segment(head, seg)
                            except Exception as e:
                                raise Exception((head, seg)) from e
                            head = seg.q

    def _tokenize_linear_segment(self, head: Coordinate, seg: LineSegment) -> Iterator[Token]:
        yield Token(TokenType.LINEAR)
        yield from self._tokenize_coordinate(seg.q)

    def _tokenize_quad_segment(self, head: Coordinate, seg: QuadraticSegment) -> Iterator[Token]:
        p, q = np.array(head), np.array(seg.q)
        v0, v1 = q - p

        c0, c1 = np.array(seg.c) - p
        c_mag = (c0*c0 + c1*c1) ** .5
        c_dev = float(np.arctan2(v0*c1 - v1*c0, v0*c0 + v1*c1))

        yield Token(TokenType.QUADRATIC)
        yield from self._tokenize_coordinate(seg.q)
        yield from self._tokenize_deviation(c_dev)
        yield from self._tokenize_magnitude(c_mag)

    def _tokenize_cubic_segment(self, head: Coordinate, seg: CubicSegment) -> Iterator[Token]:
        p, q = np.array(head), np.array(seg.q)
        v0, v1 = q - p

        pc0, pc1 = np.array(seg.pc) - p
        pc_mag = (pc0*pc0 + pc1*pc1) ** .5
        pc_dev = float(np.arctan2(v0*pc1 - v1*pc0, v0*pc0 + v1*pc1))
        
        qc0, qc1 = q - np.array(seg.qc)
        qc_mag = (qc0*qc0 + qc1*qc1) ** .5
        qc_dev = float(np.arctan2(v0*qc1 - v1*qc0, v0*qc0 + v1*qc1))

        yield Token(TokenType.CUBIC)
        yield from self._tokenize_coordinate(seg.q)
        yield from self._tokenize_deviation(pc_dev)
        yield from self._tokenize_deviation(qc_dev)
        yield from self._tokenize_magnitude(pc_mag)
        yield from self._tokenize_magnitude(qc_mag)

    def _tokenize_magnitude(self, d: float) -> Iterator[Token]:
        assert self.config.MIN_MAGNITUDE <= d <= self.config.MAX_MAGNITUDE, d
        t = np.log(d/self.config.MIN_MAGNITUDE) / np.log(self.config.MAX_MAGNITUDE/self.config.MIN_MAGNITUDE) # [0,1]
        b = min(self.config.MAGNITUDE_BINS-1, int(t * self.config.MAGNITUDE_BINS))
        yield Token(TokenType.MAGNITUDE, b)
    
    def _tokenize_timed_objects(self, bm: BeatmapEvents) -> Iterator[Token]:
        self.t = 0
        yield Token(TokenType.BOS)
        for t, event in bm.timed:
            time_shift = t - self.t
            if time_shift > 0:
                yield from self._tokenize_time_shift(time_shift)

            match event:
                case Break():
                    yield Token(TokenType.BREAK)
                    yield from self._tokenize_duration(event.duration)
                    
                case HitObject():
                    try:
                        yield from self._tokenize_hit_object(event)
                    except Exception as e:
                        raise Exception(t) from e
        yield Token(TokenType.EOS)