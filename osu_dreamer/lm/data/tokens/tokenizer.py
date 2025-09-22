
from typing import Iterator

import numpy as np

from osu_dreamer.data.load_audio import MS_PER_FRAME

from ..parse.beatmap import BeatmapEvents
from ..timed import *
from .tokens import Token, TokenType, Vocab

class Tokenizer:
    def __init__(self, vocab: Vocab, bm: BeatmapEvents):
        self.vocab = vocab
        self.bm_tokens = list(self._tokenize_timed_objects(bm))

    def encode(self, start_frame: int) -> list[tuple[int, int, int, int]]:
        x,y,t = 256, 192, 0

        seq: list[tuple[int, int, int, int]] = [(self.vocab.BOS, x, y, t)]
        encode = False
        for tok in self.bm_tokens:
            if tok.typ == TokenType.TIME:
                t = tok.value // MS_PER_FRAME - start_frame
                tok = Token(TokenType.TIME)
                
                if t < 0:
                    continue
                if t >= self.vocab.time_bins:
                    break
                encode = True

            if tok.typ == TokenType.POS:
                x,y = tok.value
                tok = Token(TokenType.POS)

            if encode:
                seq.append((self.vocab.ids[tok], x, y, t))

        seq.append((self.vocab.TIME, x, y, self.vocab.time_bins))
        return seq
    
    def _tokenize_coordinate(self, p: tuple[int, int]) -> Iterator[Token]:
        assert self.vocab.x_min <= p[0] < self.vocab.x_max, p[0]
        assert self.vocab.y_min <= p[1] < self.vocab.y_max, p[1]
        yield Token(TokenType.POS, p)

    def _tokenize_time(self, t: int) -> Iterator[Token]:
        yield Token(TokenType.TIME, t)

    def _tokenize_release(self, u: int) -> Iterator[Token]:
        yield from self._tokenize_time(u)
        yield Token(TokenType.RELEASE)

    def _tokenize_deviation(self, d: float) -> Iterator[Token]:
        assert -np.pi <= d <= np.pi, d
        if d == np.pi: d = -np.pi
        deviation_bin = round( (d+np.pi)/2/np.pi * self.vocab.DEVIATION_BINS ) % self.vocab.DEVIATION_BINS
        yield Token(TokenType.DEVIATION, deviation_bin)
    
    def _tokenize_hit_object(self, event: HitObject) -> Iterator[Token]:
        if event.new_combo: yield Token(TokenType.NEW_COMBO)
        if event.whistle: yield Token(TokenType.WHISTLE)
        if event.finish: yield Token(TokenType.FINISH)
        if event.clap: yield Token(TokenType.CLAP)

        match event:
            case Spinner():
                yield Token(TokenType.SPINNER)
                yield from self._tokenize_release(event.u)

            case HitCircle():
                yield Token(TokenType.HIT_CIRCLE)
                yield from self._tokenize_coordinate(event.p)

            case Slider():
                yield Token(TokenType.SLIDER)
                yield from self._tokenize_release(event.u)
                yield Token(TokenType.SLIDES, min(event.slides, self.vocab.SLIDES_BINS - 1))
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
                                match seg.ctrl:
                                    case [q]:
                                        yield from self._tokenize_linear_segment(head, q)
                                    case [c,q]:
                                        yield from self._tokenize_quad_segment(head, c, q)
                                    case [c1,c2,q]:
                                        yield from self._tokenize_cubic_segment(head, c1, c2, q)
                            except Exception as e:
                                raise Exception((head, seg)) from e
                            head = q

    def _tokenize_linear_segment(self, head: Coordinate, q: Coordinate) -> Iterator[Token]:
        yield Token(TokenType.LINEAR)
        yield from self._tokenize_coordinate(q)

    def _tokenize_quad_segment(self, head: Coordinate, c: Coordinate, q_: Coordinate) -> Iterator[Token]:
        p, q = np.array(head), np.array(q_)
        v0, v1 = q - p

        c0, c1 = np.array(c) - p
        c_mag = (c0*c0 + c1*c1) ** .5
        c_dev = float(np.arctan2(v0*c1 - v1*c0, v0*c0 + v1*c1))

        yield Token(TokenType.QUADRATIC)
        yield from self._tokenize_coordinate(q_)
        yield from self._tokenize_deviation(c_dev)
        yield from self._tokenize_magnitude(c_mag)

    def _tokenize_cubic_segment(self, head: Coordinate, pc_: Coordinate, qc_: Coordinate, q_: Coordinate) -> Iterator[Token]:
        p, q = np.array(head), np.array(q_)
        v0, v1 = q - p

        pc0, pc1 = np.array(pc_) - p
        pc_mag = (pc0*pc0 + pc1*pc1) ** .5
        pc_dev = float(np.arctan2(v0*pc1 - v1*pc0, v0*pc0 + v1*pc1))
        
        qc0, qc1 = q - np.array(qc_)
        qc_mag = (qc0*qc0 + qc1*qc1) ** .5
        qc_dev = float(np.arctan2(v0*qc1 - v1*qc0, v0*qc0 + v1*qc1))

        yield Token(TokenType.CUBIC)
        yield from self._tokenize_coordinate(q_)
        yield from self._tokenize_deviation(pc_dev)
        yield from self._tokenize_deviation(qc_dev)
        yield from self._tokenize_magnitude(pc_mag)
        yield from self._tokenize_magnitude(qc_mag)

    def _tokenize_magnitude(self, d: float) -> Iterator[Token]:
        assert self.vocab.MIN_MAGNITUDE <= d <= self.vocab.MAX_MAGNITUDE, d
        t = np.log(d/self.vocab.MIN_MAGNITUDE) / np.log(self.vocab.MAX_MAGNITUDE/self.vocab.MIN_MAGNITUDE) # [0,1]
        b = min(self.vocab.MAGNITUDE_BINS-1, int(t * self.vocab.MAGNITUDE_BINS))
        yield Token(TokenType.MAGNITUDE, b)
    
    def _tokenize_timed_objects(self, bm: BeatmapEvents) -> Iterator[Token]:
        for event in bm.timed:
            yield from self._tokenize_time(event.t)
            match event:
                case Break():
                    yield Token(TokenType.BREAK)
                    yield from self._tokenize_release(event.u)
                    
                case HitObject():
                    try:
                        yield from self._tokenize_hit_object(event)
                    except Exception as e:
                        raise Exception(event.t) from e