
import numpy as np

from osu_dreamer.data.load_audio import MS_PER_FRAME

from ..parse.beatmap import BeatmapEvents

from ..timed import *
from .tokens import Token, TokenType, Vocab

class Decoder:

    class UnexpectedToken(Exception): pass

    def __init__(
        self, 
        vocab: Vocab,
        token_ids: list[int],
        ctx_starts: list[int],
    ):
        assert len(token_ids) == len(ctx_starts)
        self.vocab = vocab
        self.tokens = [ vocab.tokens[tid] for tid in token_ids ]
        self.ctx_starts = ctx_starts

        self._idx = -1

    def next_token(self) -> Token:
        self._idx += 1
        return self.tokens[self._idx]
    
    def push_back(self):
        self._idx -= 1

    @property
    def ctx_start(self) -> int:
        return self.ctx_starts[self._idx]

    def parse_token(self, T: TokenType):
        match self.next_token():
            case Token(typ) if typ is T:
                return
            case _:
                raise self.UnexpectedToken()
    
    def parse_token_value(self, T: TokenType):
        match self.next_token():
            case Token(typ, value) if typ is T:
                return value
            case _:
                raise self.UnexpectedToken()

    def parse_beatmap_events(self) -> BeatmapEvents:
        events: list[Timed] = []
        self.parse_token(TokenType.BOS)
        while True:
            try:
                self.parse_token(TokenType.EOS)
                return BeatmapEvents(events)
            except self.UnexpectedToken:
                self.push_back()

            t = self.parse_time()
            match self.next_token():
                case Token(TokenType.BREAK):
                    u = self.parse_release()
                    assert u > t
                    timed = Break(t, u)
                case _:
                    self.push_back()
                    timed = self.parse_hit_object(t)
            events.append(timed)
    
    def parse_coordinate(self) -> tuple[int, int]:
        coarse_x_bin, coarse_y_bin = self.parse_token_value(TokenType.POS_COARSE)
        fine_x_bin, fine_y_bin = self.parse_token_value(TokenType.POS_FINE)

        coarse_x_bin_size = (self.vocab.x_max - self.vocab.x_min) // self.vocab.coarse_x_bins
        coarse_y_bin_size = (self.vocab.y_max - self.vocab.y_min) // self.vocab.coarse_y_bins
        fine_x_bin_size = coarse_x_bin_size // self.vocab.fine_x_bins
        fine_y_bin_size = coarse_y_bin_size // self.vocab.fine_y_bins

        x = self.vocab.x_min + coarse_x_bin * coarse_x_bin_size + (fine_x_bin + .5) * fine_x_bin_size
        y = self.vocab.y_min + coarse_y_bin * coarse_y_bin_size + (fine_y_bin + .5) * fine_y_bin_size
        
        return round(x), round(y)

    def parse_time(self) -> int:
        time_bin = self.parse_token_value(TokenType.TIME)
        return MS_PER_FRAME * (time_bin + self.ctx_start)
    
    def parse_release(self) -> int:
        time = self.parse_time()
        self.parse_token(TokenType.RELEASE)
        return time
    
    def parse_deviation(self) -> float:
        deviation_bin = self.parse_token_value(TokenType.DEVIATION)
        return deviation_bin * 2*np.pi / self.vocab.DEVIATION_BINS - np.pi
        
    def parse_hit_object(self, t: int) -> Timed:
        new_combo, whistle, finish, clap = False, False, False, False
        for _ in range(4):
            match self.next_token():
                case Token(TokenType.NEW_COMBO): new_combo = True
                case Token(TokenType.WHISTLE): whistle = True
                case Token(TokenType.FINISH): finish = True
                case Token(TokenType.CLAP): clap = True
                case _:
                    self.push_back()
                    break

        hit_object_args = new_combo, whistle, finish, clap

        match self.next_token():
            case Token(TokenType.SPINNER):
                u = self.parse_release()
                assert u > t
                return Spinner(t, u, *hit_object_args)
            case Token(TokenType.HIT_CIRCLE):
                return HitCircle(t, *hit_object_args, self.parse_coordinate())
            case Token(TokenType.SLIDER):
                u = self.parse_release()
                assert u > t
                slides: int = self.parse_token_value(TokenType.SLIDES)
                head = self.parse_coordinate()
                slider_args = t, u, *hit_object_args, slides, head
                match self.next_token():
                    case Token(TokenType.PERFECT):
                        tail = self.parse_coordinate()
                        deviation = self.parse_deviation()
                        return PerfectSlider(*slider_args, tail, deviation)
                    case Token(TokenType.POLYLINE):
                        vertices = []
                        while True:
                            try:
                                vertices.append(self.parse_coordinate())
                            except self.UnexpectedToken:
                                self.push_back()
                                break
                        return PolyLineSlider(*slider_args, vertices)
                    case Token(TokenType.BEZIER):
                        segments: list[BezierSegment] = []
                        p = head
                        while True:
                            match self.next_token():
                                case Token(TokenType.LINEAR):
                                    segments.append(self.parse_linear_segment(p))
                                case Token(TokenType.QUADRATIC):
                                    segments.append(self.parse_quadratic_segment(p))
                                case Token(TokenType.CUBIC):
                                    segments.append(self.parse_cubic_segment(p))
                                case _:
                                    self.push_back()
                                    break
                            p = segments[-1].ctrl[-1]
                        return BezierSlider(*slider_args, segments)
                    case _ as tok:
                        raise self.UnexpectedToken(tok)
            case _ as tok:
                raise self.UnexpectedToken(tok)
            
    def parse_linear_segment(self, head: Coordinate) -> BezierSegment:
        return BezierSegment([self.parse_coordinate()])
            
    def parse_quadratic_segment(self, head: Coordinate) -> BezierSegment:

        tail = self.parse_coordinate()
        c_dev = self.parse_deviation()
        c_scale = self.parse_magnitude()

        p, q = np.array(head), np.array(tail)
        v = q - p
        u = np.array([-v[1], v[0]])

        d = v*np.cos(c_dev) + u*np.sin(c_dev)
        c = p + c_scale * d / np.linalg.norm(d)

        return BezierSegment([
            tuple(c.round().astype(int).tolist())
            for c in [c,q]
        ])
            
    def parse_cubic_segment(self, head: Coordinate) -> BezierSegment:

        tail = self.parse_coordinate()
        pc_dev = self.parse_deviation()
        qc_dev = self.parse_deviation()
        pc_scale = self.parse_magnitude()
        qc_scale = self.parse_magnitude()

        p, q = np.array(head), np.array(tail)
        v = q - p
        u = np.array([-v[1], v[0]])

        pd = v*np.cos(pc_dev) + u*np.sin(pc_dev)
        qd = v*np.cos(qc_dev) + u*np.sin(qc_dev)

        pc = p + pc_scale * pd / np.linalg.norm(pd)
        qc = q - qc_scale * qd / np.linalg.norm(qd)

        return BezierSegment([
            tuple(c.round().astype(int).tolist())
            for c in [pc,qc,q]
        ])

    def parse_magnitude(self) -> float:
        b = self.parse_token_value(TokenType.MAGNITUDE)
        t = b / self.vocab.MAGNITUDE_BINS # [0,1)
        return self.vocab.MIN_MAGNITUDE * (self.vocab.MAX_MAGNITUDE / self.vocab.MIN_MAGNITUDE) ** t