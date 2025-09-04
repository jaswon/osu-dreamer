
import numpy as np

from osu_dreamer.data.load_audio import MS_PER_FRAME

from ..parse.beatmap import BeatmapEvents

from ..timed import *
from .tokens import Token, TokenType, VocabConfig

class Decoder:

    class UnexpectedToken(Exception): pass

    def __init__(
        self, 
        config: VocabConfig, 
        tokens: list[Token],
    ):
        self.config = config
        self.tokens = iter(tokens)
        self.push_back_stack = []
        self.t = 0

        # compute valid time shifts
        self.time_shifts = []
        stride = MS_PER_FRAME
        for r_past, r_future in config.context_radii:
            for i in range(r_future):
                self.time_shifts.append(stride * (i+1))
            stride *= 1 + r_past + r_future

    def next_token(self) -> Token:
        if len(self.push_back_stack):
            return self.push_back_stack.pop()
        return next(self.tokens)
    
    def push_back(self, tok: Token):
        self.push_back_stack.append(tok)

    def parse_token(self, T: TokenType):
        match self.next_token():
            case Token(typ) if typ is T:
                return
            case _ as tok:
                raise self.UnexpectedToken(tok)
    
    def parse_token_value(self, T: TokenType):
        match self.next_token():
            case Token(typ, value) if typ is T:
                return value
            case _ as tok:
                raise self.UnexpectedToken(tok)

    def parse_beatmap_events(self) -> BeatmapEvents:
        events: list[tuple[int, Timed]] = []
        self.t = 0
        self.parse_token(TokenType.BOS)
        while True:
            try:
                self.parse_token(TokenType.EOS)
                return BeatmapEvents(events)
            except self.UnexpectedToken as e:
                self.push_back(e.args[0])

            self.parse_time_shift()
            obj_time = self.t
            match self.next_token():
                case Token(TokenType.BREAK):
                    timed = Break(self.parse_duration())
                case _ as tok:
                    self.push_back(tok)
                    timed = self.parse_hit_object()
            events.append((obj_time, timed))
    
    def parse_coordinate(self) -> tuple[int, int]:
        coarse_x_bin, coarse_y_bin = self.parse_token_value(TokenType.POS_COARSE)
        fine_x_bin, fine_y_bin = self.parse_token_value(TokenType.POS_FINE)

        coarse_x_bin_size = (self.config.x_max - self.config.x_min) // self.config.coarse_x_bins
        coarse_y_bin_size = (self.config.y_max - self.config.y_min) // self.config.coarse_y_bins
        fine_x_bin_size = coarse_x_bin_size // self.config.fine_x_bins
        fine_y_bin_size = coarse_y_bin_size // self.config.fine_y_bins

        x = self.config.x_min + coarse_x_bin * coarse_x_bin_size + (fine_x_bin + .5) * fine_x_bin_size
        y = self.config.y_min + coarse_y_bin * coarse_y_bin_size + (fine_y_bin + .5) * fine_y_bin_size
        
        return round(x), round(y)

    def parse_time_shift(self) -> int:
        time_shift = 0

        while True:
            match self.next_token():
                case Token(TokenType.TIME_SHIFT, i):
                    time_shift += self.time_shifts[i]
                case _ as tok:
                    self.push_back(tok)
                    break
                
        self.t += time_shift
        return time_shift
    
    def parse_duration(self) -> int:
        time_shift = self.parse_time_shift()
        self.parse_token(TokenType.RELEASE)
        return time_shift
    
    def parse_deviation(self) -> float:
        deviation_bin = self.parse_token_value(TokenType.DEVIATION)
        return deviation_bin * 2*np.pi / self.config.DEVIATION_BINS - np.pi
        
    def parse_hit_object(self) -> Timed:
        new_combo, whistle, finish, clap = False, False, False, False
        while True:
            match self.next_token():
                case Token(TokenType.NEW_COMBO): new_combo = True
                case Token(TokenType.WHISTLE): whistle = True
                case Token(TokenType.FINISH): finish = True
                case Token(TokenType.CLAP): clap = True
                case _ as tok:
                    self.push_back(tok)
                    break

        hit_object_args = new_combo, whistle, finish, clap

        match self.next_token():
            case Token(TokenType.SPINNER):
                return Spinner(*hit_object_args, self.parse_duration())
            case Token(TokenType.HIT_CIRCLE):
                return HitCircle(*hit_object_args, self.parse_coordinate())
            case Token(TokenType.SLIDER):
                dur = self.parse_duration()
                assert dur > 0
                slides: int = self.parse_token_value(TokenType.SLIDES)
                head = self.parse_coordinate()
                slider_args = *hit_object_args, dur, slides, head
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
                            except self.UnexpectedToken as e:
                                self.push_back(e.args[0])
                                break
                        return PolyLineSlider(*slider_args, vertices)
                    case Token(TokenType.BEZIER):
                        segments: list[BezierSegment] = []
                        p = head
                        while True:
                            match self.next_token():
                                case Token(TokenType.LINEAR):
                                    segments.append(LineSegment(self.parse_coordinate()))
                                case Token(TokenType.QUADRATIC):
                                    segments.append(self.parse_quadratic_segment(p))
                                case Token(TokenType.CUBIC):
                                    segments.append(self.parse_cubic_segment(p))
                                case _ as tok:
                                    self.push_back(tok)
                                    break
                            p = segments[-1].q
                        return BezierSlider(*slider_args, segments)
                    case _ as tok:
                        raise self.UnexpectedToken(tok)
            case _ as tok:
                raise self.UnexpectedToken(tok)
            
    def parse_quadratic_segment(self, head: Coordinate) -> QuadraticSegment:

        tail = self.parse_coordinate()
        c_dev = self.parse_deviation()
        c_scale = self.parse_magnitude()

        p, q = np.array(head), np.array(tail)
        v = q - p
        u = np.array([-v[1], v[0]])

        d = v*np.cos(c_dev) + u*np.sin(c_dev)
        c = p + c_scale * d / np.linalg.norm(d)

        return QuadraticSegment(
            q = tuple(q.round().astype(int).tolist()),
            c = tuple(c.round().astype(int).tolist()),
        )
            
    def parse_cubic_segment(self, head: Coordinate) -> CubicSegment:

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

        return CubicSegment(
            q = tuple(q.round().astype(int).tolist()),
            pc = tuple(pc.round().astype(int).tolist()),
            qc = tuple(qc.round().astype(int).tolist()),
        )

    def parse_magnitude(self) -> float:
        b = self.parse_token_value(TokenType.MAGNITUDE)
        t = b / self.config.MAGNITUDE_BINS # [0,1)
        return self.config.MIN_MAGNITUDE * (self.config.MAX_MAGNITUDE / self.config.MIN_MAGNITUDE) ** t