
import numpy as np

from .intermediate import IntermediateBeatmap

from .timed import *
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

    def next_token(self) -> Token:
        if len(self.push_back_stack):
            return self.push_back_stack.pop()
        return next(self.tokens)
    
    def push_back(self, tok: Token):
        self.push_back_stack.append(tok)
    
    def parse_token_value(self, T: TokenType):
        match self.next_token():
            case Token(typ, value) if typ is T:
                return value
            case _ as tok:
                raise self.UnexpectedToken(tok)

    def parse_intermediate_beatmap(self) -> IntermediateBeatmap:
        hp = self.parse_token_value(TokenType.HP_DRAIN_RATE)
        cs = self.parse_token_value(TokenType.CIRCLE_SIZE)
        od = self.parse_token_value(TokenType.OVERALL_DIFFICULTY)
        ar = self.parse_token_value(TokenType.APPROACH_RATE)
        tr = self.parse_token_value(TokenType.SLIDER_TICK_RATE)

        timed = self.parse_timed_objects()
        return IntermediateBeatmap(hp, cs, od, ar, tr, timed)
    
    def parse_coordinate(self) -> tuple[int, int]:
        x_bin = self.parse_token_value(TokenType.X)
        y_bin = self.parse_token_value(TokenType.Y)

        x = self.config.x_min + (x_bin + .5) / self.config.x_bins * (self.config.x_max + 1 - self.config.x_min)
        y = self.config.y_min + (y_bin + .5) / self.config.y_bins * (self.config.y_max + 1 - self.config.y_min)
        return round(x),round(y)

    def parse_time_shift(self) -> int:
        time_shift = 0
        match self.next_token():
            case Token(TokenType.TIME_SHIFT_MS, ms):
                time_shift += ms
            case Token(TokenType.TIME_SHIFT_S, s):
                time_shift += 1000 * s
                match self.next_token():
                    case Token(TokenType.TIME_SHIFT_MS, ms):
                        time_shift += ms
                    case _ as tok:
                        self.push_back(tok)
            case _ as tok:
                self.push_back(tok)
        return time_shift
    
    def parse_duration(self) -> int:
        time_shift = self.parse_time_shift()
        match self.next_token():
            case Token(TokenType.RELEASE):
                pass
            case _ as tok:
                raise self.UnexpectedToken(tok)
        return time_shift
    
    def parse_deviation(self) -> float:
        deviation_bin = self.parse_token_value(TokenType.DEVIATION)
        deviation = (abs(deviation_bin) - 1) * np.pi / self.config.DEVIATION_BINS
        return deviation if deviation_bin > 0 else -deviation
        
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
                slides: int = self.parse_token_value(TokenType.SLIDES)
                head = self.parse_coordinate()
                slider_args = *hit_object_args, dur, slides, head
                match self.next_token():
                    case Token(TokenType.PERFECT):
                        tail = self.parse_coordinate()
                        deviation = self.parse_deviation()
                        return PerfectSlider(*slider_args, tail, deviation)
                    case Token(TokenType.BEZIER):
                        segments: list[BezierSegment] = []
                        p = head
                        while True:
                            match self.next_token():
                                case Token(TokenType.LINE):
                                    segments.append(LineSegment(self.parse_coordinate()))
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
            
    def parse_cubic_segment(self, head: Coordinate) -> CubicSegment:

        tail = self.parse_coordinate()
        pc_dev = self.parse_deviation()
        qc_dev = self.parse_deviation()
        pc_scale = self.parse_magnitude()
        qc_scale = self.parse_magnitude()

        p, q = np.array(head), np.array(tail)
        v = q - p
        u = np.array([-v[1], v[0]])

        pc = p + pc_scale * (v*np.cos(pc_dev) + u*np.sin(pc_dev))
        qc = q - qc_scale * (v*np.cos(qc_dev) + u*np.sin(qc_dev))

        return CubicSegment(
            pc = tuple(pc.round().astype(int).tolist()),
            qc = tuple(qc.round().astype(int).tolist()),
            q = tuple(q.round().astype(int).tolist()),
        )

    def parse_magnitude(self) -> float:
        b = self.parse_token_value(TokenType.MAGNITUDE)
        t = b / self.config.MAGNITUDE_BINS # [0,1)
        return self.config.MIN_MAGNITUDE * (self.config.MAX_MAGNITUDE / self.config.MIN_MAGNITUDE) ** t

    def parse_timed_objects(self) -> list[tuple[int, Timed]]:
        events: list[tuple[int, Timed]] = []
        cur_time = 0
        try:
            while True:
                cur_time += self.parse_time_shift()
                obj_time = cur_time
                match self.next_token():
                    case Token(TokenType.BEAT_LEN, ms):
                        timed = BeatLen(ms)
                    case Token(TokenType.BREAK):
                        timed = Break(self.parse_duration())
                    case _ as tok:
                        self.push_back(tok)
                        timed = self.parse_hit_object()
                events.append((obj_time, timed))
        except StopIteration:
            return events