
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

    def parse_intermediate_beatmap(self) -> IntermediateBeatmap:
        hp = self.parse_token_value(TokenType.HP_DRAIN_RATE)
        cs = self.parse_token_value(TokenType.CIRCLE_SIZE)
        od = self.parse_token_value(TokenType.OVERALL_DIFFICULTY)
        ar = self.parse_token_value(TokenType.APPROACH_RATE)
        tr = self.parse_token_value(TokenType.SLIDER_TICK_RATE)

        timed = self.parse_timed_objects()
        return IntermediateBeatmap(hp, cs, od, ar, tr, timed)

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
                head = self.parse_coordinate()
                dur = self.parse_duration()
                slides: int = self.parse_token_value(TokenType.SLIDES)
                slider_args = *hit_object_args, dur, slides, head
                match self.next_token():
                    case Token(TokenType.PERFECT):
                        tail = self.parse_coordinate()
                        deviation_bin = self.parse_token_value(TokenType.DEVIATION)
                        deviation = (abs(deviation_bin) - 1) * np.pi / self.config.DEVIATION_BINS
                        return PerfectSlider(*slider_args, tail, deviation if deviation_bin>0 else -deviation)
                    case Token(TokenType.BEZIER):
                        segments = []
                        while True:
                            match self.next_token():
                                case Token(TokenType.LINE):
                                    segments.append(LineSegment(self.parse_coordinate()))
                                case Token(TokenType.CUBIC):
                                    segments.append(CubicSegment(
                                        pc=self.parse_coordinate(),
                                        qc=self.parse_coordinate(),
                                        q=self.parse_coordinate(),
                                    ))
                                case _ as tok:
                                    self.push_back(tok)
                                    break
                        return BezierSlider(*slider_args, segments)
                    case _ as tok:
                        raise self.UnexpectedToken(tok)
            case _ as tok:
                raise self.UnexpectedToken(tok)

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
    
    def parse_coordinate(self) -> tuple[int, int]:
        x = self.parse_token_value(TokenType.X)
        y = self.parse_token_value(TokenType.Y)
        return x,y
    
    def parse_token_value(self, T: TokenType):
        match self.next_token():
            case Token(typ, value) if typ is T:
                return value
            case _ as tok:
                raise self.UnexpectedToken(tok)