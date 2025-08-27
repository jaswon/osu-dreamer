
from typing import NamedTuple, Any
from dataclasses import dataclass
from enum import Enum

class CustomReprEnum(Enum):
    def __repr__(self):
        return self.name
    
TokenType = CustomReprEnum('TokenType', [
    # control
    'PAD',
    'BOS',
    'EOS',

    # event types
    'BREAK',
    'HIT_CIRCLE',
    'SPINNER',
    'SLIDER',
    'RELEASE',

    # slider tokens
    'PERFECT',
    'BEZIER',
    'LINE',
    'CUBIC',
    'SLIDES',
    'DEVIATION',
    'MAGNITUDE',

    # hit object flags
    'NEW_COMBO',
    'WHISTLE',
    'FINISH',
    'CLAP',

    # numerals
    'TIME_SHIFT_MS',
    'TIME_SHIFT_S',
    'X',
    'Y',

    # map-level features
    'HP_DRAIN_RATE',
    'CIRCLE_SIZE',
    'OVERALL_DIFFICULTY',
    'APPROACH_RATE',
    'SLIDER_TICK_RATE',
])

class Token(NamedTuple):
    typ: TokenType
    value: Any = None

    def __repr__(self):
        return f"{repr(self.typ)}({"" if self.value is None else repr(self.value)})"
    

@dataclass
class VocabConfig:
    TIME_SHIFT_SECONDS: int = 60

    x_min: int = 0-256
    x_max: int = 512+256
    x_bins: int = 256
    y_min: int = 0-192
    y_max: int = 384+192
    y_bins: int = 192

    SLIDES_BINS: int = 16
    DEVIATION_BINS: int = 64

    MAGNITUDE_BINS: int = 64
    MIN_MAGNITUDE: float = .05
    MAX_MAGNITUDE: float = 50

    DIFFICULTY_BINS: int = 101 # 0-100 for 0.0-10.0
    SLIDER_TICK_RATE_BINS: int = 8 # 1-8

def make_vocab(config: VocabConfig) -> tuple[Token, ...]:
    return (
        # control
        Token(TokenType.PAD),
        Token(TokenType.BOS),
        Token(TokenType.EOS),

        # event types
        Token(TokenType.BREAK),
        Token(TokenType.HIT_CIRCLE),
        Token(TokenType.SPINNER),
        Token(TokenType.SLIDER),
        Token(TokenType.RELEASE),

        # slider tokens
        Token(TokenType.PERFECT),
        Token(TokenType.BEZIER),
        Token(TokenType.LINE),
        Token(TokenType.CUBIC),
        *[ Token(TokenType.SLIDES,i) for i in range(config.SLIDES_BINS) ],
        *[
            Token(TokenType.DEVIATION,sgn*(1+s))
            for sgn in [-1, 1]
            for s in range(config.DEVIATION_BINS)
        ],
        *[
            Token(TokenType.MAGNITUDE,m)
            for m in range(config.MAGNITUDE_BINS)
        ],

        # hit object flags
        Token(TokenType.NEW_COMBO),
        Token(TokenType.WHISTLE),
        Token(TokenType.FINISH),
        Token(TokenType.CLAP),

        # numerals
        *[ Token(TokenType.TIME_SHIFT_MS,i) for i in range(1000) ],
        *[ Token(TokenType.TIME_SHIFT_S,i) for i in range(config.TIME_SHIFT_SECONDS) ],
        *[ Token(TokenType.X,i) for i in range(config.x_bins) ],
        *[ Token(TokenType.Y,i) for i in range(config.y_bins) ],

        # map-level features
        *[ Token(TokenType.HP_DRAIN_RATE,i/10) for i in range(config.DIFFICULTY_BINS)],
        *[ Token(TokenType.CIRCLE_SIZE,i/10) for i in range(config.DIFFICULTY_BINS)],
        *[ Token(TokenType.OVERALL_DIFFICULTY,i/10) for i in range(config.DIFFICULTY_BINS)],
        *[ Token(TokenType.APPROACH_RATE,i/10) for i in range(config.DIFFICULTY_BINS)],
        *[ Token(TokenType.SLIDER_TICK_RATE,i+1) for i in range(config.SLIDER_TICK_RATE_BINS)],
    )