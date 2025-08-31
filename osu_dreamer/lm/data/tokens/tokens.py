
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
    'POS_COARSE',
    'POS_FINE',
])

class Token(NamedTuple):
    typ: TokenType
    value: Any = None

    def __repr__(self):
        return f"{repr(self.typ)}({"" if self.value is None else repr(self.value)})"
    

@dataclass
class VocabConfig:
    TIME_SHIFT_SECONDS: int = 60

    x_min: int = 0-128
    x_max: int = 512+128
    y_min: int = 0-96
    y_max: int = 384+96
    coarse_x_bins: int = 16
    coarse_y_bins: int = 16
    fine_x_bins: int = 12
    fine_y_bins: int = 9

    SLIDES_BINS: int = 16
    DEVIATION_BINS: int = 64

    MAGNITUDE_BINS: int = 64
    MIN_MAGNITUDE: float = .05
    MAX_MAGNITUDE: float = 50
    

def make_vocab(config: VocabConfig) -> tuple[Token, ...]:
    
    coarse_x_bin_size, rem = divmod(config.x_max - config.x_min, config.coarse_x_bins)
    assert rem == 0
    assert coarse_x_bin_size % config.fine_x_bins == 0

    coarse_y_bin_size, rem = divmod(config.y_max - config.y_min, config.coarse_y_bins)
    assert rem == 0
    assert coarse_y_bin_size % config.fine_y_bins == 0

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
        *[
            Token(TokenType.POS_COARSE,(x,y)) 
            for x in range(config.coarse_x_bins)
            for y in range(config.coarse_y_bins)
        ],
        *[
            Token(TokenType.POS_FINE,(x,y))
            for x in range(config.fine_x_bins)
            for y in range(config.fine_y_bins)
        ],
    )