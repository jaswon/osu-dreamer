
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
    'POLYLINE',
    'BEZIER',
    'LINEAR',
    'QUADRATIC',
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
    'TIME',
    'POS_COARSE',
    'POS_FINE',
])

class Token(NamedTuple):
    typ: TokenType
    value: Any = None

    def __repr__(self):
        return f"{repr(self.typ)}({"" if self.value is None else repr(self.value)})"
    

@dataclass
class Vocab:

    time_bins: int = 2048

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
    MIN_MAGNITUDE: float = 1.
    MAX_MAGNITUDE: float = 1000.

    def __post_init__(self):

        assert self.DEVIATION_BINS % 2 == 0
        
        coarse_x_bin_size, rem = divmod(self.x_max - self.x_min, self.coarse_x_bins)
        assert rem == 0
        assert coarse_x_bin_size % self.fine_x_bins == 0

        coarse_y_bin_size, rem = divmod(self.y_max - self.y_min, self.coarse_y_bins)
        assert rem == 0
        assert coarse_y_bin_size % self.fine_y_bins == 0

        self.tokens: tuple[Token, ...] = (
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
            Token(TokenType.POLYLINE),
            Token(TokenType.BEZIER),
            Token(TokenType.LINEAR),
            Token(TokenType.CUBIC),
            Token(TokenType.QUADRATIC),
            *[ Token(TokenType.SLIDES,i) for i in range(self.SLIDES_BINS) ],
            *[
                Token(TokenType.DEVIATION,s)
                for s in range(self.DEVIATION_BINS)
            ],
            *[
                Token(TokenType.MAGNITUDE,m)
                for m in range(self.MAGNITUDE_BINS)
            ],

            # hit object flags
            Token(TokenType.NEW_COMBO),
            Token(TokenType.WHISTLE),
            Token(TokenType.FINISH),
            Token(TokenType.CLAP),

            # numerals
            *[
                Token(TokenType.POS_COARSE,(x,y))
                for x in range(self.coarse_x_bins)
                for y in range(self.coarse_y_bins)
            ],
            *[
                Token(TokenType.POS_FINE,(x,y))
                for x in range(self.fine_x_bins)
                for y in range(self.fine_y_bins)
            ],

            # NOTE: time tokens should be last to enable invariant
            # token_id > self.ids[Token(TokenType.TIME_BIN, 0)] <=> time token
            *[
                Token(TokenType.TIME, i)
                for i in range(self.time_bins)
            ],
        )

        self.ids = { token: i for i, token in enumerate(self.tokens) }

        # frequently used control tokens
        self.PAD = self.ids[Token(TokenType.PAD)]
        self.BOS = self.ids[Token(TokenType.BOS)]
        self.EOS = self.ids[Token(TokenType.EOS)]