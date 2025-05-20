
from typing import NamedTuple, Any

from itertools import product
from enum import Enum

class CustomReprEnum(Enum):
    def __repr__(self):
        return self.name

TokenType = CustomReprEnum('TokenType', [
    # special tokens
    "BOS",
    "EOS",
    "PAD",
    "DIFF",

    # object types
    "CIRCLE",
    "SLIDER",
    "SPINNER",
    "BREAK",
    "RELEASE",

    # slider tokens
    "SLIDES",
    "LINE",
    "PERFECT",
    "BEZIER",
    "KNOT",
    "BEZIER_END",

    "FLAGS",
    "LOCATION",
    "TIMESTAMP",
])

class Token(NamedTuple):
    typ: TokenType
    value: Any = None

    def __repr__(self):
        return f"{repr(self.typ)}({"" if self.value is None else repr(self.value)})"

def encode(token: Token) -> int:
    return _token2id[token]

def decode(token_id: int) -> Token:
    return _id2token[token_id]

_id2token: tuple[Token, ...] = (
    Token(TokenType.BOS),
    Token(TokenType.EOS),
    Token(TokenType.PAD),
    Token(TokenType.DIFF),

    Token(TokenType.CIRCLE),
    Token(TokenType.SLIDER),
    Token(TokenType.SPINNER),
    Token(TokenType.BREAK),
    Token(TokenType.RELEASE),

    *(
        Token(TokenType.SLIDES, i) 
        for i in range(1, 100)
    ),
    Token(TokenType.LINE),
    Token(TokenType.PERFECT),
    Token(TokenType.BEZIER),
    Token(TokenType.KNOT),
    Token(TokenType.BEZIER_END),

    *( # new combo, whistle, finish, clap
        Token(TokenType.FLAGS, flags)
        for flags in product([False, True], repeat=4)
    ),
    *( # hi-res on-screen coordinates
        Token(TokenType.LOCATION, (x,y))
        for x in range(0, 512+4, 4)
        for y in range(0, 384+4, 4)
    ),
    *( # low-res off-screen coordinates
        Token(TokenType.LOCATION, (x,y))
        for x in range(-256, 512+256+16, 16)
        for y in range(-256, 384+256+16, 16)
        if not ((0<=x<=512) and (0<=y<=384))
    ),
    *( # one timestamp token per context frame
        Token(TokenType.TIMESTAMP, i)
        for i in range(2048) # TODO: sync with seq_len
    )
)

_token2id = { t: i for i, t in enumerate(_id2token) }

VOCAB_SIZE = len(_token2id)

BOS = encode(Token(TokenType.BOS))
EOS = encode(Token(TokenType.EOS))
PAD = encode(Token(TokenType.PAD))
DIFF = encode(Token(TokenType.DIFF))
T0 = encode(Token(TokenType.TIMESTAMP, 0))