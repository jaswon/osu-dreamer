
from typing import NamedTuple, Any

from itertools import product
from enum import Enum

class CustomReprEnum(Enum):
    def __repr__(self):
        return self.name

TokenType = CustomReprEnum('TokenType', [
    # special tokens
    "PAD",
    "BOS",
    "EOS",
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
    Token(TokenType.PAD),
    Token(TokenType.BOS),
    Token(TokenType.EOS),
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
)

_token2id = { t: i for i, t in enumerate(_id2token) }

VOCAB_SIZE = len(_token2id)

BOS = encode(Token(TokenType.BOS))
EOS = encode(Token(TokenType.EOS))
PAD = encode(Token(TokenType.PAD))
DIFF = encode(Token(TokenType.DIFF))