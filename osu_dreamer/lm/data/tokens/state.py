
import lark

from .tokens import Vocab

token_grammar = r"""
    start: "BOS" event*

    event: time ( break | hit_object )

    break: "BREAK" release

    hit_object: onset ( spinner | hit_circle | slider )
    onset: ["NEW_COMBO"] ["WHISTLE"] ["FINISH"] ["CLAP"]
    spinner: "SPINNER" release
    hit_circle: "HIT_CIRCLE" coordinate
    slider: "SLIDER" release "SLIDES" coordinate ( perfect | polyline | bezier )
    perfect: "PERFECT" coordinate "DEVIATION"
    polyline: "POLYLINE" coordinate+
    bezier: "BEZIER" ( linear | quadratic | cubic )+
    linear: "LINEAR" coordinate
    quadratic: "QUADRATIC" coordinate "DEVIATION" "MAGNITUDE"
    cubic: "CUBIC" coordinate "DEVIATION" "DEVIATION" "MAGNITUDE" "MAGNITUDE"

    release: time "RELEASE"
    time: "TIME"
    coordinate: "POS"
"""

class LogitProcessor:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.parser = lark.Lark(token_grammar, parser='lalr').parse_interactive()

    def advance(self, token_id: int) -> list[bool]:
        self.parser.feed_token(lark.Token(self.vocab.tokens[token_id].typ.name, ''))
        valid_types = self.parser.accepts()
        return [ tok.typ.name in valid_types for tok in self.vocab.tokens ]