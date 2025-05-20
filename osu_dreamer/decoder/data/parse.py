
from typing import Any, Callable, Generator

from . import events
from .tokens import Token, TokenType

class UnexpectedToken(Exception):
    def __init__(self, token):
        self.token = token

def decode_sustain() -> Generator[None, Token|float, int]:
    match (yield):
        case float(u): u = int(u)
        case _ as token: raise UnexpectedToken(token)
    match (yield):
        case Token(TokenType.RELEASE): pass
        case _ as token: raise UnexpectedToken(token)
    return u

def decode_break(t) -> Generator[None, Token|float, events.Break]:
    u = yield from decode_sustain()
    return events.Break(t,u)

def decode_onset() -> Generator[None, Token|float, tuple[bool,bool,bool,bool]]:
    match (yield):
        case Token(TokenType.FLAGS, (bool(new_combo), bool(whistle), bool(finish), bool(clap))):
            return (new_combo, whistle, finish, clap)
        case _ as token: raise UnexpectedToken(token)

def decode_position(token: Token|float|None = None) -> Generator[None, Token|float, tuple[int, int]]:
    match token if token is not None else (yield):
        case Token(TokenType.LOCATION, (int(x), int(y))):
            return (x,y)
        case _ as token: raise UnexpectedToken(token)

def decode_circle(t: int) -> Generator[None, Token|float, events.Circle]:
    onset = yield from decode_onset()
    pos = yield from decode_position()
    return events.Circle(t, *onset, pos)

def decode_spinner(t: int) -> Generator[None, Token|float, events.Spinner]:
    onset = yield from decode_onset()
    u = yield from decode_sustain()
    return events.Spinner(t, u, *onset)

def decode_slider(t: int) -> Generator[None, Token|float, events.Slider]:
    onset = yield from decode_onset()
    u = yield from decode_sustain()

    match (yield):
        case Token(TokenType.SLIDES, (int(slides))): pass
        case _ as token: raise UnexpectedToken(token)
    
    control_points = [ (yield from decode_position()) ]
    match (yield):
        case Token(TokenType.LINE):
            control_points.append((yield from decode_position()))
            return events.Line(t, u, *onset, slides, control_points)
        case Token(TokenType.PERFECT):
            control_points.append((yield from decode_position()))
            control_points.append((yield from decode_position()))
            return events.Perfect(t, u, *onset, slides, control_points)
        case Token(TokenType.BEZIER):
            control_points.append((yield from decode_position()))
            while True:
                match (yield):
                    case Token(TokenType.KNOT):
                        control_points.append(control_points[-1])
                    case Token(TokenType.BEZIER_END):
                        break
                    case _ as token:
                        control_points.append((yield from decode_position(token)))
            return events.Bezier(t, u, *onset, slides, control_points)
        case _ as token: raise UnexpectedToken(token)

def decode_event() -> Generator[None, Token|float, events.Event]:
    match (yield):
        case float(t): t = int(t)
        case _ as token: raise UnexpectedToken(token)

    match (yield):
        case Token(TokenType.BREAK):
            return (yield from decode_break(t))
        case Token(TokenType.CIRCLE):
            return (yield from decode_circle(t))
        case Token(TokenType.SPINNER):
            return (yield from decode_spinner(t))
        case Token(TokenType.SLIDER):
            return (yield from decode_slider(t))
        case _ as token: raise UnexpectedToken(token)

def coroutine(func):
    def initialize(*args, **kwargs):
        cr = func(*args, **kwargs)
        cr.send(None)
        return cr
    return initialize
 
@coroutine
def decode_beatmap(proc: Callable[[events.Event], Any], strict: bool) -> Generator[None, Token|float, None]:
    while True:
        try:
            event = (yield from decode_event())
        except UnexpectedToken as e:
            if strict:
                raise e
            print('unexpected token:', e.token)
            continue
        proc(event)


def parse_tokens(tokens: list[Token|float], strict: bool = True) -> list[events.Event]:
    events = []
    def proc_event(ev):
        events.append(ev)

    decoder = decode_beatmap(proc_event, strict)
    for tok in tokens:
        decoder.send(tok)

    return events