
from typing import NamedTuple, Any

from itertools import product
from enum import Enum

EventType = Enum('EventType', [
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

    "ONSET",
    "LOCATION",
])

class Event(NamedTuple):
    typ: EventType
    value: Any = None

def encode(event: Event) -> int:
    return _event2token[event]

def decode(token: int) -> Event:
    return _token2event[token]

def vocab_size() -> int:
    return len(_event2token)

_token2event: tuple[Event, ...] = (
    Event(EventType.BOS),
    Event(EventType.EOS),
    Event(EventType.PAD),
    Event(EventType.DIFF),

    Event(EventType.CIRCLE),
    Event(EventType.SLIDER),
    Event(EventType.SPINNER),
    Event(EventType.BREAK),
    Event(EventType.RELEASE),

    *( Event(EventType.SLIDES, i) for i in range(1, 100) ),
    Event(EventType.LINE),
    Event(EventType.PERFECT),
    Event(EventType.BEZIER),
    Event(EventType.KNOT),
    Event(EventType.BEZIER_END),

    *( # new combo, whistle, finish, clap
        Event(EventType.ONSET, flags)
        for flags in product([False, True], repeat=4)
    ),

    *( # hi-res on-screen coordinates
        Event(EventType.LOCATION, (x,y))
        for x in range(0, 512+4, 4)
        for y in range(0, 384+4, 4)
    ),
    *( # low-res off-screen coordinates
        Event(EventType.LOCATION, (x,y))
        for x in range(-256, 512+256+16, 16)
        for y in range(-256, 384+256+16, 16)
        if not ((0<=x<=512) and (0<=y<=384))
    ),
)

_event2token = { t: i for i, t in enumerate(_token2event) }

BOS = encode(Event(EventType.BOS))
EOS = encode(Event(EventType.EOS))
PAD = encode(Event(EventType.PAD))
DIFF = encode(Event(EventType.DIFF))
