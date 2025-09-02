
from typing import Iterable

from dataclasses import dataclass, field
import re

@dataclass
class OsuFile:

    # key-value sections
    general: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    difficulty: dict[str, str] = field(default_factory=dict)

    # list sections
    events: list[str] = field(default_factory=list)
    timing_points: list[str] = field(default_factory=list)
    hit_objects: list[str] = field(default_factory=list)

def parse_map_file(bmlines: Iterable[str]) -> OsuFile:
    cfg = OsuFile()
    KV_SECTIONS = {
        "General": cfg.general,
        "Metadata": cfg.metadata,
        "Difficulty": cfg.difficulty,
    }

    LIST_SECTIONS = {
        "Events": cfg.events, 
        "TimingPoints": cfg.timing_points, 
        "HitObjects": cfg.hit_objects,
    }
    section = None
    for l in bmlines:
        # comments
        if l.startswith("//"):
            continue

        # section end check
        if l.strip() == "":
            section = None
            continue

        # header check
        m = re.search(r"^\[(.*)\]$", l)
        if m is not None:
            section = m.group(1)
            continue

        if section is None:
            continue

        if section in LIST_SECTIONS:
            LIST_SECTIONS[section].append(l.strip())
        elif section in KV_SECTIONS:
            # key-value check
            m = re.search(r"^(\w*)\s?:\s?(.*)$", l)
            if m is not None:
                KV_SECTIONS[section][m.group(1)] = m.group(2).strip()

    return cfg