
import re
from typing import Iterable, Union

def parse_map_file(bmlines: Iterable[str]) -> dict[str, Union[list[str], dict[str, str]]]:
    LIST_SECTIONS = { "Events", "TimingPoints", "HitObjects" }
    cfg = {}
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
            if section in LIST_SECTIONS:
                cfg[section] = []
            else:
                cfg[section] = {}
            continue

        if section is None:
            continue

        if section in LIST_SECTIONS:
            cfg[section].append(l.strip())
        else:
            # key-value check
            m = re.search(r"^(\w*)\s?:\s?(.*)$", l)
            if m is not None:
                cfg[section][m.group(1)] = m.group(2).strip()

    return cfg