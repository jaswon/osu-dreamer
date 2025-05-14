
from typing import Optional, Union

import re
from pathlib import Path

import bisect
import numpy as np

import rosu_pp_py as rosu

from .hit_objects import Timed, TimingPoint, Circle, Spinner, Slider, Break
from .sliders import from_control_points


class Beatmap:

    @staticmethod
    def parse_map_file(bmlines):
        LIST_SECTIONS = ["Events", "TimingPoints", "HitObjects"]
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

    def __repr__(self):
        return f"{self.title} [{self.version}]"

    def __init__(self, filename, meta_only=False):

        self.sr = rosu.Performance().calculate(rosu.Beatmap(path=str(filename))).difficulty.stars
        
        self.filename = Path(filename)

        with open(filename, encoding='utf-8') as f:
            cfg = self.parse_map_file(f)

        # account for case-insensitivity
        lc_files = { f.name.lower(): f.name for f in self.filename.parent.iterdir() }
        self.audio_filename = self.filename.parent / lc_files[cfg["General"]["AudioFilename"].lower()]

        self.mode = int(cfg["General"]["Mode"])

        self.title = cfg["Metadata"]["Title"]
        self.artist = cfg["Metadata"]["Artist"]
        self.creator = cfg["Metadata"]["Creator"]
        self.version = cfg["Metadata"]["Version"]

        self.hp = float(cfg["Difficulty"]["HPDrainRate"])
        self.cs = float(cfg["Difficulty"]["CircleSize"])
        self.od = float(cfg["Difficulty"]["OverallDifficulty"])

        try:
            self.ar = float(cfg["Difficulty"]["ApproachRate"])
        except KeyError:
            self.ar = 7.

        # base slider velocity in hundreds of osu!pixels per beat
        self.slider_mult = float(cfg["Difficulty"]["SliderMultiplier"])

        # slider ticks per beat
        self.slider_tick = float(cfg["Difficulty"]["SliderTickRate"])

        try:
            self.beat_divisor = int(cfg["Editor"]["BeatDivisor"])
        except KeyError:
            self.beat_divisor = 4

        self.unparsed_hitobjects = cfg["HitObjects"]
        self.unparsed_timingpoints = cfg["TimingPoints"]
        self.unparsed_events = cfg.get("Events", [])
        if not meta_only:
            self.parse_map_data()

    def parse_map_data(self):
        self.parse_timing_points(self.unparsed_timingpoints)
        del self.unparsed_timingpoints

        self.parse_hit_objects(self.unparsed_hitobjects)
        del self.unparsed_hitobjects

        self.parse_breaks(self.unparsed_events)
        del self.unparsed_events

    def parse_breaks(self, lines):
        self.breaks: list[Break] = []
        for l in lines:
            typ, t, *params = l.strip().split(",")
            if typ == '2' or typ == 'Break':
                u, = params
                self.breaks.append(Break(int(float(t)), int(float(u))))

    def parse_timing_points(self, lines):
        self.timing_points: list[TimingPoint] = []
        
        cur_beat_length = None
        cur_slider_mult = 1.
        cur_meter = None
        
        for l in lines:
            vals = [ float(x) for x in l.strip().split(",") ]
            t, x, meter = vals[:3]
            
            if x < 0:
                # inherited timing point - controls slider multiplier
                if len(self.timing_points) == 0:
                    continue
                    
                if self.timing_points[-1].t == t:
                    self.timing_points.pop()
                    
                # .1 <= slider_mult <= 10.
                cur_slider_mult = min(10., max(.1, round(-100 / float(x), 3)))
            else:
                # uninherited timing point - controls beat length and meter, resets slider multiplier
                cur_beat_length = x
                cur_slider_mult = 1.
                cur_meter = meter

            if cur_beat_length is None or cur_meter is None:
                raise ValueError("inherited timing point appears before any uninherited timing points")
                
            tp = TimingPoint(int(t), cur_beat_length, cur_slider_mult, int(cur_meter))

            # skip adding timing points if they duplicate the last active one
            if len(self.timing_points) == 0 or tp != self.timing_points[-1]:
                self.timing_points.append(tp)
            
        if len(self.timing_points) == 0:
            raise ValueError("no timing points")
        
    def uninherited_timing_points(self) -> list[TimingPoint]:
        """returns a list of timing points that have distinct meter and beat length"""
        utp = []
        for tp in self.timing_points:
            x = TimingPoint(tp.t, tp.beat_length, -1, tp.meter)
            if len(utp) == 0 or utp[-1] != x:
                utp.append(x)
        return utp
    
    def timing_point_at(self, t) -> Optional[TimingPoint]:
        """timing point active for time `t` or None if before all timing points"""
        tp_idx = bisect.bisect(self.timing_points, Timed(t)) - 1
        return None if tp_idx < 0 else self.timing_points[tp_idx]

    def parse_hit_objects(self, lines):
        self.hit_objects: list[Union[Circle, Slider, Spinner]] = []
        for l in lines:
            spl = l.strip().split(",")
            x, y, t, typ, hit_sound = [int(float(x)) for x in spl[:5]]
            new_combo = (typ&(1<<2)) > 0 
            if typ & (1 << 0):  # hit circle
                ho = Circle(t, new_combo, hit_sound, x, y)
            elif typ & (1 << 1):  # slider
                curve, slides, length = spl[5:8]
                _, *control_points = curve.split("|")
                control_points = [np.array([x,y], dtype=float)] + [
                    np.array(list(map(float, p.split(":"))), dtype=float) for p in control_points
                ]
                
                tp = self.timing_point_at(t)
                beat_length = tp.beat_length if tp is not None else self.timing_points[0].beat_length
                slider_mult = tp.slider_mult if tp is not None else 1.
                
                ho = from_control_points(
                    t, 
                    beat_length, 
                    self.slider_mult * slider_mult,
                    new_combo,
                    hit_sound,
                    int(slides),
                    float(length),
                    control_points,
                )
            elif typ & (1 << 3):  # spinner
                ho = Spinner(t, new_combo, hit_sound, int(float(spl[5])))
            else:
                raise ValueError(f"invalid hit object type: {typ}")
                
            if len(self.hit_objects) and ho.t < self.hit_objects[-1].end_time():
                raise ValueError(f"hit object starts before previous hit object ends: {t}")
                
            self.hit_objects.append(ho)
            
        if len(self.hit_objects) == 0:
            raise ValueError("no hit objects")
