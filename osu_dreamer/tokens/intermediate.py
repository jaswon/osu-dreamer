
from dataclasses import dataclass
from typing import Iterable

from .timed import HitCircle, Timed, Break, SliderMult, BeatLen, Spinner, PerfectSlider, BezierSlider
from .parse_file import parse_map_file
from .parse_slider import parse_slider

@dataclass
class IntermediateBeatmap:
    hp_drain_rate: float
    circle_size: float
    overall_difficulty: float
    approach_rate: float
    base_slider_mult: float
    slider_tick_rate: float

    timed: list[tuple[int, Timed]]

    def __str__(self):
        return "\n".join([
            f"HP: {self.hp_drain_rate}",
            f"CS: {self.circle_size}",
            f"OD: {self.overall_difficulty}",
            f"AR: {self.approach_rate}",
            f"base slider mult: {self.base_slider_mult}",
            f"slider tick rate: {self.slider_tick_rate}",
            "",
            *( f"{t:08}: {v}" for t, v in self.timed ),
        ])

def to_intermediate(bm: Iterable[str]) -> IntermediateBeatmap:
    cfg = parse_map_file(bm)
    diff: dict[str, str] = cfg.get('Difficulty', {}) # type: ignore
    raw_events: list[str] = cfg.get('Events', []) # type: ignore
    raw_timing_points: list[str] = cfg.get('TimingPoints', []) # type: ignore
    raw_hit_objects: list[str] = cfg.get('HitObjects', []) # type: ignore

    uninherited: list[tuple[int, BeatLen]] = []
    inherited: list[tuple[int, SliderMult]] = []
    objects: list[tuple[int, Timed]] = []

    # parse breaks
    for l in raw_events:
        typ, t, *params = l.strip().split(",")
        if typ == '2' or typ == 'Break':
            u, = params
            d = int(float(u) - float(t))
            objects.append((int(t), Break(duration=d)))

    # parse timing points
    for l in raw_timing_points:
        vals = [ float(x) for x in l.strip().split(",") ]
        t, x, meter = vals[:3]

        if x < 0:
            # inherited timing point - controls slider multiplier 

            # .1 <= slider_mult <= 10
            slider_mult = min(10., max(.1, round(-100/x, 3)))

            # check if previous is redundant
            if len(inherited) > 0:
                last_t, last_sm = inherited[-1]
                if last_t == t or last_sm.mult == slider_mult:
                    inherited.pop()
                
            inherited.append((int(t), SliderMult(mult=slider_mult)))
        else:
            # uninherited timing point - controls beat length and meter, resets slider multiplier
            uninherited.append((int(t), BeatLen(ms=x, meter=int(meter))))
            inherited.append((int(t), SliderMult(mult=1)))

    # parse hit objects
    for l in raw_hit_objects:
        spl = l.strip().split(",")
        x, y, t, typ, hit_sound = [int(float(x)) for x in spl[:5]]

        hit_object_args = (
            (typ&(1<<2)) > 0,           # new_combo
            0 != hit_sound & (1 << 1),  # whistle
            0 != hit_sound & (1 << 2),  # finish
            0 != hit_sound & (1 << 3),  # clap
        )

        if typ & (1 << 0):  # hit circle
            objects.append((t, HitCircle(*hit_object_args, p=(x,y))))
            continue
        elif typ & (1 << 1):  # slider
            curve_points, slides, length = spl[5:8]
            objects.append((t, parse_slider(x,y,hit_object_args, curve_points, slides, length)))
            continue
        elif typ & (1 << 3):  # spinner
            d = int(float(spl[5]) - t)
            objects.append((t, Spinner(*hit_object_args, duration=d)))
            continue

    # `sorted` is stable- for the same time, order is maintained
    timed = sorted([ *uninherited, *inherited, *objects ], key=lambda t: t[0])

    # post-hoc computation of duration from slider length, beat length, and slider mult
    cur_beat_len = uninherited[0][1].ms
    cur_slider_mult = 1.
    for t, v in timed:
        if isinstance(v, BeatLen):
            cur_beat_len = v.ms
        elif isinstance(v, SliderMult):
            cur_slider_mult = v.mult
        elif isinstance(v, (PerfectSlider, BezierSlider)):
            slider_length = v.duration
            slide_duration = slider_length / (cur_slider_mult * 100) * cur_beat_len
            v.duration = round(v.slides * slide_duration)

    return IntermediateBeatmap(
        hp_drain_rate = float(diff.get('HPDrainRate', 5)),
        circle_size = float(diff.get('CircleSize', 5)),
        overall_difficulty = float(diff.get('OverallDifficulty', 5)),
        approach_rate = float(diff.get('ApproachRate', 5)),
        base_slider_mult = float(diff.get('SliderMultiplier', 1.4)),
        slider_tick_rate = float(diff.get('SliderTickRate', 1)),
        timed = timed,
    )


def to_beatmap(ib: IntermediateBeatmap, *args, **kwargs) -> str:
    ...