
from dataclasses import dataclass, asdict
from typing import Iterable


from .timed import *
from .parse_file import parse_map_file
from .parse_slider import parse_slider
from .template import map_template, Metadata

@dataclass
class IntermediateBeatmap:
    hp_drain_rate: float
    circle_size: float
    overall_difficulty: float
    approach_rate: float
    slider_tick_rate: float

    timed: list[tuple[int, Timed]]

    def __str__(self):
        return "\n".join([
            f"HP: {self.hp_drain_rate}",
            f"CS: {self.circle_size}",
            f"OD: {self.overall_difficulty}",
            f"AR: {self.approach_rate}",
            f"slider tick rate: {self.slider_tick_rate}",
            "",
            *( f"{t:08}: {v}" for t, v in self.timed ),
        ])

def to_intermediate(bm: Iterable[str]) -> tuple[IntermediateBeatmap, Metadata]:
    cfg = parse_map_file(bm)
    general: dict[str, str] = cfg.get('General', {}) # type: ignore
    metadata: dict[str, str] = cfg.get('Metadata', {}) # type: ignore
    diff: dict[str, str] = cfg.get('Difficulty', {}) # type: ignore
    raw_events: list[str] = cfg.get('Events', []) # type: ignore
    raw_timing_points: list[str] = cfg.get('TimingPoints', []) # type: ignore
    raw_hit_objects: list[str] = cfg.get('HitObjects', []) # type: ignore

    uninherited: list[tuple[int, BeatLen]] = []
    inherited: list[tuple[int, SliderVel]] = []
    objects: list[tuple[int, Timed]] = []

    # parse breaks
    for l in raw_events:
        typ, t, *params = l.strip().split(",")
        if typ == '2' or typ == 'Break':
            u, = params
            d = int(float(u) - float(t))
            objects.append((int(t), Break(duration=d)))

    # parse timing points
    base_slider_vel = float(diff.get('SliderMultiplier', 1.4))
    for l in raw_timing_points:
        vals = [ float(x) for x in l.strip().split(",") ]
        t, x = vals[:2]

        if x < 0:
            # inherited timing point - controls slider multiplier 

            # .1 <= slider_mult <= 10
            slider_vel = round(base_slider_vel * min(10., max(.1, -100/x)), 3)

            if len(inherited) > 0:
                last_t, last_sm = inherited[-1]
                if last_t == t:
                    # override previous inherited point at same time
                    inherited.pop()
                if last_sm.vel == slider_vel:
                    # skip emitting "same" inherited point
                    continue
                
            inherited.append((int(t), SliderVel(vel=slider_vel)))
        else:
            # uninherited timing point - controls beat length and meter, resets slider multiplier
            uninherited.append((int(t), BeatLen(ms=x)))
            inherited.append((int(t), SliderVel(vel=base_slider_vel)))

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
            curve_spec, slides, length = spl[5:8]
            objects.append((t, parse_slider(x,y,hit_object_args, curve_spec, int(slides), float(length))))
            continue
        elif typ & (1 << 3):  # spinner
            d = int(float(spl[5]) - t)
            objects.append((t, Spinner(*hit_object_args, duration=d)))
            continue

    # `sorted` is stable- for the same time, order is maintained
    timed = sorted([ *uninherited, *inherited, *objects ], key=lambda t: t[0])

    # post-hoc computation of duration from slider length, beat length, and slider mult
    cur_beat_len = uninherited[0][1].ms
    cur_slider_vel = base_slider_vel
    for i, (t, v) in enumerate(timed):
        if isinstance(v, BeatLen):
            cur_beat_len = v.ms
        elif isinstance(v, SliderVel):
            cur_slider_vel = v.vel
        elif isinstance(v, (PerfectSlider, BezierSlider)):
            slide_duration = v.length() / (cur_slider_vel * 100) * cur_beat_len
            v.duration = round(v.slides * slide_duration)

    # remove inherited timing points
    for i, (_, v) in reversed(list(enumerate(timed))):
        if isinstance(v, SliderVel):
            timed.pop(i)

    return IntermediateBeatmap(
        hp_drain_rate = float(diff.get('HPDrainRate', 5)),
        circle_size = float(diff.get('CircleSize', 5)),
        overall_difficulty = float(diff.get('OverallDifficulty', 5)),
        approach_rate = float(diff.get('ApproachRate', 5)),
        slider_tick_rate = float(diff.get('SliderTickRate', 1)),
        timed = timed,
    ), Metadata(
        audio_filename = general.get('AudioFilename', 'audio.mp3'),
        title = metadata.get('Title', 'title'),
        artist = metadata.get('Artist', 'artist'),
        version = metadata.get('Version', 'version'),
    )


def to_beatmap(
    ib: IntermediateBeatmap,
    metadata: Metadata,
) -> str:

    # first loop over inherited timing points to compute base slider vel
    min_slider_vel = float('inf')
    max_slider_vel = float('-inf')
    for _, obj in ib.timed:
        if isinstance(obj, Slider):
            vel = obj.vel()
            min_slider_vel = min(min_slider_vel, vel)
            max_slider_vel = max(max_slider_vel, vel)
    base_slider_vel = (min_slider_vel * max_slider_vel)**.5

    breaks = []
    hit_objects = []

    slider_ts = []
    slider_vels = []

    # running timing context for reconstructing slider length
    cur_beat_len: float = 600.
    cur_slider_vel: float = base_slider_vel

    for t, v in ib.timed:
        if isinstance(v, Break):
            end_time = t + v.duration
            breaks.append(f"2,{t},{end_time}")
        elif isinstance(v, HitObject):

            if isinstance(v, HitCircle):
                typ = 1 << 0
                x, y = v.p
                params = []
            elif isinstance(v, Spinner):
                typ = 1 << 3
                x,y = 256, 192
                end_time = t + v.duration
                params = [end_time]
            elif isinstance(v, Slider):
                slider_ts.append(t)
                slider_vels.append(v.vel())

                typ = 1 << 1

                if isinstance(v, PerfectSlider):
                    x, y = v.head
                    qx, qy = v.tail
                    if v.deviation == 0:
                        # straight line
                        curve = f"L|{qx}:{qy}"
                    else:
                        cx, cy = v.get_control_point()
                        curve = f"P|{cx}:{cy}|{qx}:{qy}"
                elif isinstance(v, BezierSlider):
                    # first point is encoded separately as x0,y0
                    x, y = v.head

                    # check if single line
                    if len(v.segments) == 1 and isinstance(v.segments[0], LineSegment):
                        qx, qy = v.segments[0].q
                        curve = f"L|{qx}:{qy}"
                    else:
                        pts = []
                        last_q = None
                        for seg in v.segments:
                            if last_q is not None:
                                pts.append(last_q)
                            last_q = seg.q

                            if isinstance(seg, LineSegment):
                                pts.append(seg.q)
                            elif isinstance(seg, CubicSegment):
                                pts.append(seg.pc)
                                pts.append(seg.qc)
                                pts.append(seg.q)

                        rest = "|".join(f"{px}:{py}" for px, py in pts)
                        curve = f"B|{rest}"

                params = [curve, v.slides, v.length()]

            hs = 0
            if v.whistle:
                hs |= 2
            if v.finish:
                hs |= 4
            if v.clap:
                hs |= 8
            if v.new_combo:
                typ |= (1 << 2)
            hit_objects.append(",".join(map(str,[x,y,t,typ,hs,*params])))

    base_slider_vel = 1 if len(slider_vels) == 0 else (min(slider_vels) * max(slider_vels)) ** .5 
    beat_len = 100 / base_slider_vel # set `slider_mult` to 1 (.4 <= `slider_mult` <= 3.6)

    timing_points = [f"0,{beat_len},4,0,0,50,1,0"]
    for t, vel in zip(slider_ts, slider_vels):
        SV = vel / base_slider_vel
        if SV > 10 or SV < .1:
            print('warning: SV > 10 or SV < .1 not supported, might result in bad sliders:', SV)
        
        timing_points.append(f"{t},{-100/SV},4,0,0,50,0,0")

    return map_template.format(
        **asdict(metadata), 
        ar=ib.approach_rate,
        od=ib.overall_difficulty,
        cs=ib.circle_size,
        hp=ib.hp_drain_rate,
        tr=ib.slider_tick_rate,
        breaks="\n".join(breaks),
        timing_points="\n".join(timing_points), 
        hit_objects="\n".join(hit_objects),
    )