
from dataclasses import dataclass, asdict

from ..timed import *
from .file import OsuFile
from .slider.slider import parse_slider
from .template import map_template, Metadata

@dataclass
class BeatmapDifficulty:
    hp_drain_rate: float
    circle_size: float
    overall_difficulty: float
    approach_rate: float
    slider_tick_rate: float

@dataclass
class BeatmapEvents:
    timed: list[Timed]

    def __str__(self):
        return "\n".join([ f"{v.t:08}: {v}" for v in self.timed ])

def from_beatmap(cfg: OsuFile) -> tuple[BeatmapEvents, BeatmapDifficulty, Metadata]:
    uninherited: list[BeatLen] = []
    inherited: list[SliderVel] = []
    objects: list[Timed] = []

    # parse breaks
    for l in cfg.events:
        typ, t, *params = l.strip().split(",")
        if typ == '2' or typ == 'Break':
            u, = params
            d = int(float(u) - float(t))
            objects.append(Break(round(float(t)), round(float(u))))

    # parse timing points
    base_slider_vel = float(cfg.difficulty.get('SliderMultiplier', 1.4))
    for l in cfg.timing_points:
        vals = [ float(x) for x in l.strip().split(",") ]
        t, x = vals[:2]
        t = round(t)

        if x < 0:
            # inherited timing point - controls slider multiplier 

            # .1 <= slider_mult <= 10
            slider_vel = round(base_slider_vel * min(10., max(.1, -100/x)), 3)

            if len(inherited) > 0:
                last_sm = inherited[-1]
                if last_sm.t == t:
                    # override previous inherited point at same time
                    inherited.pop()
                if last_sm.vel == slider_vel:
                    # skip emitting "same" inherited point
                    continue
                
            inherited.append(SliderVel(t, vel=slider_vel))
        else:
            # uninherited timing point - controls beat length and meter, resets slider multiplier
            uninherited.append(BeatLen(t, ms=x))
            inherited.append(SliderVel(t, vel=base_slider_vel))

    # parse hit objects
    for l in cfg.hit_objects:
        spl = l.strip().split(",")
        x, y, t, typ, hit_sound = [round(float(x)) for x in spl[:5]]

        hit_object_args = (
            (typ&(1<<2)) > 0,
            0 != hit_sound & (1 << 1),
            0 != hit_sound & (1 << 2),
            0 != hit_sound & (1 << 3),
        )

        if typ & (1 << 0):  # hit circle
            objects.append(HitCircle(t, *hit_object_args, p=(x,y)))
            continue
        elif typ & (1 << 1):  # slider
            curve_spec, slides, length = spl[5:8]
            try:
                slider = parse_slider(t,x,y, hit_object_args, curve_spec, int(slides), float(length))
            except Exception as e:
                raise Exception(t, (x,y), curve_spec) from e
            objects.append(slider)
            continue
        elif typ & (1 << 3):  # spinner
            objects.append(Spinner(t, round(float(spl[5])), *hit_object_args))
            continue

    # `sorted` is stable- for the same time, order is maintained
    timed = sorted([ *uninherited, *inherited, *objects ], key=lambda obj: obj.t)

    # post-hoc computation of duration from slider length, beat length, and slider mult
    cur_beat_len = uninherited[0].ms
    cur_slider_vel = base_slider_vel
    for i, v in enumerate(timed):
        if isinstance(v, BeatLen):
            cur_beat_len = v.ms
        elif isinstance(v, SliderVel):
            cur_slider_vel = v.vel
        elif isinstance(v, Slider):
            # (negative of) visual length of sliders is temporarily stored in u 
            slide_duration = -v.u / (cur_slider_vel * 100) * cur_beat_len
            v.u = v.t + round(v.slides * slide_duration)

    # remove timing points
    for i, v in reversed(list(enumerate(timed))):
        if isinstance(v, (BeatLen, SliderVel)):
            timed.pop(i)

    return BeatmapEvents(timed), BeatmapDifficulty(
        hp_drain_rate = float(cfg.difficulty.get('HPDrainRate', 5)),
        circle_size = float(cfg.difficulty.get('CircleSize', 5)),
        overall_difficulty = float(cfg.difficulty.get('OverallDifficulty', 5)),
        approach_rate = float(cfg.difficulty.get('ApproachRate', 5)),
        slider_tick_rate = float(cfg.difficulty.get('SliderTickRate', 1)),
    ), Metadata(
        audio_filename = cfg.general.get('AudioFilename', 'audio.mp3'),
        title = cfg.metadata.get('Title', 'title'),
        artist = cfg.metadata.get('Artist', 'artist'),
        version = cfg.metadata.get('Version', 'version'),
    )


def to_beatmap(
    ib: BeatmapEvents,
    diff: BeatmapDifficulty,
    metadata: Metadata,
) -> str:

    # first loop over inherited timing points to compute base slider vel
    min_slider_vel = float('inf')
    max_slider_vel = float('-inf')
    for obj in ib.timed:
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
    for v in ib.timed:
        t = v.t
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
                x, y = v.head

                match v:
                    case PerfectSlider():
                        qx, qy = v.tail
                        if v.deviation == 0:
                            # straight line
                            curve = f"L|{qx}:{qy}"
                        else:
                            cx, cy = v.get_control_point()
                            curve = f"P|{cx}:{cy}|{qx}:{qy}"

                    case PolyLineSlider(vertices=vertices):
                        if len(vertices) == 1:
                            # single line
                            qx, qy = vertices[0]
                            curve = f"L|{qx}:{qy}"
                        else:
                            # poly line
                            pts = [ f"{x}:{y}" for x,y in vertices for _ in range(2) ]
                            curve = "|".join(["B"] + pts[:-1])

                    case BezierSlider():

                        if len(v.segments) == 1 and len(v.segments[0].ctrl) == 1:
                            # single line
                            qx, qy = v.segments[0].ctrl[0]
                            curve = f"L|{qx}:{qy}"
                        else:
                            pts = []
                            last_q = None
                            for seg in v.segments:
                                if last_q is not None:
                                    pts.append(last_q)
                                last_q = seg.ctrl[-1]
                                pts.extend(seg.ctrl)

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
        ar=diff.approach_rate,
        od=diff.overall_difficulty,
        cs=diff.circle_size,
        hp=diff.hp_drain_rate,
        tr=diff.slider_tick_rate,
        breaks="\n".join(breaks),
        timing_points="\n".join(timing_points), 
        hit_objects="\n".join(hit_objects),
    )