
from jaxtyping import Float

from dataclasses import dataclass, asdict

import numpy as np

from osu_dreamer.data.labels import NUM_LABELS
from osu_dreamer.osu.sliders import from_control_points

from .template import map_template
from . import events

@dataclass
class Metadata:
    audio_filename: str
    title: str
    artist: str
    version: str = "osu!dreamer model"

def decode(metadata: Metadata, labels: Float[np.ndarray, str(f"{NUM_LABELS}")], evs: list[events.Event]) -> str:

    tps = []
    hos = []
    breaks = []

    slider_ts = []
    slider_vels = []

    for event in evs:
        match event:
            case events.Break(t,u):
                breaks.append(f"2,{t},{u}")
            case events.Onset(t,new_combo,whistle,finish,clap):
                combo_bit = (1 << 2) if new_combo else 0
                hitsound = sum([
                    (1 << 1) if whistle else 0,
                    (1 << 2) if finish else 0,
                    (1 << 3) if clap else 0,
                ])
                match event:
                    case events.Circle(pos=pos):
                        x,y = pos
                        hos.append(f"{x:.0f},{y:.0f},{t},{(1 << 0) + combo_bit},{hitsound},0:0:0:0:")
                    case events.Spinner(u=u):
                        hos.append(f"256,192,{t},{(1 << 3) + combo_bit},{hitsound},{u}")
                    case events.Slider(u=u, slides=slides, control_points=control_points):
                        control_points = [ np.array(p) for p in control_points ]
                        length = from_control_points(t, -1, -1, new_combo, hitsound, slides, -1, control_points).length

                        slider_type = {
                            events.Line: "L",
                            events.Perfect: "P",
                            events.Bezier: "B",
                        }[type(event)]

                        x1,y1 = control_points[0]
                        curve_pts = "|".join(f"{x:.0f}:{y:.0f}" for x,y in control_points[1:])
                        hos.append(f"{x1:.0f},{y1:.0f},{t},{(1 << 1) + combo_bit},{hitsound},{slider_type}|{curve_pts},{slides},{length:.2f}")

                        slider_ts.append(t)
                        slider_vels.append(length / (u-t))
            case _:
                print('encountered unexpected event:', event)
    

    # dur = length / (slider_mult * 100 * SV) * beat_length
    # dur = length / (slider_mult * 100) / SV * beat_length
    # SV  = length / dur / (slider_mult * 100) * beat_length
    # SV  = length / dur / (slider_mult * 100 / beat_length)
    # => base_slider_vel = slider_mult * 100 / beat_length
    # => beat_length = slider_mult * 100 / base_slider_vel
    base_slider_vel = 1 if len(slider_vels) == 0 else (min(slider_vels) * max(slider_vels)) ** .5 
    beat_len = 100 / base_slider_vel # set `slider_mult` to 1 (.4 <= `slider_mult` <= 3.6)
    print(f'`beat_len` set to {beat_len}')

    # TODO: compute timing points
    tps.append(f"0,{beat_len},4,0,0,50,1,0")

    for t, vel in zip(slider_ts, slider_vels):
        SV = vel / base_slider_vel
        if SV > 10 or SV < .1:
            print('warning: SV > 10 or SV < .1 not supported, might result in bad sliders:', SV)
        
        tps.append(f"{t},{-100/SV},4,0,0,50,0,0")

    return map_template.format(
        **asdict(metadata), 
        ar=labels[1],
        od=labels[2],
        cs=labels[3],
        hp=labels[4],
        breaks="\n".join(breaks),
        timing_points="\n".join(tps), 
        hit_objects="\n".join(hos),
    )