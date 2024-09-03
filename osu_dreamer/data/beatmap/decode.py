
from dataclasses import dataclass, asdict

from jaxtyping import Float

import numpy as np
from numpy import ndarray

from osu_dreamer.data.prepare_map import NUM_LABELS

from .fit_bezier import segment_length, Point, fit_bezier
from .encode import FrameTimes, EncodedBeatmap, BeatmapEncoding
from .hit import decode_extents, decode_onsets

@dataclass
class Metadata:
    audio_filename: str
    title: str
    artist: str
    version: str = "osu!dreamer model"

map_template = \
f"""osu file format v14

[General]
AudioFilename: {{audio_filename}}
AudioLeadIn: 0
Mode: 0

[Metadata]
Title: {{title}}
TitleUnicode: {{title}}
Artist: {{artist}}
ArtistUnicode: {{artist}}
Creator: osu!dreamer
Version: {{version}}
Tags: osu_dreamer

[Difficulty]
HPDrainRate: {{hp}}
CircleSize: {{cs}}
OverallDifficulty: {{od}}
ApproachRate: {{ar}}
SliderMultiplier: 1
SliderTickRate: 1

[TimingPoints]
{{timing_points}}

[HitObjects]
{{hit_objects}}
"""      

def slider_decoder(
    cursor_signal: Float[ndarray, "2 L"], 
    start_idx: int, 
    end_idx: int, 
    num_repeats: int,
) -> tuple[float, list[Point]]:
    """
    returns the slider's length and control points defined by
    the cursor signal, start+end indices, and number of repeats
    """

    first_slide_idx = round(start_idx + (end_idx-start_idx) / num_repeats)

    ctrl_pts: list[Point] = []
    length = 0.
    # TODO: try fit circular arc before bezier
    path = fit_bezier(cursor_signal.T[start_idx:first_slide_idx+1], max_err=50.)
    for i, seg in enumerate(path):
        seg = seg.round()
        seg_len = segment_length(seg)
        if len(path) > 1 and i == 0 and seg_len < 20:
            # skip short starting segments
            continue
        ctrl_pts.extend(seg)
        length += seg_len
    
    return length, ctrl_pts

ONSET_TOL = 2
DEFAULT_BEAT_LEN = 60000/100 # 100 bpm

def decode_beatmap(metadata: Metadata, labels: Float[np.ndarray, str(f"{NUM_LABELS}")], enc: EncodedBeatmap, frame_times: FrameTimes) -> str:

    cursor_signal = enc[[BeatmapEncoding.X, BeatmapEncoding.Y]]
    cursor_signal = (cursor_signal+1) * np.array([[256],[192]])

    onset_locs = decode_onsets(enc[BeatmapEncoding.ONSET])
    onset_loc2idx = np.full_like(frame_times, -1, dtype=int)
    for i, onset_idx in enumerate(onset_locs):
        onset_loc2idx[onset_idx-ONSET_TOL:onset_idx+ONSET_TOL+1] = i

    new_combos = [False] * len(onset_locs)
    for combo_start in decode_extents(enc[BeatmapEncoding.COMBO])[0]:
        onset_idx = onset_loc2idx[combo_start]
        if onset_idx == -1:
            continue
        new_combos[onset_idx] = True

    sustain_ends = [-1] * len(onset_locs)
    for sustain_start, sustain_end in zip(*decode_extents(enc[BeatmapEncoding.SUSTAIN])):
        onset_idx = onset_loc2idx[sustain_start]
        if onset_idx == -1:
            continue
        sustain_ends[onset_idx] = sustain_end

    slider_ends = [-1] * len(onset_locs)
    for slider_start, slider_end in zip(*decode_extents(enc[BeatmapEncoding.SLIDER])):
        onset_idx = onset_loc2idx[slider_start]
        if onset_idx == -1:
            continue
        slider_ends[onset_idx] = slider_end
    
    tps = []
    hos = []

    slider_ts = []
    slider_vels = []

    for onset_loc, new_combo, sustain_end, slider_end in zip(onset_locs, new_combos, sustain_ends, slider_ends):
        t = frame_times[onset_loc]
        combo_bit = 2**2 if new_combo else 0

        def add_hit_circle():
            x,y = cursor_signal[:, onset_loc].round().astype(int)
            hos.append(f"{x},{y},{t},{2**0 + combo_bit},0,0:0:0:0:")

        if sustain_end == -1:
            # no sustain
            add_hit_circle()
            continue

        if sustain_end - onset_loc < 4:
            # sustain too short
            add_hit_circle()
            continue

        u = frame_times[sustain_end]
        if slider_end == -1:
            # spinner
            hos.append(f"256,192,{t},{2**3 + combo_bit},0,{u}")
            continue

        if slider_end - onset_loc < 4:
            # slider too short
            add_hit_circle()
            continue

        # slider
        num_slides = max(1, round((sustain_end - onset_loc) / (slider_end - onset_loc)))
        length, ctrl_pts = slider_decoder(cursor_signal, onset_loc, sustain_end, num_slides)

        if length == 0:
            # zero length
            add_hit_circle()
            continue

        x1,y1 = ctrl_pts[0]
        curve_pts = "|".join(f"{x}:{y}" for x,y in ctrl_pts[1:])
        hos.append(f"{x1},{y1},{t},{2**1 + combo_bit},0,B|{curve_pts},{num_slides},{length}")

        slider_ts.append(t)
        slider_vels.append(length * num_slides / (u-t))

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
        ar=labels[0],
        od=labels[1],
        cs=labels[2],
        hp=labels[3],
        timing_points="\n".join(tps), 
        hit_objects="\n".join(hos),
    )