import bisect

import numpy as np
import scipy
import bezier

from osu_dreamer.osu.hit_objects import TimingPoint
from .smooth_hit import decode_hit, decode_hold
from .fit_bezier import fit_bezier
from .from_beatmap import AUX_DIM, HIT_DIM, SLIDER_DIM, CURSOR_DIM

BEAT_DIVISOR = 4

map_template = \
"""osu file format v14

[General]
AudioFilename: {audio_filename}
AudioLeadIn: 0
Mode: 0

[Metadata]
Title: {title}
TitleUnicode: {title}
Artist: {artist}
ArtistUnicode: {artist}
Creator: osu!dreamer
Version: {version}
Tags: osu_dreamer

[Difficulty]
HPDrainRate: 0
CircleSize: 3
OverallDifficulty: 0
ApproachRate: 9.5
SliderMultiplier: 1
SliderTickRate: 1

[TimingPoints]
{timing_points}

[HitObjects]
{hit_objects}
"""

def to_sorted_hits(hit_signal):
    """
    returns a list of tuples representing each hit object sorted by start: 
        `(start_idx, end_idx, object_type, new_combo)`

    `hit_signal`: [4,L] array of [0,1] where:
    - [0] represents hits
    - [1] represents slider holds
    - [2] represents spinner holds
    - [3] represents new combos
    """

    tap_sig, slider_sig, spinner_sig, new_combo_sig = hit_signal
    
    tap_idxs = decode_hit(tap_sig)
    slider_start_idxs, slider_end_idxs = decode_hold(slider_sig)
    spinner_start_idxs, spinner_end_idxs = decode_hold(spinner_sig)
    new_combo_idxs = decode_hit(new_combo_sig)

    sorted_hits = sorted([
        *[ (t, t, 0, False) for t in tap_idxs ],
        *[ (s, e, 1, False) for s,e in zip(sorted(slider_start_idxs), sorted(slider_end_idxs)) ],
        *[ (s, e, 2, False) for s,e in zip(sorted(spinner_start_idxs), sorted(spinner_end_idxs)) ],
    ])

    # associate hits with new combos
    for new_combo_idx in new_combo_idxs:
        idx = bisect.bisect_left(sorted_hits, (new_combo_idx,))
        if idx == len(sorted_hits):
            idx = idx-1
        elif idx > 0 and abs(new_combo_idx - sorted_hits[idx][0]) > abs(sorted_hits[idx-1][0] - new_combo_idx):
            idx = idx-1
        sorted_hits[idx] = ( *sorted_hits[idx][:3], True )

    return sorted_hits


def to_playfield_coordinates(cursor_signal):
    """
    transforms the cursor signal to osu!pixel coordinates
    """
     
    # rescale to fill the entire playfield
    # cs_valid_min = cursor_signal.min(axis=1, keepdims=True)
    # cs_valid_max = cursor_signal.max(axis=1, keepdims=True)
    # cursor_signal = (cursor_signal - cs_valid_min) / (cs_valid_max - cs_valid_min)
    
    # pad so that the cursor isn't too close to the edges of the screen
    # padding = 0.
    # cursor_signal = padding + cursor_signal * (1 - 2*padding)
    return cursor_signal * np.array([[512],[384]])
      

def to_slider_decoder(frame_times, cursor_signal, slider_signal):
    """
    returns a function that takes a start and end frame index and returns:
    - slider length
    - number of slides
    - slider control points
    """
    repeat_sig, seg_boundary_sig = slider_signal
    
    repeat_idxs: "L," = np.zeros_like(frame_times)
    repeat_idxs[decode_hit(repeat_sig)] = 1
    seg_boundary_idxs = decode_hit(seg_boundary_sig)
    
    def decoder(a, b):
        slides = int(sum(repeat_idxs[a:b+1]) + 1)
        ctrl_pts = []
        length = 0
        sb_idxs = [s for s in seg_boundary_idxs if a < s < b]
        for seg_start, seg_end in zip([a] + sb_idxs, sb_idxs + [b]):
            for b in fit_bezier(cursor_signal.T[seg_start:seg_end+1], max_err=100):
                b = np.array(b).round().astype(int)
                ctrl_pts.extend(b)
                length += bezier.Curve.from_nodes(b.T).length
        
        return length, slides, ctrl_pts

    return decoder
    

def to_beatmap(metadata, sig, frame_times, timing):
    """
    returns the beatmap as the string contents of the beatmap file
    """
    
    sig = sig[AUX_DIM:] # ignore auxiliary signals
    sig = (sig+1)/2 # [-1, 1] => [0, 1]
    
    hit_signal, sig = np.split(sig, (HIT_DIM,))
    slider_signal, sig = np.split(sig, (SLIDER_DIM,))
    cursor_signal, sig = np.split(sig, (CURSOR_DIM,))
    assert sig.shape[0] == 0
    
    # process hit signal
    sorted_hits = to_sorted_hits(hit_signal)
    
    # process cursor signal
    cursor_signal = to_playfield_coordinates(cursor_signal)
    
    # process slider signal
    slider_decoder = to_slider_decoder(frame_times, cursor_signal, slider_signal)

    # `timing` can be one of:
    # - List[TimingPoint] : timed according to timing points
    # - None : no prior knowledge of audio timing
    # - number : audio is constant BPM
    if isinstance(timing, list) and len(timing) > 0:
        beat_snap, timing_points = True, timing
    elif timing is None:
        # TODO: compute tempo from hit times

        # the following code only works when the whole song is a constant tempo

        # diff_dist = scipy.stats.gaussian_kde([ 
        #     np.log(frame_times[b[0]] - frame_times[a[0]])
        #     for a,b in zip(sorted_hits[:-1], sorted_hits[1:])
        # ])
        # x = np.linspace(0,20,1000)
        # timing_beat_len = np.exp(x[diff_dist(x).argmax()])
        
        beat_snap, timing_points = False, [TimingPoint(0, 1000, None, 4, None)]
    elif isinstance(timing, (int, float)):
        timing_beat_len = 60. * 1000. / float(timing)
        # compute timing offset
        offset_dist = scipy.stats.gaussian_kde([ frame_times[i] % timing_beat_len for i,_,_,_ in sorted_hits])
        offset = offset_dist.pdf(np.linspace(0, timing_beat_len, 1000)).argmax() / 1000. * timing_beat_len

        beat_snap, timing_points = True, [TimingPoint(offset, timing_beat_len, None, 4, None)]

    hos = [] # hit objects
    tps = [] # timing points

    # dur = length / (slider_mult * 100 * SV) * beat_length
    # dur = length / (slider_mult * 100) / SV * beat_length
    # SV  = length / dur / (slider_mult * 100) * beat_length
    # SV  = length / dur / (slider_mult * 100 / beat_length)
    # => base_slider_vel = slider_mult * 100 / beat_length
    beat_length = timing_points[0].beat_length
    base_slider_vel = 100 / beat_length
    beat_offset = timing_points[0].t

    def add_hit_circle(i,j,t,u, new_combo):
        x,y = cursor_signal[:, i].round().astype(int)
        hos.append(f"{x},{y},{t},{1 + new_combo},0,0:0:0:0:")

    def add_spinner(i,j,t,u, new_combo):
        if t == u:
            # start and end time are the same, add a hit circle instead
            return add_hit_circle(i,j,t,u, new_combo)
        hos.append(f"256,192,{t},{8 + new_combo},0,{u}")

    def add_slider(i,j,t,u, new_combo):
        if t == u:
            # start and end time are the same, add a hit circle instead
            return add_hit_circle(i,j,t,u, new_combo)

        length, slides, ctrl_pts = slider_decoder(i, j)

        if length == 0:
            # slider has zero length, add a hit circle instead
            return add_hit_circle(i,j,t,u, new_combo)
        
        SV = length * slides / (u-t) / base_slider_vel

        x1,y1 = ctrl_pts[0]
        curve_pts = "|".join(f"{x}:{y}" for x,y in ctrl_pts[1:])
        hos.append(f"{x1},{y1},{t},{2 + new_combo},0,B|{curve_pts},{slides},{length}")
        
        if len(tps) == 0:
            print('warning: inherited timing point added before any uninherited timing points')
        tps.append(f"{t},{-100/SV},4,0,0,50,0,0")

    last_up = None
    for i, j, t_type, new_combo in sorted_hits:
        t,u = frame_times[i], frame_times[j]
        if beat_snap:
            beat_f_len = beat_length / BEAT_DIVISOR
            t = round((t - beat_offset) / beat_f_len) * beat_f_len + beat_offset
            u = round((u - beat_offset) / beat_f_len) * beat_f_len + beat_offset
            
        t,u = int(t), int(u)
                
        # add timing points
        if len(timing_points) > 0 and t > timing_points[0].t:
            tp = timing_points.pop(0)
            tps.append(f"{tp.t},{tp.beat_length},{tp.meter},0,0,50,1,0")
            beat_length = tp.beat_length
            base_slider_vel = 100 / beat_length
            beat_offset = tp.t
            
        # ignore objects that start before the previous one ends
        if last_up is not None and t <= last_up + 1:
            continue

        [add_hit_circle, add_slider, add_spinner][t_type](i,j,t,u, 4 if new_combo else 0)
        last_up = u
            
    return map_template.format(**metadata, timing_points="\n".join(tps), hit_objects="\n".join(hos))