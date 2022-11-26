import numpy as np
import scipy

from .util import smooth_hit, HIT_SD

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

    `sig`: [4,L] array of [0,1] where:
    - [0] represents hits
    - [1] represents slider holds
    - [2] represents spinner holds
    - [3] represents new combos
    """

    f_b = max(2, HIT_SD*6)
    feat = smooth_hit(np.arange(-f_b, f_b+1), 0)

    tap_sig, slider_sig, spinner_sig, new_combo_sig = sig

    slider_sig_grad = np.gradient(slider_sig)
    slider_start_sig = np.maximum(0, slider_sig_grad)
    slider_end_sig = -np.minimum(0, slider_sig_grad)

    spinner_sig_grad = np.gradient(spinner_sig)
    spinner_start_sig = np.maximum(0, spinner_sig_grad)
    spinner_end_sig = -np.minimum(0, spinner_sig_grad)

    hit_frame_idxs = []
    for hit_sig, hit_offset, peak_h in zip(
        [tap_sig, slider_start_sig, slider_end_sig, spinner_start_sig, spinner_end_sig, new_combo_sig],
        [0,1,-1,1,-1,0],
        [.5, .25, .25, .25, .25, .5],
    ):
        corr = scipy.signal.correlate(hit_sig, feat, mode='same')
        hit_peaks = scipy.signal.find_peaks(corr, height=peak_h)[0] + hit_offset

        hit_frame_idxs.append(hit_peaks.astype(int).tolist())
    
    # sort hits
    sorted_hits = []
    (
        tap_idxs,
        slider_start_idxs,
        slider_end_idxs,
        spinner_start_idxs,
        spinner_end_idxs,
        new_combo_idxs,
    ) = hit_frame_idxs

    sorted_hits.extend([ (t, t, 0, False) for t in tap_idxs ])
    sorted_hits.extend([ (s, e, 1, False) for s,e in zip(sorted(slider_start_idxs), sorted(slider_end_idxs)) ])
    sorted_hits.extend([ (s, e, 2, False) for s,e in zip(sorted(spinner_start_idxs), sorted(spinner_end_idxs)) ])

    sorted_hits = sorted(sorted_hits)

    # associate hits with new combos
    for new_combo_idx in new_combo_idxs:
        idx = bisect.bisect_left(sorted_hits, (new_combo_idx,))
        if idx == len(sorted_hits):
            idx = idx-1
        elif idx+1 < len(sorted_hits) and abs(new_combo_idx - sorted_hits[idx][0]) > abs(sorted_hits[idx+1][0] - new_combo_idx):
            idx = idx+1
        sorted_hits[idx] = ( sorted_hits[idx][0], sorted_hits[idx][1], sorted_hits[idx][2], True )

    return sorted_hits


def to_map(metadata, sig, frame_times, bpm=None):
    """
    returns the beatmap as the string contents of the beatmap file
    """
    
    beat_length = 1000 if bpm is None else 60 * 1000 / bpm
    base_slider_vel = 100 / beat_length
    
    sig = (sig+1)/2 # [-1, 1] => [0, 1]
    hit_signal, cursor_signal = sig[:4], sig[4:]
    
    # process cursor signal
    padding = .06

    pressing = np.clip(hit_signal.max(axis=0, keepdims=True), 0, 1)
    cs_valid = cursor_signal * pressing
    
    cs_valid_min = cs_valid.min(axis=1, keepdims=True)
    cs_valid_max = cs_valid.max(axis=1, keepdims=True)
    cursor_signal = (cursor_signal - cs_valid_min) / (cs_valid_max - cs_valid_min)
    # cs_valid_center = cs_valid.mean(axis=1, keepdims=True)
    # cursor_signal = (cursor_signal - cs_valid_center + 1)/2
    
    cursor_signal *= np.array([[512],[384]]) * (1 - 2*padding)
    cursor_signal += np.array([[512],[384]]) * padding
    
    # process hit signal
    sorted_hits = to_sorted_hits(hit_signal)

    def get_ctrl_pts(a,b):
        x, y = cursor_signal.T[a:b], cursor_signal.T[a+1:b+1]
        l = np.linalg.norm(y-x, axis=1).sum()
        pts = np.array([ x,y ]).transpose(1,0,2).reshape((-1, 2))
        return pts, l

    hos = [] # hit objects
    tps = [f"0,{beat_length},4,0,0,50,1,0"] # timing points

    last_up = None
    for i, j, t_type, new_combo in sorted_hits:

        t,u = frame_times[i], frame_times[j]

        # ignore objects that start before the previous one ends
        if last_up is not None and i < last_up:
            continue

        new_combo = 4 if new_combo else 0

        if t_type == 0:
            # hit circle
            x,y = cursor_signal[:, i]
            hos.append(f"{x},{y},{t},{1 + new_combo},0")
            last_up = i
        elif t_type == 1:
            # slider
            ctrl_pts, length = get_ctrl_pts(i, j)

            SV = length / (u-t) / base_slider_vel

            x1,y1 = ctrl_pts[0]
            curve_pts = "|".join(f"{x}:{y}" for x,y in ctrl_pts[1:])
            hos.append(f"{x1},{y1},{t},{2 + new_combo},0,B|{curve_pts},1,{length}")
            tps.append(f"{t},{-100/SV},4,0,0,50,0,0")
            last_up = j
        elif t_type == 2:
            # spinner
            hos.append(f"256,192,{t},{8 + new_combo},0,{u}")
            last_up = j
            
    return map_template.format(**metadata, timing_points="\n".join(tps), hit_objects="\n".join(hos))