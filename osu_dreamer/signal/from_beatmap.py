import numpy as np

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.osu.hit_objects import Circle, Slider, Spinner, TimingPoint
from osu_dreamer.osu.sliders import Line, Perfect, Bezier

from .smooth_hit import smooth_hit

TIMING_DIM = 1
    
def timing_signal(beatmap_or_timing_points, frame_times: "L,") -> "1,L":
    if isinstance(beatmap_or_timing_points, Beatmap):
        utps = beatmap_or_timing_points.uninherited_timing_points
    elif (
        isinstance(beatmap_or_timing_points, list) and
        len(beatmap_or_timing_points) > 0 and
        isinstance(beatmap_or_timing_points[0], TimingPoint)
    ):
        utps = beatmap_or_timing_points
    else:
        raise ValueError("first argument must be a Beatmap or a list of TimingPoint")
        
    # timing point boundaries
    tpt = np.array([-np.inf] + [ utp.t for utp in utps[1:] ] + [np.inf])[:, None]
    
    # active_tp[0, i] = index into `utps` of the timing point active at frame `i`
    active_tp = (frame_times >= tpt[:-1]) & (frame_times < tpt[1:])
    active_tp: "[0,TP),L" = np.argwhere(active_tp.T)[None, :, 1]
    
    choices: "TP,L" = np.array([ 
        (frame_times - utp.t) / utp.beat_length * 2 * np.pi * 2
        for utp in utps
    ])

    x: "1,L" = np.take_along_axis(choices, active_tp, axis=0)
    return np.maximum(0, np.cos(x))


def hit_signal(beatmap, frame_times: "L,") -> "4,L":
    """
    returns an array encoding the hits occurring at the times represented by `frames`
    - [0] represents hits
    - [1] represents slider holds
    - [2] represents spinner holds
    - [3] represents new combos

    - `frame_times`: array of times at each frame in ms
    """

    sig = np.zeros((4, len(frame_times)))
    for ho in beatmap.hit_objects:
        if isinstance(ho, Circle):
            sig[0] += smooth_hit(frame_times, ho.t)
        elif isinstance(ho, Slider):
            sig[1] += smooth_hit(frame_times, (ho.t, ho.end_time()))
        else: # Spinner
            sig[2] += smooth_hit(frame_times, (ho.t, ho.end_time()))

        if ho.new_combo:
            sig[3] += smooth_hit(frame_times, ho.t)

    return sig


def slider_signal(beatmap, frame_times: "L,") -> "4,L":
    """
    returns an array encoding slider metadata
    - [0] represents a slider repeat
    - [1] represents a slider segment boundary
    - [2] represents slider segment type (1 for bezier, 0 for line) (approx. Perfect with Bezier)
    """
    
    range_sl = lambda a,b: (frame_times >= a) & (frame_times < b)
    
    sig = np.zeros((3, len(frame_times)))
    for ho in beatmap.hit_objects:
        if not isinstance(ho, Slider):
            continue
            
        single_slide = ho.slide_duration / ho.slides
            
        if ho.slides > 1:
            for i in range(ho.slides-1):
                sig[0] += smooth_hit(frame_times, ho.t + single_slide * (i+1))
                
        if isinstance(ho, Bezier):
            cur_t, last_t = ho.t, None
            for c in ho.path_segments[:-1]:
                last_t = cur_t
                cur_t += c.length / ho.length * ho.slide_duration
                sig[1] += smooth_hit(frame_times, cur_t)
                sig[2, range_sl(last_t, cur_t)] = 0 if c.nodes.shape[-1] == 2 else 1
            sig[2, range_sl(cur_t, ho.end_time())] = 0 if ho.path_segments[-1].nodes.shape[-1] == 2 else 1
        else:
            sig[2, range_sl(ho.t,ho.end_time())] = 0 if isinstance(ho, Line) else 1
            
    return sig
            
        

def hit_object_pairs(beatmap, frame_times: "L,"):
    """
    for each `t` in `frame_times` yields the pair of adjacent hit objects that surround `t`
    """
    pairs = zip([None] + beatmap.hit_objects, beatmap.hit_objects + [None])
    a,b = next(pairs)
    for t in frame_times:
        while b is not None and b.t < t:
            a,b = next(pairs)
        yield a,b

def cursor_signal(beatmap, frame_times: "L,") -> "2,L":
    """
    return [2,L] where [{0,1},i] is the {x,y} position at the times represented by `frames`

    - `frame_times`: array of times at each frame in ms
    """
    pos = []
    for t, (a,b) in zip(frame_times, hit_object_pairs(beatmap, frame_times)):
        if a is None:
            # before first hit object
            pos.append(b.start_pos())
        elif t < a.end_time():
            # hitting current hit object
            if isinstance(a, (Circle, Spinner)):
                pos.append(a.start_pos())
            elif isinstance(a, Slider):
                # NOTE: for the duration of the slider, the cursor signal generated
                # will only traverse a single slide of the slider, regardless of 
                # how many slides the slider actually has
                
                pos.append(a.lerp((t - a.t) / a.slide_duration))
        elif b is None:
            # after last hit object
            pos.append(a.end_pos())
        else:
            # moving to next hit object
            f = (t - a.end_time()) / (b.t - a.end_time())
            pos.append((1 - f) * a.end_pos() + f * b.start_pos())
            
    return np.array(pos).T
        
MAP_SIGNAL_DIM = 9

def from_beatmap(beatmap, frame_times: "L,") -> "9,L":
    """
    returns a [6,L] scaled to [-1,1]

    - `frame_times`: array of times at each frame in ms
    """
    hits: "4,L" = hit_signal(beatmap, frame_times)
    slider: "3,L" = slider_signal(beatmap, frame_times)
    cursor: "2,L" = cursor_signal(beatmap, frame_times) / np.array([[512],[384]])

    return np.concatenate([hits, slider, cursor], axis=0) * 2 - 1
    