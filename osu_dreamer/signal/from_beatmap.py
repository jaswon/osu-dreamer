import numpy as np

from osu_dreamer.osu.hit_objects import Circle, Slider, Spinner
from osu_dreamer.osu.sliders import Line, Perfect, Bezier

from .smooth_hit import smooth_hit


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
        elif isinstance(ho, Spinner):
            sig[2] += smooth_hit(frame_times, (ho.t, ho.end_time()))

        if ho.new_combo:
            sig[3] += smooth_hit(frame_times, ho.t)

    return sig


def slider_signal(beatmap, frame_times: "L,") -> "4,L":
    """
    returns an array encoding slider metadata
    - [0] represents a slider repeat
    - [1] represents a bezier slider segment boundary
    - [2] represents bezier slider segment type (1 for bezier, 0 for line) (approx. Perfect with Bezier)
    """
    # `frame_times` mask for times in range
    range_sl = lambda a,b: (frame_times >= a) & (frame_times < b)
    
    sig = np.zeros((3, len(frame_times)))

    for ho in beatmap.hit_objects:
        if not isinstance(ho, Slider):
            continue
        
        # only render intermediate slides (ie. exclude start and end)
        if ho.slides > 1:
            single_slide = ho.slide_duration / ho.slides
            for i in range(1, ho.slides):
                sig[0] += smooth_hit(frame_times, ho.t + single_slide * i)
                
        if isinstance(ho, Bezier):
            seg_len_t = np.cumsum([0] + [ c.length for c in ho.path_segments ])
            seg_boundaries = seg_len_t / ho.length * ho.slide_duration + ho.t
            for boundary in seg_boundaries[1:-1]:
                sig[1] += smooth_hit(frame_times, boundary)

            for start,end,curve in zip(seg_boundaries[:-1], seg_boundaries[1:], ho.path_segments):
                sig[2, range_sl(start,end)] = 0 if curve.nodes.shape[-1] == 2 else 1
        else:
            # slider is either Line or Perfect
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
            else: # is a slider
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

    sig = np.concatenate([hits, slider, cursor], axis=0) * 2 - 1
    assert sig.shape[0] == MAP_SIGNAL_DIM
    return sig