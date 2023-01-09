import numpy as np

from osu_dreamer.osu.hit_objects import Circle, Slider, Spinner
from osu_dreamer.osu.sliders import Line, Perfect, Bezier

from .smooth_hit import encode_hit, encode_hold

# helpers

def hit_object_pairs(beatmap, frame_times: "L,"):
    """
    for each `t` in `frame_times` yields the pair of adjacent hit objects that surround `t`
    """
    pairs = zip([None] + beatmap.hit_objects, beatmap.hit_objects + [None])
    a,b = next(pairs)
    for t in frame_times:
        while b is not None and b.t <= t:
            a,b = next(pairs)
        # here: b is None or the first hit object that starts after t
        yield a,b

def active_timing_point(beatmap, frame_times: "L,", uninherited=False):
    """
    for each `t` in `frame_times` yields the timing point active at `t`, or None if before all timing points
    """
    tps = iter(beatmap.uninherited_timing_points if uninherited else beatmap.timing_points + [None])
    cur_tp, next_tp = None, next(tps)
    for t in frame_times:
        while next_tp is not None and t > next_tp.t:
            cur_tp, next_tp = next_tp, next(tps)
        # here: next_tp is None or first timing point that starts after t
        # => cur_tp is None or current active timing point
        yield cur_tp

# map signal

HIT_DIM = 4
def hit_signal(beatmap, frame_times: "L,") -> "HIT_DIM,L":
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
            encode_hit(sig[0], frame_times, ho.t)
        elif isinstance(ho, Slider):
            encode_hold(sig[1], frame_times, ho.t, ho.end_time())
        elif isinstance(ho, Spinner):
            encode_hold(sig[2], frame_times, ho.t, ho.end_time())

        if ho.new_combo:
            encode_hit(sig[3], frame_times, ho.t)

    assert sig.shape[0] == HIT_DIM
    return sig


SLIDER_DIM = 2
def slider_signal(beatmap, frame_times: "L,") -> "SLIDER_DIM,L":
    """
    returns an array encoding slider metadata
    - [0] represents a slider repeat
    - [1] represents a bezier slider segment boundary
    """
    sig = np.zeros((2, len(frame_times)))

    for ho in beatmap.hit_objects:
        if not isinstance(ho, Slider):
            continue

        # only render first repeat slide
        if ho.slides > 1:
            encode_hit(sig[0], frame_times, ho.t + ho.slide_duration)
                
        if isinstance(ho, Bezier):
            # render segment boundaries along the first slide only
            seg_len_t = np.cumsum([0] + [ c.length for c in ho.path_segments ])
            seg_boundaries = seg_len_t / ho.length * ho.slide_duration + ho.t
            for boundary in seg_boundaries[1:-1]:
                encode_hit(sig[1], frame_times, boundary)
            
    assert sig.shape[0] == SLIDER_DIM
    return sig


CURSOR_DIM = 2
def cursor_signal(beatmap, frame_times: "L,") -> "CURSOR_DIM,L":
    """
    return [CURSOR_DIM,L] where [{0,1},i] is the {x,y} position at the times represented by `frames`

    - `frame_times`: array of times at each frame in ms
    """
    pos = []
    for t, (a,b) in zip(frame_times, hit_object_pairs(beatmap, frame_times)):
        if a is None:
            # before first hit object
            pos.append(b.start_pos())
        elif t < a.end_time():
            # hitting current hit object
            # note: will not be a `Circle` due to `hit_object_pairs`
            if isinstance(a, Spinner):
                pos.append(a.start_pos())
            else: # is a slider
                ts = (t - a.t) % (a.slide_duration * 2) / a.slide_duration
                if ts < 1:  # start -> end
                    pos.append(a.lerp(ts))
                else:  # end -> start
                    pos.append(a.lerp(2 - ts))
        elif b is None:
            # after last hit object
            pos.append(a.end_pos())
        else:
            # moving to next hit object
            f = (t - a.end_time()) / (b.t - a.end_time())
            pos.append((1 - f) * a.end_pos() + f * b.start_pos())
            
    sig = (np.array(pos) / np.array([512,384])).T
    assert sig.shape[0] == CURSOR_DIM
    return sig

# auxiliary signal

KIAI_DIM = 1
def kiai_signal(beatmap, frame_times: "L,") -> "1,L":
    """
    returns a [1,L] where [0,t] is 1 if kiai time is enabled otherwise 0
    """
    sig = np.array([
        1 if tp is not None and tp.kiai else 0
        for tp in active_timing_point(beatmap, frame_times)
    ])[None]
    assert sig.shape[0] == KIAI_DIM
    return sig

MOTION_DIM = 3
def motion_signal(beatmap, frame_times: "L,") -> "3,L":
    """
    returns a [6,L] with values in [0,1] where:
    - [0] represents magnitude of velocity (speed)
    - [1] represents the x component of the normalized velocity
    - [2] represents the y component of the normalized velocity
    """

    APPROX_MAX_SPEED = 640

    sig = []
    for t, (a,b) in zip(frame_times, hit_object_pairs(beatmap, frame_times)):
        if a is None:
            # before first hit object
            vel = np.array([0,0])
        elif t < a.end_time():
            # hitting current hit object
            # note: will not be a `Circle` due to `hit_object_pairs`
            if isinstance(a, Spinner):
                vel = np.array([0,0])
            elif isinstance(a, Slider):
                ts = (t - a.t) % (a.slide_duration * 2) / a.slide_duration
                if ts < 1:  # start -> end
                    vel_dir = 1
                else:  # end -> start
                    vel_dir = -1
                    ts = 2 - ts

                vel = a.vel(ts) * vel_dir
        elif b is None:
            # after last hit object
            vel = np.array([0,0])
        else:
            # moving to next hit object
            vel = (b.start_pos() - a.end_pos()) / (b.t - a.end_time())

        # compute signal
        speed = np.linalg.norm(vel) # [0, inf)
        exp_speed = np.exp(speed * 6 / APPROX_MAX_SPEED)
        vel = vel/speed if speed > 0 else np.array([0,0])

        sig.append([
            (exp_speed-1)/exp_speed,
            (vel[0]+1)/2,
            (vel[1]+1)/2,
        ])
    
    sig = np.array(sig).T
    assert sig.shape[0] == MOTION_DIM
    return sig
            

BEAT_DIM = 2
def beat_signal(beatmap, frame_times: "L,") -> "2,L":
    """
    returns a [2,L] where:
    - [0] represents beats
    - [1] represents the first beat of a measure
    """

    # timing point boundaries
    tpt: "N+1,1" = np.array([-np.inf] + [ utp.t for utp in beatmap.uninherited_timing_points[1:] ] + [np.inf])[:, None]

    # active_tp[0, i] = index into `utps` of the timing point active at frame `i`
    active_tp: "N,L" = (frame_times >= tpt[:-1]) & (frame_times < tpt[1:])
    active_tp: "1,1,L" = np.argwhere(active_tp.T)[None, None, :, 1]

    choices: "TP,2,L" = np.array([
        [
            (frame_times - utp.t) / utp.beat_length / f * np.pi * 2
            for f in [1, utp.meter]
        ]
        for utp in beatmap.uninherited_timing_points
    ])

    sig: "1,2,L" = np.take_along_axis(choices, active_tp, axis=0)
    
    sig = np.cos(sig[0])**2
    assert sig.shape[0] == BEAT_DIM
    return sig
        
# AUX_DIM = KIAI_DIM + BEAT_DIM + MOTION_DIM
AUX_DIM = 0
MAP_SIGNAL_DIM = HIT_DIM + SLIDER_DIM + CURSOR_DIM
X_DIM = MAP_SIGNAL_DIM + AUX_DIM

def from_beatmap(beatmap, frame_times: "L,") -> "X_DIM,L":
    """
    returns a [X_DIM,L] scaled to [-1,1]
    the first `AUX_DIM` signals are auxiliary and should be ignored during map generation

    - `frame_times`: array of times at each frame in ms
    """

    sig = np.concatenate([
        ### AUXILIARY SIGNALS
        # auxiliary signals should be predicted by the model, but are not used during map generation
        # the idea is that if the model can learn these higher level features, it can also use them to inform map generation

        # kiai_signal(beatmap, frame_times),
        # beat_signal(beatmap, frame_times),
        # motion_signal(beatmap, frame_times),

        ### MAP SIGNALS
        # map signals are interpreted for the purpose of map generation
        hit_signal(beatmap, frame_times),
        slider_signal(beatmap, frame_times),
        cursor_signal(beatmap, frame_times),
    ], axis=0) * 2 - 1
    assert sig.shape[0] == MAP_SIGNAL_DIM
    return sig