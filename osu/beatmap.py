from __future__ import annotations
from typing import Iterable, Tuple

import re
from pathlib import Path

import numpy as np

import scipy

from .hit_objects import Inherited, Uninherited, Circle, Spinner, Slider
from .sliders import from_control_points

# std dev of impulse indicating a hit
HIT_SD = 5
    
def normal_pdf(x: np.ndarray, mu: float, sigma: float):
    """
    PDF of a normal distribution with mean `mu` and std. dev `sigma`
    evaluated at values in `x`
    """
    return np.exp(-.5 * ((x-mu)/sigma) **2) / sigma / (2*np.pi)**.5


def smooth_hit(x: np.ndarray, mu: "Union[float, Tuple[float, float]]", sigma: float = HIT_SD):
    """
    a smoothed impulse
    modelled using a normal distribution with mean `mu` and std. dev `sigma`
    evaluated at values in `x`
    """
    
    if isinstance(mu, (float, int)):
        z = (x-mu)/sigma
    elif isinstance(mu, tuple):
        a,b = mu
        z = np.where(x<a,x-a,np.where(x<b,0,x-b)) / sigma
    else:
        raise NotImplementedError(f"`mu` must be float or tuple, got {type(mu)}")
        
    return np.exp(-.5 * z**2)


class Beatmap:
    SONGS_PATH = "./osu!/Songs" # replace with path to `osu!/Songs` directory

    @classmethod
    def all_maps(cls, src_path: str = None) -> Iterable[Beatmap]:
        for p in Path(src_path or cls.SONGS_PATH).glob("*/*.osu"):
            try:
                bm = Beatmap(p)
            except Exception as e:
                print(f"{p}: {e}")
                continue

            # only osu!standard
            if bm.mode != 0:
                continue

            yield bm
            
    @classmethod
    def all_mapsets(cls, src_path: str = None) -> Iterable[Tuple[Path, Iterable[Beatmap]]]:
        for mapset_dir in Path(src_path or cls.SONGS_PATH).iterdir():
            if not mapset_dir.is_dir():
                continue
                
            maps = []
            mapset_id = None
            audio_file = None
            for map_file in mapset_dir.glob("*.osu"):
                try:
                    bm = Beatmap(map_file)
                except Exception as e:
                    print(f"{map_file}: {e}")
                    continue

                # only osu!standard
                if bm.mode != 0:
                    continue
                    
                maps.append(bm)
                    
                if audio_file is None:
                    audio_file = bm.audio_filename
                elif audio_file != bm.audio_filename:
                    break
                    
                if mapset_id is None:
                    mapset_id = bm.mapset_id
                elif mapset_id != bm.mapset_id:
                    break
            else:
                if audio_file is None or mapset_id is None or len(maps) == 0:
                    continue
                yield (mapset_id, audio_file, maps)
        
        return

    @staticmethod
    def parse_map_file(bmlines):
        LIST_SECTIONS = ["Events", "TimingPoints", "HitObjects"]
        cfg = {}
        section = None
        for l in bmlines:
            # comments
            if l.startswith("//"):
                continue

            # section end check
            if l.strip() == "":
                section = None
                continue

            # header check
            m = re.search(r"^\[(.*)\]$", l)
            if m is not None:
                section = m.group(1)
                if section in LIST_SECTIONS:
                    cfg[section] = []
                else:
                    cfg[section] = {}
                continue

            if section is None:
                continue

            if section in LIST_SECTIONS:
                cfg[section].append(l.strip())
            else:
                # key-value check
                m = re.search(r"^(\w*)\s?:\s?(.*)$", l)
                if m is not None:
                    cfg[section][m.group(1)] = m.group(2).strip()

        return cfg

    def __repr__(self):
        return f"{self.title} [{self.version}]"

    def __init__(self, filename, meta_only=False):
        
        self.filename = Path(filename)

        with open(filename) as f:
            cfg = self.parse_osu_map(f)

        self.audio_filename = self.filename.parent / cfg["General"]["AudioFilename"]

        self.mode = int(cfg["General"]["Mode"])

        self.title = cfg["Metadata"]["Title"]
        self.artist = cfg["Metadata"]["Artist"]
        self.creator = cfg["Metadata"]["Creator"]
        self.version = cfg["Metadata"]["Version"]

        self.hp = float(cfg["Difficulty"]["HPDrainRate"])
        self.cs = float(cfg["Difficulty"]["CircleSize"])
        self.od = float(cfg["Difficulty"]["OverallDifficulty"])

        try:
            self.ar = float(cfg["Difficulty"]["ApproachRate"])
        except KeyError:
            self.ar = 7

        # base slider velocity in hundreds of osu!pixels per beat
        self.slider_mult = float(cfg["Difficulty"]["SliderMultiplier"])

        # slider ticks per beat
        self.slider_tick = float(cfg["Difficulty"]["SliderTickRate"])

        try:
            self.beat_divisor = int(cfg["Editor"]["BeatDivisor"])
        except KeyError:
            self.beat_divisor = 4

        self.unparsed_hitobjects = cfg["HitObjects"]
        self.unparsed_timingpoints = cfg["TimingPoints"]
        self.unparsed_events = cfg["Events"]
        if not meta_only:
            self.parse_map_data()

    def parse_map_data(self):
        self.parse_hit_objects(self.unparsed_hitobjects)
        del self.unparsed_hitobjects

        self.parse_timing_points(self.unparsed_timingpoints)
        assert len(self.timing_points) > 0
        del self.unparsed_timingpoints

        self.parse_events(self.unparsed_events)
        del self.unparsed_events

    def parse_events(self, lines):
        self.events = []
        for l in lines:
            ev = l.strip().split(",")
            if ev[0] == 2:
                self.events.append(ev)

    def parse_timing_points(self, lines):
        self.timing_points = []
        for l in lines:
            vals = [ float(x) for x in l.strip().split(",") ]
            t = vals[0]
            beat_length = vals[1]
            meter = vals[2]
            uninherited = vals[6]
            inh = uninherited == 0

            # > For uninherited timing points, the duration of a beat, in milliseconds.
            # > For inherited timing points, a negative inverse slider velocity multiplier, as a percentage.
            # > For example, -50 would make all sliders in this timing section twice as fast as SliderMultiplier.
            x = beat_length if uninherited else round(-100 / float(beat_length), 3)

            try:
                if x == next(tp.x
                    for tp in self.timing_points[::-1]
                    if isinstance(tp, Inherited)
                ):
                    # x value same as last point of same inheritance
                    continue
            except StopIteration:  # no prior point of same inheritance
                pass

            self.timing_points.append(Inherited(int(t), x) if inh else Uninherited(int(t), x, int(meter)))
            
        if len(self.timing_points) == 0:
            raise ValueError("no timing points")

    def parse_hit_objects(self, lines):
        self.hit_objects = []
        for l in lines:
            spl = l.strip().split(",")
            x, y, t, k = [int(x) for x in spl[:4]]
            if k & (1 << 0):  # hit circle
                self.hit_objects.append(Circle(t, x, y))
            elif k & (1 << 1):  # slider
                curve, slides, length = spl[5:8]
                _, *control_points = curve.split("|")
                control_points = [np.array([x,y])] + [
                    np.array(list(map(int, p.split(":")))) for p in control_points
                ]
                self.hit_objects.append(
                    from_control_points(t, int(slides), float(length), control_points)
                )
            elif k & (1 << 3):  # spinner
                self.hit_objects.append(Spinner(t, int(spl[5])))
            
        if len(self.hit_objects) == 0:
            raise ValueError("no hit objects")

    def slider_duration(self, slider):
        """
        return slider speed in osu!pixels per ms
        """
        blen = self.timing_points[0].x
        for tp in self.timing_points[::-1]:
            if isinstance(tp, Uninherited) and tp.t <= slider.t:
                blen = tp.x
                break

        smult = self.slider_mult
        for tp in self.timing_points[::-1]:
            if isinstance(tp, Inherited) and tp.t <= slider.t:
                smult *= tp.x
                break

        return slider.length / (smult * 100) * blen * slider.slides


    def hit_signal(self, frames: "...,L", hop_length, n_fft, sr) -> "2,L":
        """
        returns an array encoding the hits occurring at the times represented by `frames`
        - [0] represents hits
        - [1] represents holds, maintained at the maximum value for the duration of the hold

        `hop_length`, `n_fft`, `sr`: frame attributes
        """

        # frame_times[i] = time at frames[..., i] (in ms)
        frame_times: "L," = (np.arange(frames.shape[-1]) * hop_length + n_fft // 2) / sr * 1000.

        sig = np.zeros((2, len(frame_times)))
        for ho in self.hit_objects:
            if isinstance(ho, Circle):
                sig[0] += smooth_hit(frame_times, ho.t)
            elif isinstance(ho, Spinner):
                sig[1] += smooth_hit(frame_times, (ho.t, ho.u))
            else: # Slider
                sig[1] += smooth_hit(frame_times, (ho.t, ho.t+int(self.slider_duration(ho))))

        return sig
    
    
    def cursor_signal(self, frames: "...,L", hop_length, n_fft, sr) -> "2,L":
        """
        return [2,L] where [{0,1},i] is the {x,y} position at the time of `frames[i]`
        
        - `hop_length`, `n_fft`, `sr`: frame attributes
        """
        
        frame_times: "L," = (np.arange(frames.shape[-1]) * hop_length + n_fft // 2) / sr * 1000.
        
        return np.array([ self.cursor(t)[0] for t in frame_times ]).T
    
    
    def map_signal(self, frames: "...,L", hop_length, n_fft, sr) -> "4,L": 
        """
        returns a [4,L] scaled to [-1,1]
        """
        
        hits: "2,L" = self.hit_signal(frames, hop_length, n_fft, sr)
        cursor: "2,L" = self.cursor_signal(frames, hop_length, n_fft, sr) / np.array([[512],[384]])
        
        return np.concatenate([hits, cursor], axis=0) * 2 - 1
    
    @classmethod
    def signal_to_hits(cls, sig, hop_length, n_fft, sr):
        """
        returns an N-tuple of lists where each list contains the times (in ms) when
        a hit of type {tap, hold start, hold end} occurs
        
        `sig`: [2,L] array 
        `hop_length`, `n_fft`, `sr`: frame attributes
        """
        f_b = max(2, HIT_SD*6//hop_length)
        feat = smooth_hit(np.arange(-f_b, f_b+1) * hop_length, 0)
        
        tap_sig, hold_sig = sig
        hold_sig_grad = np.gradient(hold_sig)
        hold_start_sig = np.maximum(0, hold_sig_grad)
        hold_end_sig = -np.minimum(0, hold_sig_grad)
        
        off = hop_length / sr * 1000
        
        res = []
        for hit_sig, hit_offset, peak_h in zip(
            [tap_sig, hold_start_sig, hold_end_sig],
            [0,off,-off],
            [.5, .25, .25],
        ):
            corr = scipy.signal.correlate(hit_sig, feat, mode='same')
            hit_peaks = scipy.signal.find_peaks(corr, height=peak_h)[0]
            
            res.append(np.rint(
                ( hit_peaks * hop_length + n_fft // 2 ) / sr * 1000. + hit_offset
            ).astype(int).tolist())
            
        return tuple(res)
    
    @classmethod
    def signal_to_map(cls, audio_filename, sig, hop_length, n_fft, sr, name = None):
        hit_signal, cursor_signal = sig[:2], sig[2:]
        hit_signal = (hit_signal+1)/2
        
        padding = .1
        
        cs_valid = cursor_signal * np.clip(hit_signal.max(axis=0, keepdims=True), 0, 1)
        cs_valid_center = cs_valid.mean(axis=1, keepdims=True)
        cursor_signal = (cursor_signal - cs_valid_center + 1)/2
        cursor_signal *= np.array([[512],[384]]) * (1 - 2*padding)
        cursor_signal += np.array([[512],[384]]) * padding
        
        frame_times: "L," = (np.arange(sig.shape[-1]) * hop_length + n_fft // 2) / sr * 1000.
        hits = cls.signal_to_hits(hit_signal, hop_length, n_fft, sr)
    
        def get_pos(t):
            i = (t / 1000. * sr - n_fft // 2) / hop_length
            ia = int(max(0, np.floor(i)))
            ib = int(min(len(frame_times)-1, np.ceil(i)))
            f = i - ia
            
            a = cursor_signal[:, ia]
            b = cursor_signal[:, ib]
            
            return (1-f) * a + f * b
        
        def get_ctrl_pts(a,b, step=5):
            pts = [get_pos(a)]
            l = 0
            x = a + step
            while x < b:
                pts.append(get_pos(x))
                l += np.linalg.norm(pts[-1] - pts[-2]) 
                pts.append(get_pos(x))
                x += step
            pts.append(get_pos(b))
            l += np.linalg.norm(pts[-1] - pts[-2]) 
            return pts, l
            
        beat_length = 1000
        slider_mult = 1
            
        
        template = \
f"""osu file format v14

[General]
AudioFilename:{audio_filename.name}
AudioLeadIn: 0
Mode: 0

[Metadata]
Title:{audio_filename.parent.name}
TitleUnicode:{audio_filename.parent.name}
Creator:signal_map
Version:{name or "signal_map"}
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate: 0
CircleSize: 3
OverallDifficulty: 0
ApproachRate: 9.5
SliderMultiplier: {slider_mult}
SliderTickRate: 1

[TimingPoints]"""
        
        tps = [f"0,{beat_length},4,0,0,50,1,0"]
        
        # sort hits
        sorted_hits = []
        for hit_type, times in enumerate(hits):
            sorted_hits.extend(( (t, hit_type) for t in times ))
        sorted_hits = sorted(sorted_hits)
        
        hos = []
        hold_start = None
        for t, t_type in sorted_hits:
            new_combo = 4
            # new_combo = 4 if len(hos) % 8 == 0 else 0
            if hold_start is None:
                # not holding
                if t_type == 0:
                    # tap, make hit circle
                    x,y = get_pos(t)
                    hos.append(f"{x},{y},{t},{1 + new_combo},0")
                elif t_type == 1:
                    # start new hold
                    hold_start = t
                elif t_type == 2:
                    # hold_end when no preceding hold_start, ignore
                    pass
            else:
                # holding
                if t_type == 0:
                    # tap when holding, ignore
                    pass
                elif t_type == 1:
                    # hold_start when already holding, ignore
                    pass
                elif t_type == 2:
                    # end hold, make slider
                    
                    ctrl_pts, length = get_ctrl_pts(hold_start, t)
                    
                    dur = t - hold_start
                    # dur = length / (slider_mult * 100 * SV) * beat_length
                    SV = length * beat_length / dur / 100 / slider_mult
                    
                    x1,y1 = ctrl_pts[0]
                    curve_pts = "|".join(f"{x}:{y}" for x,y in ctrl_pts[1:])
                    hos.append(f"{x1},{y1},{hold_start},{2 + new_combo},0,B|{curve_pts},1,{length}")
                    tps.append(f"{hold_start-1},{-100/SV},4,0,0,50,0,0")
                    hold_start = None
                    
                    
        return "\n".join([template, *tps, """

[HitObjects]""", *hos])
    
    def cursor(self, t):
        """
        return cursor position + time since last click at time t (ms)
        """
        
        cx,cy = 256,192

        # before first hit object
        if t < self.hit_objects[0].t:
            ho = self.hit_objects[0]
            if isinstance(ho, Circle):
                return (ho.x, ho.y), np.inf
            elif isinstance(ho, Spinner):
                return (cx, cy), np.inf
            elif isinstance(ho, Slider):
                return ho.lerp(0), np.inf

        for ho, nho in zip(self.hit_objects, self.hit_objects[1:]):
            if ho.t <= t < nho.t:
                break
        else:  # after last hit object
            ho = self.hit_objects[-1]
            nho = None

        t -= ho.t

        # next hit object
        if isinstance(nho, Circle):
            nx, ny = nho.x, nho.y
        elif isinstance(nho, Spinner):
            nx, ny = (cx, cy)  # spin starting point
        elif isinstance(nho, Slider):
            nx, ny = nho.lerp(0)

        if isinstance(ho, Spinner):
            spin_duration = ho.u - ho.t
            if t < spin_duration:  # spinning
                return (cx, cy), 0
            else:  # moving
                t -= spin_duration
                if nho:  # to next hit object

                    f = t / (nho.t - ho.t - spin_duration)  # interpolation factor

                    return ((1 - f) * cx + f * nx, (1 - f) * cy + f * ny), t
                else:  # last object
                    return (cx, cy), t

        elif isinstance(ho, Circle):
            if nho:  # moving to next hit object
                f = t / (nho.t - ho.t)  # interpolation factor

                return ((1 - f) * ho.x + f * nx, (1 - f) * ho.y + f * ny), t
            else:
                return (ho.x, ho.y), t
        elif isinstance(ho, Slider):
            slide_duration = self.slider_duration(ho)

            if t < slide_duration:  # sliding
                single_slide = slide_duration / ho.slides

                ts = t % (single_slide * 2)
                if ts < single_slide:  # start -> end
                    return ho.lerp(ts / single_slide), 0
                else:  # end -> start
                    return ho.lerp(2 - ts / single_slide), 0
            else:  # moving
                t -= slide_duration
                end = ho.lerp(ho.slides % 2)

                if nho:  # to next hit object
                    f = t / (nho.t - ho.t - slide_duration)  # interpolation factor

                    return ((1 - f) * end[0] + f * nx, (1 - f) * end[1] + f * ny), t
                else:
                    return (end[0], end[1]), t
