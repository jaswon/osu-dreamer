from __future__ import annotations
from typing import Iterable, Tuple

import re
from pathlib import Path
import bisect

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

        with open(filename, encoding='utf-8') as f:
            cfg = self.parse_map_file(f)

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
            new_combo = (k&(1<<2)) > 0 
            if k & (1 << 0):  # hit circle
                self.hit_objects.append(Circle(t, new_combo, x, y))
            elif k & (1 << 1):  # slider
                curve, slides, length = spl[5:8]
                _, *control_points = curve.split("|")
                control_points = [np.array([x,y])] + [
                    np.array(list(map(int, p.split(":")))) for p in control_points
                ]
                self.hit_objects.append(
                    from_control_points(t, new_combo, int(slides), float(length), control_points)
                )
            elif k & (1 << 3):  # spinner
                self.hit_objects.append(Spinner(t, new_combo, int(spl[5])))
            
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


    def hit_signal(self, frames: "...,L", hop_length, n_fft, sr) -> "4,L":
        """
        returns an array encoding the hits occurring at the times represented by `frames`
        - [0] represents hits
        - [1] represents slider holds
        - [2] represents spinner holds
        - [3] represents new combos

        `hop_length`, `n_fft`, `sr`: frame attributes
        """

        # frame_times[i] = time at frames[..., i] (in ms)
        frame_times: "L," = (np.arange(frames.shape[-1]) * hop_length + n_fft // 2) / sr * 1000.

        sig = np.zeros((4, len(frame_times)))
        for ho in self.hit_objects:
            if isinstance(ho, Circle):
                sig[0] += smooth_hit(frame_times, ho.t)
            elif isinstance(ho, Slider):
                sig[1] += smooth_hit(frame_times, (ho.t, ho.t+int(self.slider_duration(ho))))
            else: # Spinner
                sig[2] += smooth_hit(frame_times, (ho.t, ho.u))
                
            if ho.new_combo:
                sig[3] += smooth_hit(frame_times, ho.t)

        return sig
    
    
    def cursor_signal(self, frames: "...,L", hop_length, n_fft, sr) -> "2,L":
        """
        return [2,L] where [{0,1},i] is the {x,y} position at the time of `frames[i]`
        
        - `hop_length`, `n_fft`, `sr`: frame attributes
        """
        
        frame_times: "L," = (np.arange(frames.shape[-1]) * hop_length + n_fft // 2) / sr * 1000.
        
        return np.array([ self.cursor(t)[0] for t in frame_times ]).T
    
    
    MAP_SIGNAL_DIM = 6
    
    def map_signal(self, frames: "...,L", hop_length, n_fft, sr) -> "6,L": 
        """
        returns a [6,L] scaled to [-1,1]
        """
        
        hits: "4,L" = self.hit_signal(frames, hop_length, n_fft, sr)
        cursor: "2,L" = self.cursor_signal(frames, hop_length, n_fft, sr) / np.array([[512],[384]])
        
        return np.concatenate([hits, cursor], axis=0) * 2 - 1
    
    @classmethod
    def signal_to_hits(cls, sig, hop_length, n_fft, sr):
        """
        returns an 6-tuple of lists where each list contains the times (in ms) when
        a hit of type {tap, slider start, slider end, spinner start, spinner end, new_combo} occurs
        
        `sig`: [4,L] array 
        `hop_length`, `n_fft`, `sr`: frame attributes
        """
        f_b = max(2, HIT_SD*6//hop_length)
        feat = smooth_hit(np.arange(-f_b, f_b+1) * hop_length, 0)
        
        tap_sig, slider_sig, spinner_sig, new_combo_sig = sig
        
        slider_sig_grad = np.gradient(slider_sig)
        slider_start_sig = np.maximum(0, slider_sig_grad)
        slider_end_sig = -np.minimum(0, slider_sig_grad)
        
        spinner_sig_grad = np.gradient(spinner_sig)
        spinner_start_sig = np.maximum(0, spinner_sig_grad)
        spinner_end_sig = -np.minimum(0, spinner_sig_grad)
        
        off = hop_length / sr * 1000
        
        res = []
        for hit_sig, hit_offset, peak_h in zip(
            [tap_sig, slider_start_sig, slider_end_sig, spinner_start_sig, spinner_end_sig, new_combo_sig],
            [0,off,-off,off,-off,0],
            [.5, .25, .25, .25, .25, .5],
        ):
            corr = scipy.signal.correlate(hit_sig, feat, mode='same')
            hit_peaks = scipy.signal.find_peaks(corr, height=peak_h)[0]
            
            res.append(np.rint(
                ( hit_peaks * hop_length + n_fft // 2 ) / sr * 1000. + hit_offset
            ).astype(int).tolist())
            
        return tuple(res)
    
    @classmethod
    def signal_to_map(cls, metadata, sig, hop_length, n_fft, sr, bpm=None):
        hit_signal, cursor_signal = sig[:4], sig[4:]
        hit_signal = (hit_signal+1)/2
        
        padding = .06
        
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
            return np.array(pts), l
            
        beat_length = 1000 if bpm is None else 60 * 1000 / bpm
        base_slider_vel = 100 / beat_length
            
        
        template = \
f"""osu file format v14

[General]
AudioFilename: {metadata['audio_filename']}
AudioLeadIn: 0
Mode: 0

[Metadata]
Title: {metadata['title']}
TitleUnicode: {metadata['title']}
Artist: {metadata['artist']}
ArtistUnicode: {metadata['artist']}
Creator: osu!dreamer
Version: {metadata['version']}

[Difficulty]
HPDrainRate: 0
CircleSize: 3
OverallDifficulty: 0
ApproachRate: 9.5
SliderMultiplier: 1
SliderTickRate: 1

[TimingPoints]"""
        
        
        # sort hits
        sorted_hits = []
        tap_times, slider_start_times, slider_end_times, spinner_start_times, spinner_end_times, new_combo_times = hits
        
        sorted_hits.extend([ (t, None, 0, False) for t in tap_times ])
        sorted_hits.extend([ (s, e, 1, False) for s,e in zip(sorted(slider_start_times), sorted(slider_end_times)) ])
        sorted_hits.extend([ (s, e, 2, False) for s,e in zip(sorted(spinner_start_times), sorted(spinner_end_times)) ])
            
        sorted_hits = sorted(sorted_hits)
        
        
        # associate hits with new combos
        for new_combo_time in new_combo_times:
            idx = bisect.bisect_left(sorted_hits, (new_combo_time,))
            if idx == len(sorted_hits):
                idx = idx-1
            elif idx+1 < len(sorted_hits) and abs(new_combo_time - sorted_hits[idx][0]) > abs(sorted_hits[idx+1][0] - new_combo_time):
                idx = idx+1
            sorted_hits[idx] = ( sorted_hits[idx][0], sorted_hits[idx][1], sorted_hits[idx][2], True )

        
        hos = [] # hit objects
        tps = [f"0,{beat_length},4,0,0,50,1,0"] # timing points
        
        last_up = None
        for t, u, t_type, new_combo in sorted_hits:
            
            # ignore objects that start before the previous one ends
            if last_up is not None and t < last_up:
                continue

            new_combo = 4 if new_combo else 0
                
            if u is None:
                # hit circle
                x,y = get_pos(t)
                hos.append(f"{x},{y},{t},{1 + new_combo},0")
                last_up = t
            elif t_type == 1:
                # slider
                ctrl_pts, length = get_ctrl_pts(t, u)

                SV = length / (u-t) / base_slider_vel

                x1,y1 = ctrl_pts[0]
                curve_pts = "|".join(f"{x}:{y}" for x,y in ctrl_pts[1:])
                hos.append(f"{x1},{y1},{t},{2 + new_combo},0,B|{curve_pts},1,{length}")
                tps.append(f"{t-1},{-100/SV},4,0,0,50,0,0")
                last_up = u
            elif t_type == 2:
                # spinner
                hos.append(f"256,192,{t},{8 + new_combo},0,{u}")
                last_up = u
                    
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
