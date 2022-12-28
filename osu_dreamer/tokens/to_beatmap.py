
import bezier
import numpy as np

from osu_dreamer.osu.hit_objects import TimingPoint

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

def iter_sentences(tokens):
    sentence = []
    for tok in tokens:
        if tok == 'END':
            yield sentence
            sentence = []
        else:
            sentence.append(tok)

def parse_position(tok_iter):
    x_tok, y_tok = next(tok_iter), next(tok_iter)
    return int(x_tok[1:]), int(y_tok[1:])


def to_beatmap(metadata, tokens):
    """
    returns the beatmap as the string contents of the beatmap file
    """

    beat_snap, timing_points = False, [TimingPoint(0, 1000, None, 4, None)]

    beat_length = timing_points[0].beat_length
    base_slider_vel = 100 / beat_length
    beat_offset = timing_points[0].t

    hos = [] # hit objects
    tps = [] # timing points

    last_up = None
    for sentence in iter_sentences(tokens):
        toks = iter(sentence)
        t = next(toks)
                
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

        new_combo = 0
        typ = next(toks)
        if typ == 'NEW_COMBO':
            new_combo = 4
            typ = next(toks)

        if typ == 'SPINNER':
            u = next(toks)
            hos.append(f"256,192,{t},{8 + new_combo},0,{u}")
            last_up = u
        elif typ == 'CIRCLE':
            x,y = parse_position(toks)
            hos.append(f"{x},{y},{t},{1 + new_combo},0,0:0:0:0:")
            last_up = t
        elif typ == 'SLIDER':
            # parse control points
            ctrl_pts = [parse_position(toks)]
            length = 0
            for tok in toks:
                if tok == 'LINE':
                    ctrl_pts.append(parse_position(toks))
                    length += bezier.Curve.from_nodes(np.array(ctrl_pts[-2:]).T).length
                    ctrl_pts.append(ctrl_pts[-1])
                elif tok == 'CUBIC':
                    ctrl_pts.extend([ parse_position(toks) for _ in range(3) ])
                    length += bezier.Curve.from_nodes(np.array(ctrl_pts[-4:]).T).length
                    ctrl_pts.append(ctrl_pts[-1])
                else:
                    ctrl_pts.pop()
                    break

            # parse slides
            slides = 1
            while tok == 'SLIDE':
                slides += 1
                tok = next(toks)

            u = tok
            SV = length * slides / (u-t) / base_slider_vel

            x1,y1 = ctrl_pts[0]
            curve_pts = "|".join(f"{x}:{y}" for x,y in ctrl_pts[1:])
            hos.append(f"{x1},{y1},{t},{2 + new_combo},0,B|{curve_pts},{slides},{length:.2f}")
            last_up = u
            
            if len(tps) == 0:
                print('warning: inherited timing point added before any uninherited timing points')
            tps.append(f"{t},{-100/SV},4,0,0,50,0,0")
            
    return map_template.format(**metadata, timing_points="\n".join(tps), hit_objects="\n".join(hos))