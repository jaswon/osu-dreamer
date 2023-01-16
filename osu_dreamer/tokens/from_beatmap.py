
import numpy as np

from osu_dreamer.osu.hit_objects import Circle, Spinner, Slider, HitObject
from osu_dreamer.osu.sliders import Line, Perfect, Bezier

from osu_dreamer.signal.fit_bezier import fit_bezier

# number of samples to take for cubic approximation per unit length of original curve
BEZIER_SAMPLE_DENSITY = 0.1

def arc_to_cubic_bezier(center, radius, start, end) -> "4,2":
    """standard approximation of circular arc with cubic bezier"""

    alpha = 4/3 * np.tan((end - start)/4)
    p0 = center + radius * np.array([np.cos(start), np.sin(start)])
    p3 = center + radius * np.array([np.cos(end), np.sin(end)])

    p0r = p0 - center
    p3r = p3 - center

    p1 = p0 + np.array([ -p0r[1], p0r[0] ]) * alpha
    p2 = p3 + np.array([ p3r[1], -p3r[0] ]) * alpha
    
    return [p0,p1,p2,p3]

def pos_tokens(x,y):
    return [ f'X{round(x):+d}', f'Y{round(y):+d}' ]

def from_beatmap(bm):
    sentence_starts = [] # start times of sentences
    sentence_ends = [] # end times of sentences
    sentences = []

    for ho in bm.hit_objects:
        assert isinstance(ho, HitObject)

        sentence_starts.append(round(ho.t))
        sentence_ends.append(round(ho.end_time()))

        # all sentences start with a timestamp
        sentence = [ round(ho.t) ]

        # followed by an optional new combo flag
        if ho.new_combo:
            sentence.append('NEW_COMBO')

        if isinstance(ho, Spinner):
            # spinners provide only an end timestamp
            sentence.extend([ 'SPINNER', round(ho.end_time()), 'END' ])

        elif isinstance(ho, Circle):
            # hit circles provide only a position
            sentence.extend([ 'CIRCLE', *pos_tokens(ho.x, ho.y), 'END' ])

        elif isinstance(ho, Slider):
            sentence.append('SLIDER')

            # encode slider path
            if isinstance(ho, Line):
                sentence.extend([ *pos_tokens(*ho.start), 'LINE', *pos_tokens(*ho.end) ])
            elif isinstance(ho, Perfect):
                # approximate with cubic bezier

                # approximation worsens for greater arc angles - 
                # split arc into (at most) quarter-circle sections
                num_arc_sections = int(abs(ho.end - ho.start)/(np.pi/2)) + 1
                arc_section_ends = np.linspace(ho.start, ho.end, num_arc_sections + 1)
                for i, (section_start, section_end) in enumerate(zip(arc_section_ends[:-1], arc_section_ends[1:])):
                    for j,p in enumerate(arc_to_cubic_bezier(ho.center, ho.radius, section_start, section_end)):
                        if j == 0:
                            if i == 0:
                                sentence.extend(pos_tokens(*p))
                            sentence.append('CUBIC')
                        else:
                            sentence.extend(pos_tokens(*p))
            elif isinstance(ho, Bezier):
                for i,c in enumerate(ho.path_segments):
                    # convert arbitrary bezier to cubic beziers
                    sampled_path = c.evaluate_multi(np.linspace(0, 1, int(ho.length * BEZIER_SAMPLE_DENSITY)+1)).T
                    cubic_approx = fit_bezier(sampled_path, max_err=100)

                    for k,cc in enumerate(cubic_approx):
                        typ = {2:'LINE', 4:'CUBIC'}[len(cc)]
                        for j,p in enumerate(cc):
                            if j == 0:
                                if i == 0 and k == 0:
                                    sentence.extend(pos_tokens(*p))
                                sentence.append(typ)
                            else:
                                sentence.extend(pos_tokens(*p))

            # slider sentences end by providing `SLIDE` for each slide (not including the initial)
            sentence.extend(['SLIDE'] * (ho.slides - 1))

            # followed by the end timestamp
            sentence.extend([int(ho.t + ho.slide_duration), 'END'])

        sentences.append(sentence)

    return sentences, sentence_starts, sentence_ends