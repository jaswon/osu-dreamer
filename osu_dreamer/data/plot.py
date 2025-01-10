
from contextlib import contextmanager

from jaxtyping import Float

from numpy import ndarray

import matplotlib.pyplot as plt

@contextmanager
def plot_signals(
    audio: Float[ndarray, "A L"], 
    signals: list[Float[ndarray, "X L"]],
    temporal_scale: float = .01,
):
    margin, margin_left = .1, .5
    height_ratios = [.8] + [.6] * len(signals)
    plots_per_row = len(height_ratios)
    w, h = audio.shape[-1] * temporal_scale, sum(height_ratios) * .4

    # split plot across multiple rows
    split = int(1 + ((w/h)/(3/5)) ** .5) # 3 wide by 5 tall aspect ratio
    w = w // split
    h = h * split
    height_ratios = height_ratios * split
    
    fig, all_axs = plt.subplots(
        len(height_ratios), 1,
        figsize=(w, h),
        gridspec_kw=dict(
            height_ratios=height_ratios,
            hspace=.1,
            left=margin_left/w,
            right=1-margin/w,
            top=1-margin/h,
            bottom=margin/h,
        )
    )

    win_len = audio.shape[-1] // split
    for i in range(split):
        sl = (..., slice(i * win_len, (i+1) * win_len))
        ax1, *axs = all_axs[i * plots_per_row: (i+1) * plots_per_row]
        ax1.pcolormesh(audio[sl])
        for sample, ax in zip(signals, axs):
            ax.margins(x=0)
            ax.plot(sample[sl].T)

    yield fig
    plt.close(fig)