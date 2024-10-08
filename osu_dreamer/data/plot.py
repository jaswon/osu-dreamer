
from contextlib import contextmanager

from jaxtyping import Float

from numpy import ndarray
import librosa

import matplotlib.pyplot as plt

@contextmanager
def plot_signals(audio: Float[ndarray, "A L"], signals: list[Float[ndarray, "X L"]]):
        margin, margin_left = .1, .5
        height_ratios = [.8] + [.6] * len(signals)
        plots_per_row = len(height_ratios)
        w, h = audio.shape[-1] * .01, sum(height_ratios) * .4

        # split plot across multiple rows
        split = ((w/h)/(3/5)) ** .5 # 3 wide by 5 tall aspect ratio
        split = int(split + 1)
        w = w // split
        h = h * split
        height_ratios = height_ratios * split
        
        fig, all_axs = plt.subplots(
            len(height_ratios), 1,
            figsize=(w, h),
            sharex=True,
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
            ax1, *axs = all_axs[i * plots_per_row: (i+1) * plots_per_row]
            sl = (..., slice(i * win_len, (i+1) * win_len))

            ax1.imshow(librosa.power_to_db(audio[sl]), origin="lower", aspect='auto')
            
            for (i, sample), ax in zip(enumerate(signals), axs):
                ax.margins(x=0)
                for ch in sample[sl]:
                    ax.plot(ch)

        yield fig
        plt.close(fig)