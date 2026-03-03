import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_multi_pr_sci(curves, save_path):
    """
    curves: list of dict
        [{
            'snr': snr_value,
            'y_true': ndarray,
            'y_score': ndarray
        }, ...]
    """

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.markersize'] = 6

    fig, ax = plt.subplots(figsize=(6, 4.2))

    for item in curves:
        precision, recall, _ = precision_recall_curve(
            item['y_true'], item['y_score']
        )

        ap = average_precision_score(
            item['y_true'], item['y_score']
        )

        ax.plot(
            recall, precision,
            linewidth=1.8,
            label=f"SNR={item['snr']} dB (AP={ap:.3f})"
        )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.tick_params(direction='in', length=5, width=1.2)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    ax.grid(True, linestyle='-', alpha=0.5)
    ax.legend(loc='lower left', frameon=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=600)
    plt.close(fig)