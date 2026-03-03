import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_multi_roc_sci(curves, save_path):
    """
    curves: list of dict
      [{
        'snr': -9,
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
        fpr, tpr, _ = roc_curve(item['y_true'], item['y_score'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.8, label=f"SNR={item['snr']} dB (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.tick_params(direction='in', length=5, width=1.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    ax.grid(True, linestyle='-', alpha=0.5)
    ax.legend(loc='lower right', frameon=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=600)
    plt.close(fig)