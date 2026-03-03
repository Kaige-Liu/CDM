import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def plot_detection_false_alarm(snr, alice_acc, eve_acc, save_path):

    # =====================
    # 计算指标
    # =====================
    Pd = alice_acc                  # Detection probability
    Pfa = 1.0 - eve_acc             # False alarm probability

    # =====================
    # 全局风格设置
    # =====================
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.markersize'] = 6

    # =====================
    # 绘图
    # =====================
    fig, ax = plt.subplots(figsize=(6, 4.2))

    # Detection probability
    ax.plot(
        snr, Pd * 100,
        color='black', linestyle='-', linewidth=1.8,
        marker='o', markerfacecolor='none', markeredgecolor='black',
        label='Detection probability'
    )

    # False alarm probability
    ax.plot(
        snr, Pfa * 100,
        color='red', linestyle='-', linewidth=1.8,
        marker='s', markerfacecolor='none', markeredgecolor='red',
        label='False alarm probability'
    )

    # =====================
    # 坐标轴设置
    # =====================
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Probability (%)')

    ax.set_xticks(snr)
    ax.set_xlim(min(snr), max(snr))
    ax.set_ylim(0, 102)
    ax.set_yticks(np.arange(0, 101, 10))

    ax.tick_params(direction='in', length=5, width=1.2)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    ax.grid(True, linestyle='-', alpha=0.5)

    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.35, 0.7),
        frameon=True  # 必须先开 frame，后面才能改
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=600)
    plt.close(fig)


if __name__ == "__main__":

    snr = np.array([-9, -6, -3, 0, 3, 6, 9, 12, 15, 18])

    alice_acc = np.array([
        0.5428788, 0.62136365, 0.70515154, 0.79666669, 0.82454548,
        0.9087879, 0.9709091, 0.98621213, 0.99515152, 0.99878788
    ])

    eve_acc = np.array([
        0.96393942, 0.9707576, 0.97196972, 0.97424244, 0.98757577,
        0.99681818, 0.99909091, 0.99727273, 0.99833333, 0.99909091
    ])

    plot_detection_false_alarm(
        snr,
        alice_acc,
        eve_acc,
        save_path='detection_false_alarm_sci_style.png'
    )