import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_vs_snr(snr, alice_acc, eve_acc, save_path):

    # =====================
    # 全局风格设置（SCI标准）
    # =====================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.markersize'] = 6

    # =====================
    # 绘图
    # =====================
    fig, ax = plt.subplots(figsize=(6, 4.2))

    # Alice — 黑色实线 + 空心圆
    ax.plot(
        snr, alice_acc * 100,
        color='black', linestyle='-', linewidth=1.8,
        marker='o', markerfacecolor='none', markeredgecolor='black',
        label='Alice detection accuracy'
    )

    # Eve — 红色实线 + 空心方块
    ax.plot(
        snr, eve_acc * 100,
        color='red', linestyle='-', linewidth=1.8,
        marker='s', markerfacecolor='none', markeredgecolor='red',
        label='Eve detection accuracy'
    )

    # =====================
    # 坐标轴设置
    # =====================
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Accuracy (%)')

    ax.set_xticks(snr)
    ax.set_xlim(min(snr), max(snr))
    ax.set_ylim(0, 102)
    ax.set_yticks(np.arange(0, 101, 10))

    ax.tick_params(direction='in', length=5, width=1.2)

    # 打开四周边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    # =====================
    # 网格与图例
    # =====================
    ax.grid(True, linestyle='-', alpha=0.5)

    ax.legend(
        loc='lower right',
        frameon=True
    )

    # =====================
    # 保存
    # =====================
    fig.tight_layout()
    fig.savefig(save_path, dpi=600)
    plt.close(fig)


# =====================
# 主程序
# =====================
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

    plot_accuracy_vs_snr(
        snr,
        alice_acc,
        eve_acc,
        save_path='accuracy_sci_style.png'
    )