import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def plot_f1_vs_snr(snr, alice_acc, eve_acc, save_path, pos_prior=0.5):  # 正类占所有的样本的比例
    """
    snr        : SNR数组
    alice_acc  : Alice检测准确率 (≈ TPR)
    eve_acc    : Eve检测准确率 (≈ TNR)
    pos_prior  : 正类比例（默认0.5）
    """

    # =====================
    # 计算F1
    # =====================
    neg_prior = 1.0 - pos_prior

    tpr = alice_acc
    tnr = eve_acc
    fpr = 1.0 - tnr

    precision = (tpr * pos_prior) / (tpr * pos_prior + fpr * neg_prior + 1e-12)
    recall = tpr

    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

    # =====================
    # 全局风格设置（完全按照你给的模板）
    # =====================
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.markersize'] = 6

    # =====================
    # 绘图
    # =====================
    fig, ax = plt.subplots(figsize=(6, 4.2))

    ax.plot(
        snr, f1 * 100,  # 变成百分比
        color='black', linestyle='-', linewidth=1.8,
        marker='o', markerfacecolor='none', markeredgecolor='black',
        label='Proposed scheme'
    )

    # =====================
    # 坐标轴设置
    # =====================
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('F1-score (%)')

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

    plot_f1_vs_snr(
        snr,
        alice_acc,
        eve_acc,
        save_path='f1_score_sci_style.png'
    )