# import matplotlib
# import numpy as np
#
# matplotlib.use('Agg')
#
# import matplotlib.pyplot as plt
#
# def plot_confusion_matrix_sci(TP, FP, TN, FN, save_path, title=None, normalize=True):
#     """
#     normalize=True: 显示比例（更好看，也更好比较不同SNR）
#     """
#
#     # 矩阵：行是真实，列是预测
#     # [[TP, FN],
#     #  [FP, TN]]
#     cm = np.array([[TP, FN],
#                    [FP, TN]], dtype=float)
#
#     if normalize:
#         cm_show = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)  # 每行归一化
#     else:
#         cm_show = cm
#
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.size'] = 16
#     plt.rcParams['axes.linewidth'] = 1.2
#
#     fig, ax = plt.subplots(figsize=(6, 4.8))
#
#     im = ax.imshow(cm_show, interpolation='nearest')  # 不指定颜色，默认即可
#
#     # 坐标轴
#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#     ax.set_xticklabels(['Pred=1', 'Pred=0'])
#     ax.set_yticklabels(['True=1', 'True=0'])
#
#     ax.set_xlabel('Prediction')
#     ax.set_ylabel('Ground Truth')
#
#     if title is not None:
#         ax.set_title(title)
#
#     ax.tick_params(direction='in', length=5, width=1.2)
#
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_linewidth(1.2)
#
#     # 在格子里写数值
#     for i in range(2):
#         for j in range(2):
#             if normalize:
#                 text = f"{cm_show[i, j]*100:.2f}%\n({int(cm[i, j])})"
#             else:
#                 text = f"{int(cm[i, j])}"
#             ax.text(j, i, text, ha='center', va='center')
#
#     fig.tight_layout()
#     fig.savefig(save_path, dpi=600)
#     plt.close(fig)


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix_seaborn(cm, save_path, title=None, normalize=True):
    """
    cm: dict with TP, FP, TN, FN
    normalize=True: 每行归一化（更适合不同SNR比较）
    """
    TP, FP, TN, FN = cm['TP'], cm['FP'], cm['TN'], cm['FN']

    # 行=真实(True)，列=预测(Pred)
    # True=1: [TP, FN]
    # True=0: [FP, TN]
    mat = np.array([[TP, FN],
                    [FP, TN]], dtype=float)

    if normalize:
        mat_show = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)
        fmt = '.2%'
    else:
        mat_show = mat
        fmt = '.0f'

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(6, 4.8))

    sns.heatmap(
        mat_show,
        annot=True,
        fmt=fmt,
        cmap='Greys',      # 更“通信论文味”；你也可以换 'Blues'
        cbar=False,
        linewidths=1,
        linecolor='black',
        ax=ax
    )

    ax.set_xticklabels(['Pred=1', 'Pred=0'])
    ax.set_yticklabels(['True=1', 'True=0'], rotation=0)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground Truth')

    if title is not None:
        ax.set_title(title)

    ax.tick_params(direction='in', length=5, width=1.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=600)
    plt.close(fig)