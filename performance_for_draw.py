import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from draw.hot import plot_confusion_matrix_seaborn
from draw.pr import plot_multi_pr_sci
from draw.roc import plot_multi_roc_sci
from models.transceiver import DeepSC, Key_net, Attacker, MAC, CAEM_Fig2_SNR_1D, FeatureMapSelectionModule_SNR_AllC, \
    VerificationDiscriminatorLN
from utlis.tools import SNR_to_noise, SeqtoText, BleuScore
from utlis.trainer_for_next_work import train_step, val_step, train_mi, greedy_decode, initNetParams, \
    greedy_decode_for_draw
from dataset.dataloader import return_iter, return_iter_10, return_iter_eve
from models.transceiver import DeepSC, Key_net, Attacker, MAC, KnowledgeBase, KB_Mapping
from models.mutual_info import Mine
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='./data/vocab.json', type=str)
parser.add_argument('--vocab_path', default='./data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoints/12', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=88, type=int)  # 这里控制的是每次拿(从数据集中读取)多少张牌(个句子)
parser.add_argument('--epochs', default=1, type=int)

parser.add_argument('--encoder-num-layer', default=4, type=int, help='The number of encoder layers')
parser.add_argument('--encoder-d-model', default=128, type=int, help='The output dimension of attention')
parser.add_argument('--encoder-d-ff', default=512, type=int, help='The output dimension of ffn')
parser.add_argument('--encoder-num-heads', default=8, type=int, help='The number heads')
parser.add_argument('--encoder-dropout', default=0.1, type=float, help='The encoder dropout rate')

parser.add_argument('--decoder-num-layer', default=4, type=int, help='The number of decoder layers')
parser.add_argument('--decoder-d-model', default=128, type=int, help='The output dimension of decoder')
parser.add_argument('--decoder-d-ff', default=512, type=int, help='The output dimension of ffn')
parser.add_argument('--decoder-num-heads', default=8, type=int, help='The number heads')
parser.add_argument('--decoder-dropout', default=0.1, type=float, help='The decoder dropout rate')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def update_roc_buffers(logits_pos, logits_eve, logits_perm, scores_list, labels_list):
    """
    logits_pos : alice_verifier(g, channel_dec_g_pp)        -> 正样本 logits
    logits_eve : alice_verifier(g, channel_dec_g_eve_pp)    -> 负样本 logits
    logits_perm: alice_verifier(g, channel_dec_g_pp_perm)   -> 负样本 logits

    scores_list/labels_list: Python list，用来跨 batch 累积
    """

    # BCEWithLogitsLoss -> ROC 必须用 sigmoid 概率（0~1）
    s_pos  = torch.sigmoid(logits_pos).view(-1).detach().cpu()
    s_eve  = torch.sigmoid(logits_eve).view(-1).detach().cpu()
    s_perm = torch.sigmoid(logits_perm).view(-1).detach().cpu()

    # 标签：正样本=1，负样本=0
    y_pos  = torch.ones_like(s_pos)
    y_eve  = torch.zeros_like(s_eve)
    y_perm = torch.zeros_like(s_perm)

    # 拼到 list（注意：这里先 append tensor，最后再 cat）
    scores_list.append(s_pos);  labels_list.append(y_pos)
    scores_list.append(s_eve);  labels_list.append(y_eve)
    scores_list.append(s_perm); labels_list.append(y_perm)


def update_confusion_counts(logits_pos, logits_eve, logits_perm, cm, threshold=0.5):
    # probs
    p_pos  = torch.sigmoid(logits_pos).view(-1)
    p_eve  = torch.sigmoid(logits_eve).view(-1)
    p_perm = torch.sigmoid(logits_perm).view(-1)

    # preds
    pred_pos  = (p_pos  >= threshold).long()
    pred_eve  = (p_eve  >= threshold).long()
    pred_perm = (p_perm >= threshold).long()

    # True=1 for pos
    cm['TP'] += int((pred_pos == 1).sum().item())
    cm['FN'] += int((pred_pos == 0).sum().item())

    # True=0 for neg (eve + perm)
    cm['FP'] += int((pred_eve == 1).sum().item())
    cm['TN'] += int((pred_eve == 0).sum().item())

    cm['FP'] += int((pred_perm == 1).sum().item())
    cm['TN'] += int((pred_perm == 0).sum().item())

# 需要对解码数据进行BLEU分数计算
# 输入是解码结果、原文、以及
def performance(CAEM_with_SNR, fms, alice_verifier, args, SNR, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping):
    test_iterator = return_iter(args, 'test')
    test_iterator_eve = return_iter_eve(args, 'test')
    iter_eve = iter(test_iterator_eve)

    deepsc.eval()
    key_ab.eval()
    alice_bob_mac.eval()
    eve.eval()
    Alice_KB.eval()
    Bob_KB.eval()
    Eve_KB.eval()
    Alice_mapping.eval()
    Bob_mapping.eval()
    Eve_mapping.eval()
    CAEM_with_SNR.eval()
    fms.eval()
    alice_verifier.eval()

    curves = []

    with torch.no_grad():
        for epoch in range(args.epochs):
            for snr in tqdm(SNR):
                noise_std = SNR_to_noise(snr)
                # 每个 SNR 单独收集
                scores_list, labels_list = [], []
                cm = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

                for sents in test_iterator:
                    sents = sents.to(device)
                    try:
                        sents_eve = next(iter_eve).to(device)
                    except:
                        iter_eve = iter(test_iterator_eve)
                        sents_eve = next(iter_eve).to(device)

                    logits, logits_eve, logits_perm = greedy_decode_for_draw(
                        CAEM_with_SNR, fms, alice_verifier, args, deepsc,
                        alice_bob_mac, key_ab, eve,
                        Alice_KB, Bob_KB, Eve_KB,
                        Alice_mapping, Bob_mapping, Eve_mapping,
                        sents, sents_eve,
                        noise_std, args.MAX_LENGTH,
                        pad_idx, start_idx, args.channel
                    )

                    # 把这个 batch 的3组logits加入当前snr的buffer
                    update_roc_buffers(logits, logits_eve, logits_perm,
                                       scores_list, labels_list)

                    update_confusion_counts(logits, logits_eve, logits_perm, cm, threshold=0.5)

                y_score = torch.cat(scores_list).numpy()
                y_true = torch.cat(labels_list).numpy()
                curves.append({'snr': snr, 'y_true': y_true, 'y_score': y_score})

                plot_confusion_matrix_seaborn(
                    cm,
                    save_path=f'cm_snr_{snr}dB.png',
                    title=f'Confusion Matrix (SNR={snr} dB, thr=0.5)',
                    normalize=True
                )

    plot_multi_roc_sci(curves, save_path='roc_multi_snr.png')
    plot_multi_pr_sci(curves, save_path='pr_multi_snr.png')

    # return alice_score, eve_score, perm_score




if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 3, 6]
    args.vocab_file = args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    with open(args.vocab_path, 'r') as f:  # 使用'r'而不是'rb'，因为json.load默认读取文本
        vocab = json.load(f)
    args.vocab_size = len(vocab['token_to_idx'])
    token_to_idx = vocab['token_to_idx']
    args.pad_idx = token_to_idx["<PAD>"]
    args.start_idx = token_to_idx["<START>"]
    args.end_idx = token_to_idx["<END>"]
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']


    StoT = SeqtoText(token_to_idx, args.end_idx)


    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)
    alice_bob_mac = MAC().to(device)

    key_ab = Key_net(args).to(device)
    eve = Attacker().to(device)
    Alice_KB = KnowledgeBase().to(device)
    Bob_KB = KnowledgeBase().to(device)
    Eve_KB = KnowledgeBase().to(device)
    Alice_mapping = KB_Mapping().to(device)
    Bob_mapping = KB_Mapping().to(device)
    Eve_mapping = KB_Mapping().to(device)

    CAEM_with_SNR = CAEM_Fig2_SNR_1D(C_in=31, C_out=8, use_resnet=True).to(device)  # 加入SNR考量的映射 alice和bob共享
    fms = FeatureMapSelectionModule_SNR_AllC(C=8, hidden=64).to(device)  # 对上面的映射进行特征选择 只有bob用 16就是上面的输出通道数 注意这里筛选完也是16通道 只不过有的被置零了
    alice_verifier = VerificationDiscriminatorLN(C=8, L=128, output_logits=True).to(device)  # alice的验证器 128是特征长度

    checkpoint = torch.load(r'/root/autodl-tmp/for_work_12/checkpoints/checkpoint_109.pth')
    # checkpoint_12 = torch.load(r'/root/autodl-tmp/for_work_12/checkpoints/12/2026-01-29-17_55_16/checkpoint_399_0.9968_0.9851.pth')  # 12部分的那三个网络
    checkpoint_12 = torch.load(
        r'/root/autodl-tmp/for_work_12/checkpoints/12/2026-03-03-18_13_51/checkpoint_23_0.8601_0.9153.pth')
    model_state_dict = checkpoint['deepsc']
    alice_bob_mac_state_dict = checkpoint['alice_bob_mac']
    key_state_dict = checkpoint['key_ab']
    eve_state_dict = checkpoint['eve']
    Alice_KB_state_dict = checkpoint['Alice_KB']
    Bob_KB_state_dict = checkpoint['Bob_KB']
    Eve_KB_state_dict = checkpoint['Eve_KB']
    Alice_mapping_state_dict = checkpoint['Alice_mapping']
    Bob_mapping_state_dict = checkpoint['Bob_mapping']
    Eve_mapping_state_dict = checkpoint['Eve_mapping']
    CAEM_with_SNR_state_dict = checkpoint_12['CAEM_with_SNR']
    fms_state_dict = checkpoint_12['fms']
    alice_verifier_state_dict = checkpoint_12['alice_verifier']


    deepsc.load_state_dict(model_state_dict)
    alice_bob_mac.load_state_dict(alice_bob_mac_state_dict)
    key_ab.load_state_dict(key_state_dict)
    eve.load_state_dict(eve_state_dict)
    Alice_KB.load_state_dict(Alice_KB_state_dict)
    Bob_KB.load_state_dict(Bob_KB_state_dict)
    Eve_KB.load_state_dict(Eve_KB_state_dict)
    Alice_mapping.load_state_dict(Alice_mapping_state_dict)
    Bob_mapping.load_state_dict(Bob_mapping_state_dict)
    Eve_mapping.load_state_dict(Eve_mapping_state_dict)
    CAEM_with_SNR.load_state_dict(CAEM_with_SNR_state_dict)
    fms.load_state_dict(fms_state_dict)
    alice_verifier.load_state_dict(alice_verifier_state_dict)

    deepsc = deepsc.to(device)
    alice_bob_mac = alice_bob_mac.to(device)
    key_ab = key_ab.to(device)
    eve = eve.to(device)
    Alice_KB = Alice_KB.to(device)
    Bob_KB = Bob_KB.to(device)
    Eve_KB = Eve_KB.to(device)
    Alice_mapping = Alice_mapping.to(device)
    Bob_mapping = Bob_mapping.to(device)
    Eve_mapping = Eve_mapping.to(device)
    CAEM_with_SNR = CAEM_with_SNR.to(device)
    fms = fms.to(device)
    alice_verifier = alice_verifier.to(device)

    performance(CAEM_with_SNR, fms, alice_verifier, args, SNR, deepsc, alice_bob_mac, key_ab, eve, Alice_KB, Bob_KB, Eve_KB, Alice_mapping, Bob_mapping, Eve_mapping)


