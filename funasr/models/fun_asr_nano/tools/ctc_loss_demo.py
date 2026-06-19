"""
CTC 损失函数 Mock 数据演示
=========================
用最简单的数据手动走一遍 CTC 损失计算，帮助理解输入输出含义。

运行方式: python funasr/models/fun_asr_nano/tools/ctc_loss_demo.py
"""

import torch
import torch.nn as nn

# ============================================================
# 场景设定
# ============================================================
# 假设我们有一个极简词表: {0: blank, 1: "a", 2: "b"}
vocab_size = 3  # blank + a + b
blank = 0

# 音频有 T=4 帧，目标文本是 "ab"（2个字符）
# 注意: T >= 2 * L - 1 是 CTC 的基本约束（L=目标长度）
#       因为每个字符之间至少要有一个 blank 分隔

# ============================================================
# 构造输入
# ============================================================

# log_probs: 每帧在每个词表 token 上的 log 概率
# 形状: (T, B, C) — 注意 T 在前，这是 CTCLoss 的要求
# B=1 (1个样本), T=4 (4帧), C=3 (blank, a, b)
#
# 我们手动设定每帧的概率分布，让结果可预测：
#   帧0: 主要预测 blank  → log_prob 接近 0（概率接近 1）
#   帧1: 主要预测 a     → log_prob 接近 0
#   帧2: 主要预测 b     → log_prob 接近 0
#   帧3: 主要预测 blank  → log_prob 接近 0
#
# 直觉: 音频的 4 帧依次对应 [静音, "a", "b", 静音]

log_probs = torch.tensor([
    # 帧0: blank 概率最高
    [[-0.1, -5.0, -5.0]],
    # 帧1: "a" 概率最高
    [[-5.0, -0.1, -5.0]],
    # 帧2: "b" 概率最高
    [[-5.0, -5.0, -0.1]],
    # 帧3: blank 概率最高
    [[-0.1, -5.0, -5.0]],
]).log_softmax(dim=2)  # 再过一次 log_softmax 确保数值合法

print("=" * 60)
print("CTC 损失函数 Mock 数据演示")
print("=" * 60)
print(f"\n词表: {{0: blank, 1: 'a', 2: 'b'}}")
print(f"帧数 T=4, 目标文本='ab' (长度=2)")
print(f"\nlog_probs 形状: {log_probs.shape}  (T, B, C)")

# targets: 目标文本的 token ID 序列
# 形状: (B*S,) — 一维的，所有样本的目标拼在一起
targets = torch.tensor([1, 2])  # "ab"

# input_lengths: 每个样本的实际帧数
# 形状: (B,)
input_lengths = torch.tensor([4])  # 4 帧

# target_lengths: 每个样本的目标文本长度
# 形状: (B,)
target_lengths = torch.tensor([2])  # 2 个字符

print(f"\n输入数据:")
print(f"  targets       = {targets.tolist()}  (token ID: 'a'=1, 'b'=2)")
print(f"  input_lengths = {input_lengths.tolist()}  (4帧)")
print(f"  target_lengths= {target_lengths.tolist()}  (2个字符)")

# ============================================================
# 计算 CTC 损失
# ============================================================
ctc_loss = nn.CTCLoss(reduction="none", blank=blank)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

print(f"\n{'=' * 60}")
print(f"CTC 损失计算结果")
print(f"{'=' * 60}")
print(f"  loss = {loss.item():.4f}")
print(f"  这个值越小，说明模型预测越接近目标文本 'ab'")

# ============================================================
# 对比: 如果概率分布完全随机，损失会怎样？
# ============================================================
print(f"\n{'=' * 60}")
print(f"对比: 概率分布完全随机 vs 完美预测")
print(f"{'=' * 60}")

# 随机分布: 每帧每个 token 概率相等
random_probs = torch.zeros(4, 1, 3).log_softmax(dim=2)
random_loss = ctc_loss(random_probs, targets, input_lengths, target_lengths)
print(f"  随机分布 loss = {random_loss.item():.4f}  (每帧概率均等 1/3)")

# 完美预测: 概率更集中
perfect_probs = torch.tensor([
    [[-0.001, -10.0, -10.0]],  # 帧0: blank 概率 ≈ 1
    [[-10.0, -0.001, -10.0]],  # 帧1: a 概率 ≈ 1
    [[-10.0, -10.0, -0.001]],  # 帧2: b 概率 ≈ 1
    [[-0.001, -10.0, -10.0]],  # 帧3: blank 概率 ≈ 1
]).log_softmax(dim=2)
perfect_loss = ctc_loss(perfect_probs, targets, input_lengths, target_lengths)
print(f"  完美预测 loss = {perfect_loss.item():.6f}  (每帧概率 ≈ 1)")

# ============================================================
# 对比: 如果目标文本不同，损失会怎样？
# ============================================================
print(f"\n{'=' * 60}")
print(f"对比: 相同输入，不同目标文本")
print(f"{'=' * 60}")

# 目标 "ba" (与输入分布不匹配)
targets_ba = torch.tensor([2, 1])  # "ba"
loss_ba = ctc_loss(log_probs, targets_ba, input_lengths, target_lengths)
print(f"  目标='ab' loss = {loss.item():.4f}  (匹配)")
print(f"  目标='ba' loss = {loss_ba.item():.4f}  (不匹配)")

# ============================================================
# CTC 解码演示: argmax + 路径折叠
# ============================================================
print(f"\n{'=' * 60}")
print(f"CTC 贪心解码演示")
print(f"{'=' * 60}")

# 用完美预测的概率做解码
raw_output = perfect_probs.argmax(dim=2).squeeze()  # (T,)
print(f"  argmax 结果: {raw_output.tolist()}")
print(f"  对应字符:    {[['blank','a','b'][i] for i in raw_output.tolist()]}")

# 路径折叠: 去重 + 去 blank
def ctc_greedy_decode(token_ids, blank=0):
    """CTC 贪心解码: 先去重，再去 blank"""
    prev = None
    result = []
    for tid in token_ids:
        if tid != prev:          # 去重: 与前一帧相同则跳过
            if tid != blank:     # 去 blank: blank 不输出
                result.append(tid)
        prev = tid
    return result

decoded = ctc_greedy_decode(raw_output.tolist(), blank=0)
decoded_text = "".join([["blank", "a", "b"][i] for i in decoded])
print(f"  折叠后结果:  {decoded}")
print(f"  解码文本:    {decoded_text}")

# ============================================================
# 多样本演示
# ============================================================
print(f"\n{'=' * 60}")
print(f"多样本 CTC 损失演示 (B=2)")
print(f"{'=' * 60}")

# 样本1: "ab", 样本2: "a"
log_probs_2 = torch.tensor([
    # 帧0
    [[-0.1, -5.0, -5.0],   # 样本1: blank
     [-5.0, -0.1, -5.0]],  # 样本2: a
    # 帧1
    [[-5.0, -0.1, -5.0],   # 样本1: a
     [-0.1, -5.0, -5.0]],  # 样本2: blank
    # 帧2
    [[-5.0, -5.0, -0.1],   # 样本1: b
     [-0.1, -5.0, -5.0]],  # 样本2: blank
    # 帧3
    [[-0.1, -5.0, -5.0],   # 样本1: blank
     [-0.1, -5.0, -5.0]],  # 样本2: blank
]).log_softmax(dim=2)

targets_2 = torch.tensor([1, 2, 1])  # 拼接: "ab" + "a"
input_lengths_2 = torch.tensor([4, 4])
target_lengths_2 = torch.tensor([2, 1])

loss_2 = ctc_loss(log_probs_2, targets_2, input_lengths_2, target_lengths_2)
print(f"  样本1 目标='ab', loss = {loss_2[0].item():.4f}")
print(f"  样本2 目标='a',  loss = {loss_2[1].item():.4f}")

print(f"\n{'=' * 60}")
print(f"关键要点总结")
print(f"{'=' * 60}")
print("""
1. CTCLoss 输入形状: log_probs (T,B,C), targets (N,), lengths (B,)
2. targets 是一维的，多个样本的目标拼在一起，用 target_lengths 切分
3. loss 越小 = 模型预测越接近目标文本
4. CTC 解码 = argmax + 去重 + 去 blank
5. T >= 2*L-1 是 CTC 的基本约束（字符间需要 blank 分隔）
""")
