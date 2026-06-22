# WavFrontend 核心知识点

本文档整理理解 `funasr/frontends/wav_frontend.py` 所需提前掌握的核心知识点。

---

## 知识地图

```
WavFrontend 代码理解
├── 1. 数字音频基础
│   ├── 采样率 (Sample Rate)
│   ├── 分帧 (Framing)
│   ├── 帧长 (Frame Length) & 帧移 (Frame Shift)
│   └── 窗函数 (Window Function)
├── 2. 声学特征提取
│   ├── STFT (短时傅里叶变换)
│   ├── Mel 滤波器组 (Mel Filter Bank)
│   ├── FBank 特征
│   └── Kaldi fbank vs Librosa STFT
├── 3. 特征后处理
│   ├── CMVN (倒谱均值方差归一化)
│   ├── LFR (低帧率)
│   ├── Dither (抖动)
│   └── snip_edges
├── 4. 批量处理与 Padding
│   └── pad_sequence
├── 5. 内存视图与高效切片
│   └── as_strided
├── 6. 流式推理机制
│   ├── 缓存 (Cache) 管理
│   ├── 帧数计算
│   └── 在线 LFR 拼接
└── 7. 三个类的设计对比
    ├── WavFrontend (离线)
    ├── WavFrontendOnline (流式)
    └── WavFrontendMel23 (23维Mel)
```

---

## 一、数字音频基础

### 1.1 采样率 (Sample Rate, fs)

**定义**：每秒对模拟音频信号的采样次数，单位 Hz。

```
原始模拟声波（连续）
      ↓ 采样（每秒 fs 次）
离散数字信号（数组）
```

| 采样率 | 典型用途 |
|--------|---------|
| 8,000 Hz | 电话语音 |
| 16,000 Hz | FunASR 默认，ASR 常用 |
| 44,100 Hz | CD 音质 |
| 48,000 Hz | 专业音频 |

**在代码中**：`fs=16000`，即 1 秒音频 = 16000 个采样点。
```python
self.fs = fs  # 16000
self.frame_sample_length = int(self.frame_length * self.fs / 1000)  # 25ms → 400 samples
```

### 1.2 分帧 (Framing)

语音信号是非平稳信号，但**短时**（10-30ms）内可视为平稳。因此将长音频切成若干短片段（帧）进行分析。

```
长音频:  [s₁, s₂, s₃, ..., sₙ]  (n 个采样点)
           ↓ 分帧
帧 1:  [s₁, s₂, ..., s_{frame_len}]
帧 2:  [s_{1+shift}, s_{2+shift}, ..., s_{frame_len+shift}]
帧 3:  [s_{1+2*shift}, ...]
...
```

### 1.3 帧长 (Frame Length) & 帧移 (Frame Shift)

| 参数 | 含义 | 默认值 | 对应采样点数 |
|------|------|--------|-------------|
| `frame_length` | 每帧的时间长度 | 25 ms | 25 × 16 = 400 点 |
| `frame_shift` | 相邻帧之间的时间偏移 | 10 ms | 10 × 16 = 160 点 |

```
时间轴:  0    10   20   25   35   45   50  (ms)
         ├────────┤ 帧1 (25ms)
              ├────────┤ 帧2 (25ms)
                   ├────────┤ 帧3 (25ms)
                        ├────────┤ 帧4 (25ms)
         ↑    ↑
       帧移=10ms (相邻帧重叠15ms)
```

**代码对应**：
```python
self.frame_length = frame_length       # 25 (ms)
self.frame_shift = frame_shift         # 10 (ms)
self.frame_sample_length = int(self.frame_length * self.fs / 1000)      # 400 samples
self.frame_shift_sample_length = int(self.frame_shift * self.fs / 1000) # 160 samples
```

### 1.4 窗函数 (Window Function)

**目的**：直接截断信号会产生频谱泄漏（Gibbs 效应），加窗可以平滑边界，减少泄漏。

```
原始帧:  [s₁, s₂, ..., sₙ]
           ↓ 乘以窗函数
加窗后:  [s₁×w₁, s₂×w₂, ..., sₙ×wₙ]
```

| 窗类型 | 特点 |
|--------|------|
| Hamming（默认） | 平衡主瓣宽度和旁瓣抑制 |
| Hann | 旁瓣衰减更大 |
| Rectangular | 等同于不加窗 |

**代码中**：
```python
self.window = window  # "hamming"
# 传给 kaldi.fbank → window_type=self.window
```

---

## 二、声学特征提取

### 2.1 FBank 特征（核心）

FBank = **Filter Bank**（滤波器组特征），是 ASR 最常用的声学特征之一。

完整的 FBank 提取流程：

```
原始音频波形 [N samples]
    ↓ ① 分帧（25ms 帧长，10ms 帧移）
各帧信号
    ↓ ② 加窗（Hamming）
加窗帧
    ↓ ③ FFT（短时傅里叶变换）
频谱（功率谱）
    ↓ ④ 过 Mel 滤波器组（如 80 个三角滤波器）
Mel 频谱  [T, n_mels]
    ↓ ⑤ 取 log
FBank 特征 [T, n_mels]
```

**关键概念**：

| 步骤 | 输入 | 输出 | 变换 |
|------|------|------|------|
| 分帧 | 波形 [N] | 帧 [T, L] | 滑动窗口 |
| 加窗 | 帧 [T, L] | 加窗帧 [T, L] | × Hamming 窗 |
| FFT | 加窗帧 [T, L] | 复数谱 [T, F] | 时→频域 |
| 功率谱 | 复数谱 [T, F] | 功率谱 [T, F] | \|·\|² |
| Mel 滤波 | 功率谱 [T, F] | Mel 能量 [T, M] | 矩阵乘法 |
| log | Mel 能量 [T, M] | **FBank [T, M]** | log10 |

**为什么用 Mel 尺度？**
人耳对音高的感知是非线性的——低频区分辨率高，高频区分辨率低。Mel 尺度模拟了这种感知特性。

```
线性频率 → Mel 频率:
mel = 2595 × log10(1 + f/700)
```

### 2.2 Kaldi fbank（WavFrontend / WavFrontendOnline 使用）

```python
mat = kaldi.fbank(
    waveform,                              # [1, N] 音频张量
    num_mel_bins=self.n_mels,              # Mel 滤波器数量，默认 80
    frame_length=self.frame_length,        # 帧长 25ms
    frame_shift=self.frame_shift,          # 帧移 10ms
    dither=self.dither,                    # 抖动系数，默认 1.0
    energy_floor=0.0,                      # 能量下限
    window_type=self.window,               # 窗函数 "hamming"
    sample_frequency=self.fs,              # 采样率 16000
    snip_edges=self.snip_edges,            # 是否截断边缘
)
# 输出: [T, 80] 的 FBank 特征矩阵
```

**Kaldi fbank 的特点**：
- C++ 实现，性能高，与 Kaldi 工具链兼容
- 自动完成分帧→加窗→FFT→Mel→log 全流程
- `dither` 参数：在信号中添加微小噪声，防止 log(0)

### 2.3 Librosa STFT + 自定义 Mel（WavFrontendMel23 使用）

`WavFrontendMel23` 不使用 Kaldi，而是调用 `eend_ola_feature.py` 中的函数：

```python
# 步骤 1: STFT
mat = eend_ola_feature.stft(waveform, frame_length=25, frame_shift=10)
# → 复数谱 [T, F]

# 步骤 2: 转 Mel + log + 均值减法
mat = eend_ola_feature.transform(mat)
# → |STFT|^2 → Mel 滤波 (23 bins) → log10 → 减均值

# 步骤 3: 上下文拼接
mat = eend_ola_feature.splice(mat, context_size=self.lfr_m)
# → [T, 23 * (2*lfr_m+1)]

# 步骤 4: 降采样
mat = mat[::self.lfr_n]
# → [T/lfr_n, 23 * (2*lfr_m+1)]
```

**关键差异**：
| | Kaldi fbank | eend_ola_feature |
|---|---|---|
| Mel 数量 | 80 | 23 |
| 输出维度 | `80 * lfr_m` | `23 * (2*lfr_m + 1)` |
| 均值减法 | 外部 CMVN | 内置在 transform 中 |
| 实现 | C++ (libtorchaudio) | Python (librosa) |

---

## 三、特征后处理

### 3.1 CMVN — 倒谱均值方差归一化

> 详细学习笔记见：[CMS归一化学习笔记.md](./CMS归一化学习笔记.md)

**CMVN = Cepstral Mean and Variance Normalization**

```
CMVN(x) = (x - mean) / std
```

| 步骤 | 操作 | 效果 |
|------|------|------|
| 减均值 | `x - mean` | 消除信道效应（常量偏移） |
| 除标准差 | `x / std` | 统一特征尺度 |

**代码实现**（`apply_cmvn` / `WavFrontendOnline.apply_cmvn`）：

```python
def apply_cmvn(inputs, cmvn):
    means = cmvn[0:1, :dim]    # 均值向量
    vars = cmvn[1:2, :dim]     # 标准差向量（Kaldi 格式称为 vars）
    inputs += means.to(device)  # 加均值（Kaldi 中 mean 存的是 -原始均值）
    inputs *= vars.to(device)   # 乘标准差倒数
    return inputs
```

**CMVN 文件格式**（Kaldi 风格）：
```
<AddShift>
<LearnRateCoef> 0 0 -0.123 0.456 ... 0.789 ]
<Rescale>
<LearnRateCoef> 0 0 0.023 0.045 ... 0.012 ]
```

`load_cmvn` 函数解析这个文件，提取 means 和 vars。

### 3.2 LFR — 低帧率 (Low Frame Rate)

**目的**：将帧率从 100fps（帧移 10ms）降为 50fps 甚至更低，减少序列长度，降低后续模型（如 Transformer）的计算量。

**核心思想**：把相邻帧的特征拼接起来，然后跳帧取。

```
原始特征序列（100fps）：
t=0: [f₀]
t=1: [f₁]
t=2: [f₂]
t=3: [f₃]
t=4: [f₄]
...

LFR(m=3, n=2) 后的新帧：
t'=0: [f₋₁, f₀, f₁]   ← 拼接3帧（左右各1帧上下文）
t'=1: [f₁, f₂, f₃]     ← 跳2帧
t'=2: [f₃, f₄, f₅]
...
```

**参数含义**：
| 参数 | 含义 | 效果 |
|------|------|------|
| `lfr_m` | 拼接帧数 | 输出维度 = `n_mels × lfr_m` |
| `lfr_n` | 跳帧步长 | 帧率降为原来的 1/lfr_n |

**as_strided 实现**（高效，无需复制数据）：

```python
def apply_lfr(inputs, lfr_m, lfr_n):
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))           # 新帧数
    feat_dim = inputs.shape[-1]
    strides = (lfr_n * feat_dim, 1)            # 行步长, 列步长
    sizes = (T_lfr, lfr_m * feat_dim)          # 新形状
    LFR_outputs = inputs.as_strided(sizes, strides)
    return LFR_outputs
```

**as_strided 可视化**：

```
原始矩阵 (T=5, D=3):
[[a₀, a₁, a₂],
 [b₀, b₁, b₂],
 [c₀, c₁, c₂],
 [d₀, d₁, d₂],
 [e₀, e₁, e₂]]

as_strided(sizes=(3, 6), strides=(6, 1)):  # lfr_m=2, lfr_n=2
[[a₀, a₁, a₂, b₀, b₁, b₂],    ← 帧0:a, 帧1:b
 [c₀, c₁, c₂, d₀, d₁, d₂],    ← 帧2:c, 帧3:d
 [e₀, e₁, e₂, ...              ← 帧4:e（可能越界，需要 padding）
```

### 3.3 Dither（抖动）

**定义**：在信号中添加微小的高斯噪声。

**目的**：
- 防止对完全静音的帧取 log(0)
- 某些情况下可以小幅改善识别精度

**代码中**：
```python
self.dither = dither  # 默认 1.0
# 传给 kaldi.fbank，Kaldi 内部处理
```

### 3.4 snip_edges

**定义**：控制如何处理音频边缘不足以构成完整帧的剩余采样点。

| snip_edges | 行为 |
|------------|------|
| `True`（默认） | 直接丢弃边缘不足一帧的采样点 |
| `False` | 保留所有帧，最后不足一帧的用零填充 |

---

## 四、批量处理与 Padding

### 4.1 pad_sequence

**问题**：一个 batch 中的音频长度不同，提取出的帧数也不同。但神经网络需要规整的张量。

**解决**：用 `pad_sequence` 将短序列补零到最长序列的长度。

```python
from torch.nn.utils.rnn import pad_sequence

feats = [tensor_1(T₁, D), tensor_2(T₂, D), tensor_3(T₃, D)]
# T₁ < T₂ < T₃

feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
# → [3, T₃, D]
# tensor_1: 后 (T₃-T₁) 行全 0
# tensor_2: 后 (T₃-T₂) 行全 0
```

**在 WavFrontend.forward() 中**：
```python
feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)
# 输出 shape: [batch_size, max_T, n_mels * lfr_m]
```

---

## 五、内存视图 as_strided

### 5.1 原理

`as_strided` 是 PyTorch/NumPy 中的高级操作，它**不复制数据**，只改变对原数据的"观察方式"（步长和形状）。

```
as_strided(sizes, strides)

sizes:  新张量的形状
strides: 在新张量中，每移动一个元素位置，在原数据中跳多少字节
```

### 5.2 LFR 中的应用

```python
# inputs: [T, D] 的 FBank 特征
strides = (lfr_n * D, 1)   # 行间跳 lfr_n 帧，列间跳 1 个元素
sizes = (T_lfr, lfr_m * D) # 输出帧数, 每帧 lfr_m×D 维
output = inputs.as_strided(sizes, strides)
```

**直观理解**：新矩阵的第 i 行 = 原矩阵中从第 `i × lfr_n` 行开始的 `lfr_m` 行的拼接视图。

### 5.3 splice 中的应用（WavFrontendMel23）

```python
# Y_pad: 已在上下各 pad 了 context_size 行
Y_spliced = np.lib.stride_tricks.as_strided(
    Y_pad,
    (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),  # 每行拼接周围帧
    (Y.itemsize * Y.shape[1], Y.itemsize),
)
```

---

## 六、流式推理机制（WavFrontendOnline）

### 6.1 核心挑战

离线场景：整个音频一次输入，提取全部特征。
流式场景：音频逐块到达，需要**增量式**提取特征，并保持与离线一致的结果。

### 6.2 Cache 结构

```python
cache = {
    "input_cache":        torch.Tensor,   # 未处理的残留波形
    "reserve_waveforms":  torch.Tensor,   # 保留的已处理波形（用于帧对齐）
    "waveforms":          torch.Tensor,   # 当前处理的所有波形
    "lfr_splice_cache":   List[Tensor],   # LFR 拼接所需的上下文缓存
    "fbanks":             torch.Tensor,   # 当前 FBank 特征
    "fbanks_lens":        torch.Tensor,   # FBank 帧数
}
```

### 6.3 流式处理流程

```
音频块到达
    ↓
forward() 
    ├── ① forward_fbank(): 波形 → FBank
    │       波形拼接 input_cache + new_chunk
    │       计算能提取的完整帧数
    │       提取 FBank
    │       缓存未处理的残留波形
    │
    ├── ② 检查是否有足够帧做 LFR
    │       帧数 >= lfr_m? 
    │       ├── 是 → 执行 LFR + CMVN
    │       └── 否 → 缓存帧，等待下一块
    │
    └── ③ 输出特征
            更新 reserve_waveforms（帧对齐）
            返回 feats, feats_lengths
```

### 6.4 帧数计算

```python
@staticmethod
def compute_frame_num(sample_length, frame_sample_length, frame_shift_sample_length):
    frame_num = int((sample_length - frame_sample_length) / frame_shift_sample_length + 1)
    return frame_num if frame_num >= 1 and sample_length >= frame_sample_length else 0
```

公式：
```
帧数 = (音频长度 - 帧长) / 帧移 + 1
```

例如：16000 采样点（1秒），帧长 400，帧移 160：
```
帧数 = (16000 - 400) / 160 + 1 = 97 + 1 = 98.5 → 98 帧
```

### 6.5 在线 LFR

在线 LFR 与离线 LFR 不同：
- **离线**：已知全部帧，直接 as_strided
- **在线**：每块可能凑不够 `lfr_m` 帧，需要缓存前面的帧

```python
def apply_lfr(inputs, lfr_m, lfr_n, is_final=False):
    # 关键逻辑：
    if is_final:
        # 最后一块：补齐 padding，输出所有帧
        inputs = torch.vstack([inputs] + [inputs[-1:]] * num_padding)
    else:
        # 中间块：只有完整凑齐 lfr_m 帧的才输出
        sizes = (last_idx, lfr_m * feat_dim)  # 缩小输出尺寸
    # ...
    return LFR_outputs, lfr_splice_cache, splice_idx
```

### 6.6 缓存拼接

```python
# 第一次：初始化 lfr_splice_cache
cache["lfr_splice_cache"].append(
    feats[i][0, :].unsqueeze(0).repeat((self.lfr_m - 1) // 2, 1)
)
# 把第一帧复制 (lfr_m-1)//2 次作为左侧上下文

# 后续：拼接缓存 + 新帧
lfr_splice_cache_tensor = torch.stack(cache["lfr_splice_cache"])
feats = torch.cat((lfr_splice_cache_tensor, feats), dim=1)
```

---

## 七、三个类的设计对比

### 7.1 WavFrontend（离线）

```
forward()
  │
  ├── for each utterance:
  │     ├── kaldi.fbank → [T, 80]
  │     ├── apply_lfr (optional)
  │     └── apply_cmvn (optional)
  │
  └── pad_sequence → [B, T_max, D]
```

**特点**：
- 支持 batch > 1
- 一次性处理完整音频
- 使用 Kaldi fbank
- LFR/CMVN 可选

### 7.2 WavFrontendOnline（流式）

```
forward()
  │
  ├── init_cache (首次)
  ├── forward_fbank → [1, T, 80]
  │     │  处理缓存中的波形 + 新波形
  │     └  缓存残留波形和 FBank
  ├── 拼接 lfr_splice_cache
  ├── forward_lfr_cmvn → [1, T', D]
  └── 更新 reserve_waveforms（帧对齐）
```

**特点**：
- **batch_size 必须 = 1**
- 通过 `cache` dict 维护跨调用的状态
- 输出波形与特征帧对齐，方便后续 VAD 等任务
- `is_final` 标志处理最后一块

### 7.3 WavFrontendMel23（专用离线）

```
forward()
  │
  └── for each utterance:
        ├── eend_ola_feature.stft → [T, F]
        ├── eend_ola_feature.transform → [T, 23]
        │     (|STFT|^2 → Mel 23 → log10 → 减均值)
        └── eend_ola_feature.splice(context_size=lfr_m) → [T, 23*(2*lfr_m+1)]
```

**特点**：
- 23 维 Mel（而非 80 维）
- 使用 Librosa STFT + 自定义处理
- 内置均值减法（在 transform 中）
- output_size = `23 * (2*lfr_m + 1)`（与 WavFrontend 的 `80 * lfr_m` 不同）

### 7.4 对比总结表

| 维度 | WavFrontend | WavFrontendOnline | WavFrontendMel23 |
|------|------------|-------------------|-----------------|
| 模式 | 离线 | 流式 | 离线 |
| 特征提取 | Kaldi fbank | Kaldi fbank | Librosa STFT |
| Mel 维度 | 80 | 80 | 23 |
| Batch 支持 | ✅ | ❌ (必须=1) | ✅ |
| CMVN | 可选（文件） | 可选（文件） | 内置均值减法 |
| LFR | as_strided | 在线 splice | splice + step |
| output_size | `80 * lfr_m` | `80 * lfr_m` | `23 * (2*lfr_m+1)` |
| 帧对齐 | ❌ | ✅ | ❌ |
| 状态管理 | 无状态 | cache dict | 无状态 |

---

## 八、数据流总览

### 8.1 WavFrontend 完整数据流

```
输入: input [B, T_audio]  +  input_lengths [B]
          │
          ▼
    ┌─────────────────────────┐
    │  for each batch item:   │
    │  waveform [T_i]         │
    │    ↓                    │
    │  * (1 << 15)  (放大)    │  可选的整数化处理
    │    ↓                    │
    │  kaldi.fbank()          │  → [T_fbank, 80]
    │    ↓                    │
    │  apply_lfr() (可选)     │  → [T_lfr, 80*lfr_m]
    │    ↓                    │
    │  apply_cmvn() (可选)    │  → [T_lfr, 80*lfr_m]
    └─────────────────────────┘
          │
          ▼
    pad_sequence()            → [B, T_max, D]
          │
          ▼
    输出: feats_pad [B, T_max, D], feats_lens [B]
```

### 8.2 WavFrontendOnline 流式数据流

```
第 1 块到达:
  input_cache: [] + chunk1 → FBank(T₁帧) → LFR 可能不够 → 缓存到 lfr_splice_cache
  
第 2 块到达:
  input_cache: [残留] + chunk2 → FBank(T₂帧)
  拼接: lfr_splice_cache + new_fbank
  LFR → 输出若干帧
  更新 reserve_waveforms（记录用到了哪些波形样点）

...

最后一块 (is_final=True):
  所有缓存帧 → LFR (padding 补齐) → 全部输出
```

### 8.3 WavFrontendMel23 数据流

```
输入: waveform [T_audio]
          │
          ▼
    eend_ola_feature.stft()
          │  → 复数谱 [T_stft, F]
          ▼
    eend_ola_feature.transform()
          │  → |·|² → Mel(23) → log10 → 减均值
          │  → [T_stft, 23]
          ▼
    eend_ola_feature.splice(context_size=lfr_m)
          │  → 拼接上下文
          │  → [T_stft, 23*(2*lfr_m+1)]
          ▼
    [::lfr_n] 降采样
          │  → [T_stft/lfr_n, 23*(2*lfr_m+1)]
          ▼
    pad_sequence → [B, T_max, D]
```

---

## 九、关键概念速查表

| 概念 | 一句话解释 | 在代码中的位置 |
|------|-----------|---------------|
| **FBank** | 模拟人耳听觉的声学特征，ASR 标配 | `kaldi.fbank()` 调用 |
| **STFT** | 将时域波形转为时频谱 | `eend_ola_feature.stft()` |
| **Mel 尺度** | 非线性频率尺度，模拟人耳感知 | `n_mels=80` 参数 |
| **CMVN** | 减均值除标准差，消除信道差异 | `load_cmvn()` → `apply_cmvn()` |
| **LFR** | 帧拼接+跳帧，降低帧率 | `apply_lfr()` 函数 |
| **as_strided** | 零拷贝改变张量视图 | `apply_lfr()` 和 `splice()` 内部 |
| **pad_sequence** | 变长序列补零到同长 | `forward()` 末尾 |
| **dither** | 加微小噪声，防 log(0) | `dither=1.0` 参数 |
| **snip_edges** | 控制边缘帧处理方式 | `snip_edges=True` 参数 |
| **Streaming Cache** | 流式推理的状态保存 | `cache` dict |
| **帧对齐** | 特征帧与原始波形帧的对应关系 | `reserve_waveforms` |
| **upsacle_samples** | 音频值放大 2^15 倍 | `waveform * (1 << 15)` |

---

## 十、延伸阅读

- **CMS 归一化** → [CMS归一化学习笔记.md](./CMS归一化学习笔记.md)
- **傅里叶变换** → [彻底理解傅里叶变换.md](./彻底理解傅里叶变换.md)
- **Kaldi 官方文档** → https://kaldi-asr.org/doc/feat.html
- **Librosa 文档** → https://librosa.org/doc/latest/
- **PyTorch as_strided** → https://pytorch.org/docs/stable/generated/torch.Tensor.as_strided.html