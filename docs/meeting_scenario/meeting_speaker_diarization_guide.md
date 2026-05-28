# 会议室场景声纹识别与会议日志方案

基于 FunASR 构建会议室场景的说话人日志（Speaker Diarization）和结构化会议记录。

---

## 一、核心流水线

FunASR 通过 `AutoModel` 将 VAD → ASR → 声纹提取 → 聚类 四个环节串联，**一次调用**完成：

```
原始录音
  └─ FSMN-VAD（切分说话段）
        └─ ASR 模型（语音转写）
              └─ CAM++（提取 192 维声纹 embedding）
                    └─ 聚类（SB 谱聚类 / HDBSCAN）
                          └─ 按标点/时间将说话人标签分配给每个句子
```

### 基础代码（开箱即用）

```python
from funasr import AutoModel

# Step 1: 加载模型（VAD + 转写 + 声纹 三合一）
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    spk_model="cam++",
    device="cuda",
)

# Step 2: 一次调用，输出带说话人标签的转写结果
result = model.generate(input="meeting_2h.wav")

# Step 3: 结果在 result[0]["sentence_info"] 中
for sent in result[0]["sentence_info"]:
    # {"text": "...", "start": 1200, "end": 2500, "spk": 0, "timestamp": [...]}
    ts_s = sent["start"] / 1000.0
    ts_e = sent["end"] / 1000.0
    print(f"[{ts_s:6.1f}s → {ts_e:6.1f}s] 说话人{sent['spk']}: {sent['text']}")
```

---

## 二、模型选择策略

| 场景 | ASR 模型 ID | 特点 |
|------|------------|------|
| **中文会议** | `"paraformer-zh"` + `punc_model="ct-punc"` | 中文最优，需单独加载标点模型 |
| **多语种会议** | `"iic/SenseVoiceSmall"` | 中/英/日/韩/粤，自带标点和情感识别，234M 参数 |
| **高精度/方言/歌词** | `"FunAudioLLM/Fun-ASR-Nano-2512"` | LLM 架构，31 语言，`hub="hf"`，输出自带标点 |
| **Whisper 用户迁移** | `"whisper-large-v3"` 等 | 多语种，支持翻译，通过 `hub="hf"` 加载 |

**推荐起步用 SenseVoiceSmall**：参数小（234M）、自带标点、自带情感识别、CPU 可跑（17x 实时率，比 Whisper 在 GPU 上还快）。

### VAD 模型

- ID: `"fsmn-vad"` — 轻量级（0.4M 参数），实时流式 VAD
- 关键参数见下节调优

### 声纹模型

- ID: `"cam++"` — CAM++ 架构，7.2M 参数，输出 192 维 speaker embedding
- 推理链路：80 维 fbank → FCM 前端 → 多层 CAMDenseTDNNBlock → StatsPool → 192 维输出

---

## 三、关键调优参数

### 3.1 VAD 参数（决定"切多长一段给 ASR"）

VAD 切分粒度直接决定声纹识别的准确性：段太长则说话人混叠概率增大，段太短则 embedding 不准确（缺乏上下文）。

```python
result = model.generate(
    input="meeting.wav",
    vad_kwargs={
        "max_single_segment_time": 15000,   # 单段最长 15s（默认 60s），会议室建议 10-20s
        "max_end_silence_time": 400,        # 句尾静音容忍 400ms（默认 800ms），更激进切分
        "speech_noise_thres": 0.8,          # 语音/噪声阈值（默认 0.6），嘈杂环境可提高到 0.7-0.8
        "speech_2_noise_ratio": 1.5,        # 信噪比权重（默认 1.0），提高可减少噪声误触发
    },
)
```

**调参要点：**

| 参数 | 默认值 | 建议值 | 说明 |
|------|--------|--------|------|
| `max_single_segment_time` | 60000ms | 10000-20000ms | 会议室多人轮流发言，缩短可减少段内多人语音 |
| `max_end_silence_time` | 800ms | 300-500ms | 降低可更及时地切分，避免两句话被合并 |
| `speech_noise_thres` | 0.6 | 0.7-0.8 | 嘈杂环境提高阈值，减少噪声误触发 |
| `speech_2_noise_ratio` | 1.0 | 1.0-1.5 | 信噪比权重 |

合并策略配合：

```python
result = model.generate(
    input="meeting.wav",
    merge_vad=True,
    merge_length_s=5,       # 短于 5s 的段合并到相邻段
)
```

### 3.2 声纹聚类参数

```python
result = model.generate(
    input="meeting.wav",
    spk_model="cam++",
    spk_mode="punc_segment",    # 按标点分句后再分配说话人（推荐，更细粒度）
    # spk_mode="vad_segment",   # 或按 VAD 段整段分配（长段场景）
)
```

聚类后端自动选择：

- embedding 数量 < 2048 → **谱聚类**（SpectralCluster，来自 `speechbrain`）
- embedding 数量 ≥ 2048 → **UMAP 降维 + HDBSCAN 聚类**

### 3.3 ASR 参数（影响识别准确性）

```python
result = model.generate(
    input="meeting.wav",
    # 热词增强 — 会议领域术语（权重可调，数字越大越偏好）
    hotword="Q3 OKR 复盘 里程碑 排期 对齐 人力 20",
    # 批处理加速，VAD 段较多时加大 batch 提升吞吐
    batch_size=64,
    batch_size_s=300,           # 按秒数动态组 batch
    # SenseVoice 专用
    language="zh",              # 固定语言提升精度，避免多语种检测开销
    use_itn=True,               # 逆文本正则化（数字/日期标准化）
    # 长音频的 AED 合并（仅 SenseVoice 系），提升 Decoder 准确率
    merge_vad=True,
    merge_length_s=15,
)
```

### 3.4 推理加速选项

```bash
# vLLM 加速（仅 Fun-ASR-Nano 可用，2-3x LLM 解码加速）
pip install vllm
```

```python
model = AutoModel(
    model="FunAudioLLM/Fun-ASR-Nano-2512",
    hub="hf",
    trust_remote_code=True,
    device="cuda",
    ncpu=4,         # CPU 线程数
)
```

### 3.5 部署为 API 服务

```bash
# OpenAI 兼容 API
pip install vllm fastapi uvicorn python-multipart
funasr-server --device cuda --model sensevoice

# 验证
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

可接入 LangChain / Dify / AutoGen 等 Agent 框架，或通过 `examples/mcp_server/` 直接让 Claude/Cursor 调用。

---

## 四、会议日志生成

### 4.1 基本日志输出

```python
from collections import defaultdict

result = model.generate(input="meeting.wav")
sentence_info = result[0]["sentence_info"]

# 按说话人统计发言时长
spk_stats = defaultdict(float)
for sent in sentence_info:
    duration = (sent["end"] - sent["start"]) / 1000.0
    spk_stats[sent["spk"]] += duration

print("=== 发言统计 ===")
for spk_id, total_sec in sorted(spk_stats.items()):
    print(f"说话人 {spk_id}: {total_sec:.1f}秒 ({total_sec/60:.1f}分钟)")

# 完整时间轴会议记录
print("\n=== 会议记录 ===")
for sent in sentence_info:
    ts_s = sent["start"] / 1000.0
    ts_e = sent["end"] / 1000.0
    print(f"[{ts_s:6.1f}s → {ts_e:6.1f}s] 说话人{sent['spk']}: {sent['text']}")
```

### 4.2 输出格式整合

| 输出形式 | 对应文件 |
|----------|---------|
| **TXT/时间轴日志** | 上述 Python 代码 |
| **SRT/VTT 字幕** | `examples/subtitle/generate_subtitle.py`，支持 `--spk` 参数 |
| **OpenAI API JSON** | `funasr-server` + `POST /v1/audio/transcriptions` |
| **Agent 工具** | `examples/mcp_server/funasr_mcp.py` |

### 4.3 情感识别附加（可选）

SenseVoice 自带情感/音频事件检测，可关联到每句话：

```python
model = AutoModel(model="iic/SenseVoiceSmall", device="cuda")
result = model.generate(input="meeting.wav")
# 输出格式："<|HAPPY|>你说的对<|Speech|>" — 可用 rich_transcription_postprocess 整理
```

---

## 五、精度调优 Checklist

按影响从大到小排列：

| 优先级 | 调优项 | 说明 |
|--------|--------|------|
| **P0** | VAD 阈值 | 现场噪声大时提高 `speech_noise_thres`（0.7-0.8），降低 `max_end_silence_time`（300-500ms） |
| **P0** | 段长度 | `max_single_segment_time` 控制在 10-20s，避免一段含多个说话人 |
| **P1** | 音频质量 | 16kHz 单声道，近讲麦克风/阵列效果远好于远讲；避免混响严重的大会议室 |
| **P1** | 热词 | 会议术语加入 `hotword`，权重可调（末尾数字：`"术语 20"`） |
| **P1** | 语种固定 | SenseVoice 设 `language="zh"` 避免多语种检测的开销和误判 |
| **P2** | 声纹注册微调 | 已知参会人时，用注册音频 few-shot 微调 CAM++ |
| **P2** | ASR 微调 | 使用 `funasr-train` + Hydra config 做 domain adaptation |
| **P3** | 模型升级 | SenseVoiceSmall(234M) → Fun-ASR-Nano(800M)，精度更高但需要 GPU |

---

## 六、常见问题与排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| **说话人编号混乱** | VAD 段太长，一段内含多人切换 | 降低 `max_single_segment_time`（10-15s） |
| **漏识别（部分语音未被转写）** | VAD 阈值太高，将低声量语音判为噪声 | 降低 `speech_noise_thres`（0.5-0.6） |
| **一句话被拆成多句** | VAD 过于激进 + `max_end_silence_time` 太短 | 增大 `max_single_segment_time`，或设 `merge_vad=True, merge_length_s=10` |
| **声纹聚类人数不准** | 自动聚类无先验人数 | 减少 embedding 噪声（调 VAD），或在长音频场景用 HDBSCAN 的参数调优 |
| **GPU 显存不足** | batch_size 太大 | 降低 `batch_size`（如 16），或切分音频分段处理 |
| **远讲麦克风效果差** | 远讲信噪比低、混响 | 换用近讲麦克风，或在 ASR 前端加语音增强模块 |

---

## 七、实时流式场景

对于需要**实时**会议转写 + 声纹识别的场景（如在线会议），使用 WebSocket 流式方案：

```bash
# 流式服务端
python examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py
```

流式模式下声纹模型换用 `eres2netv2_sv`，支持逐 chunk 提取 embedding。

---

## 参考文件索引

| 功能 | 文件路径 |
|------|---------|
| AutoModel 入口（VAD+ASR+SPK 流水线） | `funasr/auto/auto_model.py` |
| 声纹后处理与聚类 | `funasr/utils/speaker_utils.py` |
| CAM++ 声纹模型定义 | `funasr/models/campplus/model.py` |
| FSMN-VAD 模型及参数 | `funasr/models/fsmn_vad_streaming/model.py` |
| 声纹 Demo（Fun-ASR-Nano） | `examples/industrial_data_pretraining/fun_asr_nano/demo_spk.py` |
| 声纹 Demo（SenseVoice） | `examples/industrial_data_pretraining/sense_voice/demo_spk.py` |
| 字幕生成（支持 --spk） | `examples/subtitle/generate_subtitle.py` |
| 流式实时 WebSocket 服务 | `examples/industrial_data_pretraining/fun_asr_nano/serve_realtime_ws.py` |
| OpenAI 兼容 API 服务 | `funasr/bin/server.py` |
| MCP Server（Claude/Cursor） | `examples/mcp_server/funasr_mcp.py` |
| 模型选择指南 | `docs/model_selection.md` |
| 部署矩阵文档 | `docs/deployment_matrix.md` |
