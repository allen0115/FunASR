# 读懂 `inference_vllm.py` 需要掌握的关键概念

> 目标文件：`funasr/models/fun_asr_nano/inference_vllm.py`
> 模块定位：Fun-ASR-Nano 的 **vLLM 推理引擎**。它把"PyTorch 跑音频编码 + vLLM 跑 LLM 解码"拼在一起，实现高吞吐语音识别。

这个文件不是一个普通工具脚本，它横跨了**音频处理、大模型推理引擎、多模态对齐**三个领域。按难度从浅到深，理解它需要先掌握以下概念。

---

## 一、整体架构：要先把这张图装进脑子

读任何具体函数前，先理解这个文件想表达的**数据流**，否则后面所有细节都会散乱。

```
音频 wav
  │
  ├─→ WavFrontend ──→ Fbank 声学特征          (PyTorch)
  │                      │
  │              SenseVoiceEncoder ──→ 编码向量  (PyTorch)
  │                      │
  │              AudioAdaptor ──→ 音频嵌入      (PyTorch, 降维+降帧率)
  │
  ├─→ 文本 prompt (ChatML 格式) ──→ LLM embed_tokens ──→ 文本嵌入
  │
  └─→ [prefix_text_emb | audio_emb | suffix_text_emb]  ← 拼成一条序列
                          │
                          ▼
                   vLLM (Qwen3-0.6B)  ←  用 EmbedsPrompt 直接喂嵌入
                          │
                          ▼
                      生成文本
```

**关键设计点**（这是整个文件最反直觉的地方，先记住再看代码）：
- vLLM 通常吃 **token id**，而这里喂的是**已经算好的 embedding 向量**（`EmbedsPrompt`）。
- 音频部分没法 tokenize，所以提前在 PyTorch 侧把音频变成 embedding，和文本 embedding 拼起来，**绕过 vLLM 的输入层**直接进 Transformer。
- 音频编码器和适配器**不在 vLLM 里**，而是单 GPU PyTorch 跑；只有 LLM 部分交给 vLLM（并可张量并行）。

---

## 二、音频侧前置概念（对应 `_encode_audio`、`_load_audio_components`）

### 1. Fbank 特征（Filter-bank）
- 语音信号 → 短时傅里叶变换 → 梅尔滤波器组 → 对数压缩，得到 `(T_frames, 80)` 左右的特征矩阵。
- 代码里 `extract_fbank(...)` 就是这一步，`self.frontend`（`WavFrontend`）封装了它。
- **帧率约定**：Fbank 默认每 10ms 一帧（100Hz）。后面时间戳计算里的 `* 6 * 10 / 1000` 就是从帧索引换算回秒的关键（6 是下采样倍数，10 是帧移毫秒数）。

### 2. 下采样与"帧率"
- 音频经过编码器内部两层卷积（stride=2），时间维度被 **4×** 下采样。
- 适配器 `AudioAdaptor` 再做一次 **2×** 下采样 → 总共 **8×**，即每帧约对应 80ms 音频。
- `use_low_frame_rate=True` 时，代码用卷积输出长度公式 `1 + (L - 3 + 2*pad) // stride` 手动重算 `adaptor_out_lens`。**这个公式出现两次**（`_encode_audio` 和 `generate`），读懂它的几何意义：卷积输出长度 = `(输入长 + 2*pad - kernel) / stride + 1`。

### 3. CMVN（Cepstral Mean and Variance Normalization）
- 配置里的 `cmvn_file`：倒谱均值方差归一化参数，让特征分布稳定。
- 代码把相对路径转成绝对路径（`os.path.join(model_dir, cmvn_file)`），因为运行时 cwd 不一定是模型目录。

### 4. `eval()` + `requires_grad=False` + `@torch.no_grad()`
- 三层手段确保音频侧**纯推理、不建图、不占显存**。理解 PyTorch 推理模式是基础。

---

## 三、vLLM 侧前置概念（对应 `__init__`、`generate`）—— **本文件最大门槛**

如果完全没接触过 vLLM，这段代码的 `LLM(...)`、`EmbedsPrompt`、`SamplingParams`、`gpu_memory_utilization` 会像天书。必须先理解：

### 5. vLLM 是什么
- 一个**高吞吐 LLM 推理引擎**，核心创新是 **PagedAttention**（把 KV cache 像操作系统分页一样管理）和 **continuous batching**（请求动态进出 batch）。
- 它**接管了整个 Transformer 解码过程**，包括 KV cache、采样、批调度。你只管喂数据和参数，它批量吐结果。
- 建议先读 vLLM 官方文档的"Quickstart"和"Architecture"，了解它的输入输出模型。

### 6. 张量并行（Tensor Parallelism, `tensor_parallel_size`）
- 把 LLM 的权重矩阵**按列/行切分到多个 GPU**，每个 GPU 算一部分再 AllReduce。
- 这就是为什么音频侧用单 GPU、而 LLM 可以多 GPU——TP 只对 LLM 这种大矩阵运算有意义。
- `tensor_parallel_size=2` = 用 2 张卡跑一个模型（不是跑 2 个模型）。

### 7. `gpu_memory_utilization` 与 KV cache
- vLLM 启动时**预占** GPU 内存的百分比（默认 0.9，这里 0.8），用来放模型权重 + KV cache 池。
- KV cache 越大，能同时处理的请求/上下文越多。理解这个参数要先懂 **KV cache** 是什么（自回归解码时缓存已算过的 K/V，避免重复计算）。

### 8. `EmbedsPrompt`（核心 trick）
- 正常 vLLM 调用：`LLM.generate([{"prompt": "你好"}])` —— 喂文本/token id。
- 这里的调用：`EmbedsPrompt(prompt_embeds=tensor)` —— **直接喂 embedding 张量**，跳过 vLLM 内部的 embedding 查表。
- **为什么必须这样**：音频 embedding 是 PyTorch 算出来的连续向量，没有对应的 token id。只能用"嵌入直喂"模式让它们和文本 embedding 拼在一起进 LLM。
- `enable_prompt_embeds=True` 是打开这个能力的前提开关。
- 这一行（`EmbedsPrompt(prompt_embeds=input_embeds.float())`）是**理解整个文件的钥匙**，没看懂它就读不懂为什么代码要拼 embedding 而不是拼 token。

### 9. `SamplingParams` 与解码策略
- `temperature`：采样温度，0 = 贪心（greedy）。
- `top_p` / `top_k`：核采样 / Top-k 截断。
- `repetition_penalty`：重复惩罚。
- ASR 任务通常 `temperature=0`（要确定性结果），这点要理解为什么。

### 10. `max_model_len` 与 `enforce_eager`
- `max_model_len`：vLLM 接受的最大序列长度（prefix + audio + suffix + 生成）。
- `enforce_eager=True`：禁用 **CUDA Graph**（默认 vLLM 用它加速重复解码），调试时打开。

---

## 四、模型加载与权重拆分概念（对应 `prepare_vllm_model_dir`、`_load_audio_components`、`_load_embedding_layer`）

### 11. `model.pt` 的混合存储
- Fun-ASR-Nano 把**所有**权重（audio_encoder / audio_adaptor / ctc / llm）塞进一个 `model.pt`。
- 不同模块用**前缀**区分：`audio_encoder.*`、`audio_adaptor.*`、`ctc.*`、`ctc_decoder.*`、`llm.*`。
- 代码反复出现这种模式：
  ```python
  {k[len("audio_encoder."):]: v for k, v in state_dict.items() if k.startswith("audio_encoder.")}
  ```
  这就是"按前缀切片 + 去前缀"——必须理解 dict comprehension 和 `load_state_dict(strict=False)`。

### 12. HuggingFace 格式 vs 单文件 checkpoint
- vLLM 只认 **HuggingFace 标准目录结构**：`config.json` + `*.safetensors` + `model.safetensors.index.json` + tokenizer 文件。
- `prepare_vllm_model_dir` 的全部意义：从 `model.pt` 抠出 `llm.*` → 去 `llm.` 前缀 → 存成 `model.safetensors` + 生成 index.json + 从 `Qwen3-0.6B/` 拷配置。**这是一次性的格式转换**，转换后会被缓存（开头那段 glob 检查）。

### 13. safetensors vs pickle (.bin)
- safetensors：内存安全、零拷贝加载快（vLLM 推荐）。
- `model.bin`：PyTorch pickle 格式（回退方案）。
- `model.safetensors.index.json` 里的 `weight_map`：告诉加载器"哪个权重在哪个分片文件"，多分片模型必需。

### 14. 嵌入层共享（`nn.Embedding.from_pretrained`）
- `_load_embedding_layer` 从 `model.pt` 抠出 `llm.embed_tokens.weight`，单独建一个 `nn.Embedding`。
- **为什么要在 PyTorch 侧再建一遍 embedding 层**：因为用 `EmbedsPrompt` 绕过了 vLLM 的输入层，文本 prompt 的 token 必须在 PyTorch 侧自己查表成 embedding 才能和音频 embedding 拼。

---

## 五、Prompt 构建与解码前置概念（对应 `_build_prompt_text`、`_build_input_embeds`、`generate` 末尾）

### 15. ChatML 格式
- Qwen 系列用的对话模板，用 `<|im_start|>role\n...<|im_end|>` 包裹每轮。
- 代码里 `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` 就是 ChatML 的三段。这些特殊 token 是 tokenizer 词表里的，`encode(add_special_tokens=False)` 不会被自动再加 BOS。

### 16. 语音占位标记 `<|startofspeech|>` / `<|endofspeech|>`
- 训练时模型见过这种标记，知道"这里要插入一段音频"。
- 推理时把它们拆成 **prefix（含 startofspeech）+ 音频 embedding + suffix（含 endofspeech）**，音频塞在两个标记之间。

### 17. ITN（Inverse Text Normalization，逆文本归一化）
- 把"一百二十"转成"120"、"两点五"转成"2.5"这类口语→书面格式转换。
- `itn=True/False` 通过 prompt 文本（"不进行文本规整"）告诉模型要不要做。

### 18. 热词（Hotwords）
- 把领域专有词塞进 prompt 的"上下文信息"段，引导 LLM 倾向输出这些词。
- 不是改模型权重，而是**靠 prompt 工程**提升准确率。

### 19. 正则清洗（生成结果后处理）
- vLLM 偶尔会吐出 `<tag>`、`[xxx]`、`endofpatch`、`/sil`、`FFFF` 这类伪 token 或训练残留。
- 那几行 `re.sub(...)` 是经验性清洗，不是理论产物——读懂即可，不必深究每条规则。

---

## 六、CTC 时间戳前置概念（对应 `_compute_timestamps`）

### 20. CTC 与强制对齐
- **CTC**（Connectionist Temporal Classification）：声学模型把每帧映射到词表概率分布，含 blank token。
- **强制对齐**：已知音频和正确文本，反推每个 token 对应哪些帧。
- 调用的是同目录 `tools/utils.py::forced_align`（基于 `torchaudio.functional.forced_align`，内部是 Viterbi DP）。
- **这块要看懂必须先读 CTC 原理**，否则 `blank_id`、`log_softmax`、分组合并都不可解。

### 21. 双解码头设计
- LLM 负责**生成转写文本**（准确率高）。
- CTC 头（`self.ctc` + `self.ctc_decoder`）从**同一个编码器输出**算帧级概率，专门用来**打时间戳**。
- LLM 生成完文本后，把文本当 target，让 CTC 做强制对齐拿时间戳。两个头分工不同，这是工业级 ASR 常见套路。

### 22. 时间戳换算公式
```python
ts["start_time"] = ts["start_time"] * 6 * 10 / 1000
```
- `start_time` 是 CTC 帧索引；`*6` = 编码器下采样倍数（回到 Fbank 帧率）；`*10/1000` = 帧移毫秒转秒。
- 读懂这个公式需要回到第 1、2 点（Fbank 帧率 + 下采样链）。

---

## 七、工程与生态概念

### 23. FunASR 注册表（`tables`）
- `from funasr.register import tables` → `tables.frontend_classes.get(name)` 按字符串名拿类。
- 配置 `config.yaml` 里写的是**类名字符串**，运行时靠注册表反查。理解这个"依赖注入"模式才能看懂为什么配置能驱动实例化。（详见 `docs/learning/learning-path.md` 第二站）

### 24. OmegaConf 配置
- `OmegaConf.load(...) / to_container(..., resolve=True)` 读 YAML 并解析 `${}` 插值。
- FunASR 配置系统的基础。

### 25. ModelScope / HuggingFace Hub
- `from_pretrained` 支持从 ModelScope（`ms`）或 HuggingFace（`hf`）`snapshot_download` 拉模型，或直接用本地目录。
- 理解"模型仓库 = 一个目录"这个心智模型。

### 26. dtype 与混合精度
- `bf16` / `fp16` / `fp32` 三档；代码里音频编码器用 fp32（保数值稳定），适配器和 embedding 用 bf16（省显存）。
- 理解 `dtype_map` 和 `to(dtype=...)` 是 PyTorch 基础。

### 27. 批处理与显存控制
- `generate` 里音频编码按 **每批 8 条** 分组（`batch_size_enc=8`），避免长音频把显存撑爆。
- 文本 embedding 只算一次（跨整个 batch 复用 `prefix_emb`/`suffix_emb`）—— 这是性能优化点。

---

## 八、建议的学习顺序（按依赖关系排序）

| 阶段 | 必读概念 | 目标 |
|------|----------|------|
| **1. 入门** | 1 Fbank、2 下采样、4 推理模式 | 读懂 `_encode_audio` |
| **2. 框架** | 13 HF 格式、23 注册表、24 OmegaConf | 读懂模型加载 |
| **3. 核心** | 5–10 **vLLM 全套**、**8 EmbedsPrompt（最关键）** | 读懂 `__init__` 和 `generate` 主体 |
| **4. Prompt** | 15 ChatML、16 语音标记、17 ITN、18 热词 | 读懂 `_build_input_embeds` |
| **5. 进阶** | 11 前缀切片、14 嵌入共享、12 格式转换 | 读懂 `prepare_vllm_model_dir` |
| **6. 时间戳** | 20 CTC、21 双解码头、22 帧率换算 | 读懂 `_compute_timestamps` |

---

## 九、最低门槛 vs 最大门槛

- **最低门槛**（没这些绝对读不懂）：Fbank、ChatML、PyTorch 推理模式、dict comprehension。
- **最大门槛**（卡住最多人）：**vLLM 的 `EmbedsPrompt` 机制** —— 整个文件的设计都围绕"如何把 PyTorch 算的音频 embedding 喂进 vLLM"展开。**强烈建议先把 vLLM 官方文档的 prompt embeds / vision-language model 部分读一遍**，再回来看这个文件，会有"豁然开朗"的效果。

---

## 十、配套阅读清单

- `funasr/models/fun_asr_nano/model.py` — PyTorch 原版实现，理解训练时完整的 forward 流程
- `funasr/models/fun_asr_nano/tools/utils.py` — `forced_align` 的实现（CTC 强制对齐）
- `funasr/register.py` — `tables` 注册表机制
- `docs/learning/learning-path.md` — FunASR 整体学习路线
- vLLM 官方文档：Quickstart、Architecture、Embeddings/ multimodal input
- Hannun《Sequence Modeling with CTC》—— CTC 入门经典
