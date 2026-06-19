#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Fun-ASR-Nano 流式 vLLM 推理引擎

设计思路：
    - 音频被切分为 720ms 的块（chunk），采用累积重编码策略（cumulative re-encoding）
      即每个 chunk 包含从开头到当前时刻的所有音频，而非仅包含新片段
    - 所有 chunk 被批量放入单次 vLLM generate() 调用中，保证推理正确性和吞吐量
    - 固定/非固定区域：每条输出文本的最后 rollback_chars 个字符为"非固定区域"
      （可能因下一个 chunk 的到来而改变），其余部分为"固定区域"（已确认稳定）
    - 输出随着音频累积逐渐趋于稳定（约 3 秒以上即可获得较稳定的识别结果）

注意：vLLM 将所有 chunk 在一个批次中处理以提升吞吐量。
若需实时流式输出，请使用 demo2.py 中基于 torch 的推理方式。
"""

import logging
import os
import re
from typing import Generator, List, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
# 数据类型映射表：将字符串标识符映射到 PyTorch 的 dtype 对象
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# CJK（中日韩）字符正则表达式，用于判断文本是否包含有意义的 ASR 内容
_CJK_RE = re.compile(r"[一-鿿]")


def _clean_text(text: str) -> str:
    """清理模型输出的原始文本。

    移除 XML 标签、重复垃圾字符、填充标记（filler tokens）以及无效字符，
    得到干净的可读转录文本。

    Args:
        text: 模型生成的原始输出文本。

    Returns:
        清理后的纯文本字符串。
    """
    # 移除所有 XML 风格的标签（如 <tag>...</tag>）
    text = re.sub(r'<[^>]*>|</[^>]*>', '', text)
    # 移除重复出现的垃圾模式（如 >xxx 后跟多个控制字符）
    text = re.sub(r'(>.{2,8}?)\x7f{3,}', '', text)
    # 移除呼吸声、静音、噪声等特殊标记
    text = re.sub(r'\[breath\]|\[noise\]|/sil|endofbreak|FFFF', '', text)
    # 将连续空白字符合并为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 移除替换字符和行首多余的 '>' 符号
    text = text.replace('\ufffd', '').lstrip('>')
    return text.strip()


def _is_meaningful(text: str) -> bool:
    """判断文本是否包含有意义的 ASR 转录内容。

    通过统计文本中的 CJK（中日韩）字符数量来判断。
    至少需要 2 个 CJK 字符才认为该文本具有实际语义价值。

    Args:
        text: 待检查的文本字符串。

    Returns:
        若文本包含足够多的 CJK 字符则返回 True，否则返回 False。
    """
    return len(_CJK_RE.findall(text)) >= 2


class FunASRNanoStreamingVLLM:
    """基于 vLLM 后端的流式自动语音识别（ASR）引擎。

    采用「全量批处理」策略进行流式推理：
    - 音频按固定时长（默认 720ms）切分为 chunk
    - 每个 chunk 采用**累积重编码**：编码的是从音频起点到当前 chunk 结束的全部音频
      （而非仅编码新增片段），这使得模型能利用完整的上下文信息
    - 所有 chunk 的 embedding 被打包进**单次** vLLM generate() 调用中并行生成
    - 结果按 chunk 逐个返回，每个结果区分「固定区域」和「非固定区域」

    关于固定/非固定区域（fixed/unfixed region）：
    - 每次输出的最后 rollback_chars（默认 8）个字符属于「非固定区域」
    - 非固定区域的文字可能因为后续 chunk 带来更多上下文而发生改变
    - 非固定区域之前的文字为「固定区域」，表示已确认稳定的转录内容
    - 这种机制保证了流式输出的最终一致性

    Args:
        model_dir: Fun-ASR-Nano 模型目录路径。
        device: 音频编码器和适配器使用的设备（如 "cuda:0"）。
        dtype: 计算数据类型（"bf16"、"fp16"、"fp32"）。
        tensor_parallel_size: vLLM 张量并行使用的 GPU 数量。
        gpu_memory_utilization: GPU 显存中分配给 KV cache 的比例。
        max_model_len: 模型最大序列长度限制。
        chunk_ms: 每个 chunk 的时长（毫秒），默认 720ms。
        rollback_chars: 每个 chunk 回滚的字符数（非固定区域大小），默认 8。
    """

    def __init__(self, model_dir, device="cuda:0", dtype="bf16",
                 tensor_parallel_size=1, gpu_memory_utilization=0.8,
                 max_model_len=2048, enforce_eager=False,
                 chunk_ms=720, rollback_chars=8, **kwargs):
        """初始化流式 ASR 推理引擎。"""
        from vllm import LLM
        from funasr.models.fun_asr_nano.inference_vllm import prepare_vllm_model_dir

        # 保存基本配置参数
        self.device = device
        self.dtype = dtype
        self.torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        self.model_dir = model_dir
        self.chunk_ms = chunk_ms                    # 每个 chunk 的毫秒数
        self.rollback_chars = rollback_chars         # 非固定区域（回滚）字符数

        # 准备 vLLM 所需的模型目录格式
        vllm_model_dir = prepare_vllm_model_dir(model_dir)
        # 加载音频处理组件（前端、编码器、适配器）
        self._load_audio_components(model_dir)

        # 初始化 vLLM 推理引擎
        vllm_kwargs = kwargs.get("vllm_kwargs", {})
        self.vllm_engine = LLM(
            enable_prompt_embeds=True,              # 启用自定义 embedding 输入模式（关键！）
            model=vllm_model_dir,                   # vLLM 格式的模型路径
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "auto"}.get(dtype, dtype),
            trust_remote_code=True,
            **vllm_kwargs,
        )
        # 获取分词器，用于文本与 token ID 之间的转换
        self.tokenizer = self.vllm_engine.get_tokenizer()
        # 加载 LLM 的 embedding 层（用于手动构建输入 embedding）
        self._load_embedding_layer(model_dir)
        # 从前端获取采样率（通常为 16000 Hz）
        self.sample_rate = self.frontend.fs
        # 计算每个 chunk 对应的采样点数量
        self.chunk_samples = int(self.sample_rate * self.chunk_ms / 1000)

    def _load_audio_components(self, model_dir):
        """加载音频处理三件套：前端（frontend）、编码器（encoder）、适配器（adaptor）。

        这三个组件构成音频信号→LLM 输入嵌入的完整流水线：
        1. frontend: 原始波形 → FBank 特征
        2. encoder:   FBank 特征 → 音频隐状态序列
        3. adaptor:   音频隐状态 → LLM 维度的嵌入向量

        Args:
            model_dir: 模型根目录路径。
        """
        from omegaconf import OmegaConf
        from funasr.register import tables
        # 加载模型配置文件
        config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
        self._config = OmegaConf.to_container(config, resolve=True)

        # === 1. 加载音频前端（Frontend）：负责波形到声学特征的提取 ===
        frontend_class = tables.frontend_classes.get(config["frontend"])
        frontend_conf = OmegaConf.to_container(config.get("frontend_conf", {}), resolve=True)
        self.frontend = frontend_class(**frontend_conf)
        self.frontend.eval()

        # === 2. 加载音频编码器（Encoder）：负责声学特征到隐状态的转换 ===
        encoder_conf = OmegaConf.to_container(config.get("audio_encoder_conf", {}), resolve=True)
        if encoder_conf.get("hub") == "ms":
            # 从 ModelScope Hub 加载预训练编码器
            from funasr import AutoModel as FAM
            enc_m = FAM(model=config["audio_encoder"], model_revision="master", disable_update=True)
            self.audio_encoder_output_size = getattr(enc_m.model, "encoder_output_size", -1)
            self.audio_encoder = enc_m.model.model.encoder if hasattr(enc_m.model, "model") else enc_m.model.encoder
        else:
            # 从本地注册表加载编码器类并实例化
            encoder_class = tables.encoder_classes.get(config["audio_encoder"])
            self.audio_encoder = encoder_class(input_size=self.frontend.output_size(), **encoder_conf)
            self.audio_encoder_output_size = self.audio_encoder.output_size()
        self.audio_encoder.eval()
        # 冻结编码器参数（不参与梯度计算）
        for p in self.audio_encoder.parameters(): p.requires_grad = False

        # === 3. 加载音频适配器（Adaptor）：负责将编码器输出投影到 LLM 嵌入空间 ===
        adaptor_conf = OmegaConf.to_container(config.get("audio_adaptor_conf", {}), resolve=True)
        adaptor_class = tables.adaptor_classes.get(config["audio_adaptor"])
        if self.audio_encoder_output_size > 0:
            adaptor_conf["encoder_dim"] = self.audio_encoder_output_size
        self.audio_adaptor = adaptor_class(**adaptor_conf)
        self.audio_adaptor.eval()
        # 冻结适配器参数
        for p in self.audio_adaptor.parameters(): p.requires_grad = False

        # === 4. 加载模型权重 ===
        model_pt = os.path.join(model_dir, "model.pt")
        if os.path.exists(model_pt):
            ckpt = torch.load(model_pt, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)
            # 提取并加载编码器权重（去掉 "audio_encoder." 前缀）
            enc_s = {k[len("audio_encoder."):]: v for k, v in sd.items() if k.startswith("audio_encoder.")}
            if enc_s: self.audio_encoder.load_state_dict(enc_s, strict=False)
            # 提取并加载适配器权重（去掉 "audio_adaptor." 前缀）
            adp_s = {k[len("audio_adaptor."):]: v for k, v in sd.items() if k.startswith("audio_adaptor.")}
            if adp_s: self.audio_adaptor.load_state_dict(adp_s, strict=False)

        # 将组件移至目标设备；编码器用 fp32 保证精度，适配器可用低精度
        self.audio_encoder = self.audio_encoder.to(self.device, dtype=torch.float32)
        self.audio_adaptor = self.audio_adaptor.to(self.device, dtype=self.torch_dtype)

    def _load_embedding_layer(self, model_dir):
        """加载 LLM 的 token embedding 层。

        该层用于在 _build_embeds 中将文本 token ID 转换为嵌入向量，
        与音频嵌入拼接后作为 vLLM 的完整输入。

        Args:
            model_dir: 模型根目录路径。

        Raises:
            RuntimeError: 如果在模型权重中找不到 LLM embedding 权重。
        """
        ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        for key in sd:
            # 查找 LLM 的 embed_tokens 权重（键名格式如 "llm.embed_tokens.weight"）
            if "embed_tokens.weight" in key and key.startswith("llm."):
                # 从预训练权重创建冻结的 Embedding 层
                self.embed_tokens = nn.Embedding.from_pretrained(sd[key], freeze=True)
                self.embed_tokens = self.embed_tokens.to(self.device, dtype=self.torch_dtype)
                return
        raise RuntimeError("Could not find LLM embedding weights")

    @torch.no_grad()
    def _encode_audio(self, audio_samples):
        """将原始音频样本编码为 LLM 嵌入空间的向量。

        处理流程：原始波形 → FBank 特征 → 编码器隐状态 → 适配器输出（LLM维度）

        注意：这里采用的是**累积编码**——传入的 audio_samples 是从音频开头
        到当前位置的全部样本，而非仅当前 chunk 的新增部分。这是流式 ASR
        累积重编码策略的核心。

        Args:
            audio_samples: 原始音频波形数据（Tensor 或 numpy 数组），16kHz 采样率。

        Returns:
            tuple: (adaptor_output, adaptor_output_lengths)
                - adaptor_output: 形状为 [1, seq_len, embed_dim] 的嵌入张量
                - adaptor_output_lens: 各样本的有效长度
        """
        from funasr.utils.load_utils import extract_fbank
        # Step 1: 提取 FBank（滤波器组）声学特征
        speech, speech_lengths = extract_fbank(
            audio_samples, data_type="sound", frontend=self.frontend, is_final=True)
        speech = speech.to(self.device, dtype=torch.float32)
        speech_lengths = speech_lengths.to(self.device)
        # Step 2: 通过音频编码器获取隐状态序列
        enc_out, enc_lens = self.audio_encoder(speech, speech_lengths)
        # Step 3: 通过适配器将隐状态投影到 LLM 的嵌入维度
        adp_out, adp_lens = self.audio_adaptor(enc_out.to(dtype=self.torch_dtype), enc_lens)
        return adp_out, adp_lens

    def _build_prompt_text(self, hotwords=None, language=None, itn=True):
        """构建用户提示文本（prompt），包含热词、语言和 ITN 设置。

        Args:
            hotwords: 热词列表，用于引导模型优先识别特定词汇。
            language: 目标转写语言（如 "中文"）。
            itn: 是否执行逆文本标准化（Inverse Text Normalization），
                 例如将 "一百二十三" 转换为 "123"。

        Returns:
            构建好的提示文本字符串。
        """
        hotwords = hotwords or []
        if hotwords:
            # 有热词时构建带上下文信息的 prompt
            prompt = "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            prompt += f"热词列表：[{', '.join(hotwords)}]\n"
        else:
            prompt = ""
        # 追加语言设置和 ITN 开关
        prompt += f"语音转写成{language}" if language else "语音转写"
        if not itn: prompt += "，不进行文本规整"
        return prompt + "："

    @torch.no_grad()
    def _build_embeds(self, audio_embeds, audio_embed_lens, prev_text="", hotwords=None, language=None, itn=True):
        """构建完整的输入嵌入序列——核心的三段式拼接逻辑。

        最终输入 vLLM 的嵌入由三部分按顺序拼接而成：

        ┌─────────────────────────────────────────────────────────────┐
        │  Prefix（前缀） │  Audio（音频嵌入）│  Suffix（后缀/上文）     │
        │  system + user  │  来自 adaptor     │  assistant + prev_text │
        │  prompt 文本     │  的音频特征       │  已确认的转录前文       │
        ├─────────────────────────────────────────────────────────────┤
        │  [system_role]   │  [可变长度]       │  [assistant_prefix]    │
        │  + user_prompt   │  取决于音频长度   │  + 已有转录文本         │
        └─────────────────────────────────────────────────────────────┘

        三段拼接的意义：
        - Prefix：提供系统角色和任务指令，告诉 LLM 这是一个语音转写任务
        - Audio：核心输入，承载了当前累积音频的所有声学信息
        - Suffix：
          * 提供 assistant 角色前缀，引导 LLM 以助手身份生成转写文本
          * 当存在 prev_text 时，将其追加在后缀末尾，实现「续写」效果
            （即 LLM 会基于已有转录继续生成新的内容）

        Args:
            audio_embeds: 音频适配器的输出嵌入，形状 [batch, seq_len, embed_dim]。
            audio_embed_lens: 音频嵌入的有效长度。
            prev_text: 之前已确认的转录文本（Stage 2 使用），作为"上文"传入
                       使模型在此基础上继续生成，避免重复输出已有内容。
            hotwords: 热词列表。
            language: 目标语言。
            itn: 是否启用逆文本标准化。

        Returns:
            拼接后的完整嵌入张量，形状 [total_seq_len, embed_dim]。
        """
        # 构建用户侧提示文本
        prompt = self._build_prompt_text(hotwords, language, itn)

        # === 构建 Prefix 段：系统提示 + 用户指令 + 语音开始标记 ===
        prefix_text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{prompt}<|startofspeech|>"
        )
        # 这里的 <startofspeech|> 是特殊标记，告知模型接下来是音频内容

        # === 构建 Suffix 段：语音结束标记 + 助手角色前缀 + 已有转录文本 ===
        suffix_text = (
            "<|endofspeech|><|im_end|>\n"
            "<|im_start|>assistant\n"
            "###\n\n###\n\n"  # 特殊分隔符/格式标记
        )
        # 如果有已确认的前文（prev_text），追加到 suffix 末尾
        # 这样 LLM 就会"看到"之前的转录结果，从而只生成新的增量内容
        if prev_text:
            suffix_text += prev_text

        # 将文本转换为 token ID，再通过 embedding 层得到嵌入向量
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)
        prefix_emb = self.embed_tokens(torch.tensor(prefix_ids, dtype=torch.long, device=self.device))
        suffix_emb = self.embed_tokens(torch.tensor(suffix_ids, dtype=torch.long, device=self.device))

        # 截取有效的音频嵌入（去掉 padding 部分）
        audio_emb = audio_embeds[0, :audio_embed_lens[0].item(), :]

        # === 三段拼接：prefix → audio → suffix ===
        return torch.cat([prefix_emb, audio_emb, suffix_emb], dim=0)

    def streaming_generate(self, audio_input, chunk_ms=None, rollback_chars=None,
                           hotwords=None, language=None, itn=True,
                           max_new_tokens=200, temperature=0.0, **kwargs):
        """流式 ASR 推理主方法——处理所有 chunk 并逐个返回结果。

        ========== 两阶段策略（Two-Stage Strategy）==========

        对于长音频，本方法采用两阶段策略来平衡输出质量和效率：

        【Stage 1 — 无前文阶段（Fresh Generation）】
        - 处理前 N 个 chunk（默认 N=10，约 7.2 秒音频）
        - 每个 chunk 独立编码，**不传入任何 prev_text**
        - 模型完全基于音频内容从头生成转录
        - 目的：让模型有足够的音频上下文来产生稳定、准确的初始输出
        - 从 Stage 1 的所有输出中挑选最佳（最长且有意义的）作为 stable output

        【Stage 2 — 带前文阶段（Continuation with Context）】
        - 处理剩余的 chunk
        - 将 Stage 1 选出的 stable output（去除末尾 rollback_chars）作为 prev_text
        - 每个新 chunk 的 embedding 都会带上这个前文
        - 模型在前文基础上**续写**新的转录内容
        - 目的：避免重复输出已有内容，同时保持上下文连贯性

        ========== 关键机制说明 ==========

        【累积重编码（Cumulative Re-encoding）】
        第 i 个 chunk 编码的是音频的第 0 ~ (i+1)*chunk_samples 个采样点，
        而非第 i*(i+1)*chunk_samples ~ (i+1)*chunk_samples 个采样点。
        这意味着随着 chunk 索引增加，编码的音频越来越长，
        模型能利用越来越多的上下文信息，因此输出质量逐步提升。

        【固定/非固定区域（Fixed/Unfixed Region）】
        每个 chunk 的输出文本被划分为两个区域：
        - 固定区域（fixed_text）：除去末尾 rollback_chars 字符的部分，
          这部分内容被认为已经稳定，不会因后续 chunk 而改变
        - 非固定区域：末尾 rollback_chars 个字符，可能随下一个 chunk 变化
        这个机制允许调用方安全地使用 fixed_text 作为"已确认"的输出，
        同时保留非固定区域供后续修正。

        【EmbedsPrompt】
        vLLM 的 EmbedsPrompt 是一种特殊的输入格式，允许直接传入预计算的
        嵌入张量而非文本 prompt。本方法通过 _build_embeds 构建好完整的
        嵌入序列后，包装为 EmbedsPrompt 对象传给 vllm_engine.generate()。

        Args:
            audio_input: 音频输入，支持以下格式：
                - str: 音频文件路径
                - numpy.ndarray: 16kHz 采样的音频数组
                - torch.Tensor: 16kHz 采样的音频张量
            chunk_ms: 每个 chunk 的时长（毫秒），默认使用实例化时的值（720ms）。
            rollback_chars: 回滚字符数（非固定区域大小），默认 8。
            hotwords: 热词列表，用于提升特定词汇的识别准确率。
            language: 语言提示（如 "中文"）。
            itn: 是否启用逆文本标准化，默认 True。
            max_new_tokens: 每个 chunk 生成的最大 token 数，默认 200。
            temperature: 采样温度，0 表示贪心解码（确定性输出）。

        Yields:
            dict: 包含以下键的字典：
                - text: 当前 chunk 的完整转录文本
                - fixed_text: 已确认稳定的转录文本（去除非固定区域）
                - is_final: 是否为最后一个 chunk（全部音频处理完毕）
                - chunk_idx: 当前 chunk 的索引（从 1 开始）
                - audio_duration_ms: 当前累积处理的音频时长（毫秒）
        """
        from vllm import SamplingParams
        try:
            from vllm.inputs import EmbedsPrompt
        except ImportError:
            # 兼容不同版本的 vLLM API
            from vllm.inputs.data import EmbedsPrompt
        from funasr.utils.load_utils import load_audio_text_image_video

        # 使用传入参数或回退到实例化时的默认值
        chunk_ms = chunk_ms or self.chunk_ms
        rollback_chars = rollback_chars or self.rollback_chars

        # === 音频输入预处理：统一转为 torch Tensor ===
        if isinstance(audio_input, str):
            # 字符串路径：加载音频文件
            audio_data = load_audio_text_image_video(audio_input, fs=self.sample_rate)
        elif isinstance(audio_input, np.ndarray):
            # numpy 数组：直接转为 tensor
            audio_data = torch.from_numpy(audio_input).float()
        elif isinstance(audio_input, torch.Tensor):
            # tensor：确保为 float 类型
            audio_data = audio_input.float()
        else:
            raise ValueError(f"Unsupported audio type: {type(audio_input)}")
        # 如果是多通道音频，压扁为单通道
        if audio_data.dim() > 1:
            audio_data = audio_data.squeeze()

        # === 计算分块信息 ===
        total_samples = audio_data.shape[0]                          # 总采样点数
        chunk_samples = int(self.sample_rate * chunk_ms / 1000)      # 每个 chunk 的采样点数
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples  # 总 chunk 数（向上取整）

        # 配置 vLLM 采样参数
        params = SamplingParams(
            max_tokens=max_new_tokens,      # 每个 chunk 最大生成 token 数
            temperature=temperature,         # 0 = 贪心解码（确定性输出）
            repetition_penalty=1.3,         # 重复惩罚系数，减少重复输出
            skip_special_tokens=True,       # 输出中跳过特殊 token
        )

        # ==================== Stage 1：无前文的初始阶段 ====================
        # 前 stage1_count 个 chunk 不带任何 prev_text，让模型独立生成
        # 通常 10 个 chunk（约 7.2 秒）足以让输出趋于稳定
        stage1_count = min(10, num_chunks)

        # 构造 Stage 1 的所有 prompt（每个 chunk 独立编码+构建嵌入）
        prompts_s1 = []
        chunk_infos_s1 = []           # 记录每个 chunk 的元信息
        for i in range(stage1_count):
            end_sample = min((i + 1) * chunk_samples, total_samples)
            # 累积重编码：编码从音频开头到当前 chunk 结束的全部音频
            adaptor_out, adaptor_out_lens = self._encode_audio(audio_data[:end_sample])
            # 构建嵌入（prev_text 为空 —— 不带入任何已有转录）
            embeds = self._build_embeds(
                adaptor_out, adaptor_out_lens, prev_text="",
                hotwords=hotwords, language=language, itn=itn)
            # 包装为 EmbedsPrompt 对象，供 vLLM 直接消费嵌入张量
            prompts_s1.append(EmbedsPrompt(prompt_embeds=embeds.float()))
            # 记录当前 chunk 的元信息
            chunk_infos_s1.append({
                "chunk_idx": i + 1,                                    # chunk 序号（从 1 开始）
                "is_final": end_sample >= total_samples,               # 是否为最后一个 chunk
                "audio_duration_ms": end_sample * 1000 / self.sample_rate,  # 累积音频时长(ms)
            })

        # ★ 核心操作：将 Stage 1 所有 chunk 的嵌入一次性送入 vLLM 批量生成
        outputs_s1 = self.vllm_engine.generate(prompts_s1, params, use_tqdm=False)

        # 从 Stage 1 输出中挑选最佳的稳定文本
        best_text = ""
        results_s1 = []
        for output in outputs_s1:
            # 提取生成的文本
            text = output.outputs[0].text
            if not text and output.outputs[0].token_ids:
                # 如果 text 为空但有 token_ids，手动解码
                text = self.tokenizer.decode(list(output.outputs[0].token_ids), skip_special_tokens=True)
            text = _clean_text(text)       # 清理特殊标记和垃圾字符
            results_s1.append(text)
            # 选择最长且有意义的结果作为 best_text（通常后面的 chunk 质量更好）
            if _is_meaningful(text) and len(text) > len(best_text):
                best_text = text

        # 逐个 yield Stage 1 的结果
        for i, (text, info) in enumerate(zip(results_s1, chunk_infos_s1)):
            if info["is_final"]:
                # 最后一个 chunk：全部文本都是固定的
                fixed_text = text
            elif _is_meaningful(text) and len(text) > rollback_chars:
                # 有意义且足够长：截去末尾 rollback_chars 作为非固定区域
                fixed_text = text[:-rollback_chars]
            else:
                # 太短或无意义：不输出固定文本
                fixed_text = ""
            yield {"text": text, "fixed_text": fixed_text, **info}

        # ==================== Stage 2：带前文的续写阶段 ====================
        # 如果还有未处理的 chunk，进入 Stage 2
        if stage1_count < num_chunks:
            # 用 Stage 1 最佳结果的固定区域（去掉末尾回滚字符）作为前文
            # 这样 Stage 2 的模型就能"看到"已有的转录，只需生成新增内容
            prev_text = best_text[:-rollback_chars] if len(best_text) > rollback_chars else best_text

            # 构造 Stage 2 的所有 prompt（这次带有 prev_text）
            prompts_s2 = []
            chunk_infos_s2 = []
            for i in range(stage1_count, num_chunks):
                end_sample = min((i + 1) * chunk_samples, total_samples)
                # 继续累积重编码
                adaptor_out, adaptor_out_lens = self._encode_audio(audio_data[:end_sample])
                # 构建嵌入时传入 prev_text，实现"续写"效果
                embeds = self._build_embeds(
                    adaptor_out, adaptor_out_lens, prev_text=prev_text,
                    hotwords=hotwords, language=language, itn=itn)
                prompts_s2.append(EmbedsPrompt(prompt_embeds=embeds.float()))
                chunk_infos_s2.append({
                    "chunk_idx": i + 1,
                    "is_final": end_sample >= total_samples,
                    "audio_duration_ms": end_sample * 1000 / self.sample_rate,
                })

            # ★ 批量生成 Stage 2 的结果
            outputs_s2 = self.vllm_engine.generate(prompts_s2, params, use_tqdm=False)

            # 逐个 yield Stage 2 的结果
            for output, info in zip(outputs_s2, chunk_infos_s2):
                text = output.outputs[0].text
                if not text and output.outputs[0].token_ids:
                    text = self.tokenizer.decode(list(output.outputs[0].token_ids), skip_special_tokens=True)
                text = _clean_text(text)
                # 完整文本 = 前文（prev_text）+ 本次新生成的文本
                full_text = prev_text + text

                if info["is_final"]:
                    # 最后一个 chunk：全部文本都确认为固定
                    fixed_text = full_text
                elif _is_meaningful(full_text) and len(full_text) > rollback_chars:
                    # 截去末尾 rollback_chars 作为非固定区域
                    fixed_text = full_text[:-rollback_chars]
                else:
                    # 输出太短或无意义：固定文本至少保留前文部分
                    fixed_text = prev_text

                yield {"text": full_text, "fixed_text": fixed_text, **info}

    def generate(self, audio_input, **kwargs):
        """执行流式推理并以列表形式返回所有 chunk 的结果。

        这是 streaming_generate 的便捷封装，将生成器结果收集为列表返回。

        Args:
            audio_input: 音频输入（同 streaming_generate）。
            **kwargs: 传递给 streaming_generate 的额外参数。

        Returns:
            list[dict]: 所有 chunk 的结果字典列表。
        """
        return list(self.streaming_generate(audio_input, **kwargs))

    @classmethod
    def from_pretrained(cls, model="FunAudioLLM/Fun-ASR-Nano-2512", hub="ms",
                        device="cuda:0", dtype="bf16", tensor_parallel_size=1,
                        gpu_memory_utilization=0.8, max_model_len=2048,
                        chunk_ms=720, rollback_chars=8, **kwargs):
        """工厂方法：从 ModelScope Hub 或 HuggingFace Hub 或本地路径加载模型。

        Args:
            model: 模型标识符或本地路径。
                - 若为本地目录路径，则直接使用
                - 若为 ModelScope 模型名（hub="ms"），从 ModelScope 下载
                - 否则从 HuggingFace Hub 下载
            hub: 模型仓库来源（"ms"/"modelscope" 或 "hf"）。
            device: 计算设备。
            dtype: 数据类型。
            tensor_parallel_size: 张量并行 GPU 数。
            gpu_memory_utilization: GPU 显存利用率。
            max_model_len: 最大序列长度。
            chunk_ms: chunk 时长（毫秒）。
            rollback_chars: 回滚字符数。

        Returns:
            初始化完成的 FunASRNanoStreamingVLLM 实例。
        """
        if os.path.isdir(model):
            # 本地目录：直接使用
            model_dir = model
        else:
            if hub in ("ms", "modelscope"):
                # 从 ModelScope Hub 下载模型
                from modelscope.hub.snapshot_download import snapshot_download
                model_dir = snapshot_download(model, revision=kwargs.pop("revision", "master"))
            else:
                # 从 HuggingFace Hub 下载模型
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download(model)
        return cls(model_dir=model_dir, device=device, dtype=dtype,
                   tensor_parallel_size=tensor_parallel_size,
                   gpu_memory_utilization=gpu_memory_utilization,
                   max_model_len=max_model_len, chunk_ms=chunk_ms,
                   rollback_chars=rollback_chars, **kwargs)
