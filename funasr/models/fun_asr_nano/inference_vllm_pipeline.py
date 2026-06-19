#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Fun-ASR-Nano vLLM Pipeline: VAD + ASR(vLLM) + Speaker Diarization.
（Fun-ASR-Nano vLLM 推理流水线：语音活动检测 + 自动语音识别(vLLM) + 说话人分离）

Replicates AutoModel's inference_with_vad pipeline but uses vLLM for
the LLM decoding step, enabling batch processing of all VAD segments
in a single generate() call.

本模块复现了 AutoModel 的 inference_with_vad 流水线，但在 LLM 解码步骤中使用 vLLM，
使得所有 VAD 分段可以在单次 generate() 调用中批量处理，大幅提升推理效率。

Pipeline 流程概述:
    输入音频 → [VAD] → 语音分段列表 → [Audio Encoder] → 音频嵌入 → [EmbedsPrompt]
    → [vLLM generate 批量解码] → 文本结果 → [Speaker Embedding] → [聚类] → 最终输出

Usage:
    from funasr.models.fun_asr_nano.inference_vllm_pipeline import FunASRNanoVLLMPipeline

    model = FunASRNanoVLLMPipeline(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        vad_model="fsmn-vad",
        spk_model="cam++",
        tensor_parallel_size=2,
    )
    results = model.generate("long_meeting.wav", language="中文")
"""

import logging
import os
import re
import time
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# 数据类型映射表：将字符串表示的精度类型转换为 PyTorch 的 dtype 对象
# bf16: 半精度浮点（Brain Float 16），适合训练和推理，节省显存
# fp16: 标准 16 位浮点数
# fp32: 标准单精度浮点数
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _clean_text(text: str) -> str:
    """Remove tags, fillers, and garbage from output.
    （清理 ASR 模型输出的文本：移除特殊标签、填充词和无意义字符）

    本函数对 ASR 模型生成的原始文本进行后处理清理：
    - 移除 XML/HTML 风格的标签（如 <lang>、</lang>）
    - 移除重复的填充模式（如 >>>>>>）
    - 移除呼吸声、静音标记等非文本符号
    - 压缩多余空白字符

    Args:
        text: ASR 模型输出的原始文本字符串

    Returns:
        清理后的纯文本字符串
    """
    # 使用正则表达式移除所有尖括号标签（如 <lang_code>、</lang_code> 等）
    text = re.sub(r"<[^>]*>|</[^>]*>", "", text)
    # 移除连续重复 3 次以上的填充模式（如 ">>>>" 这类无意义重复）
    text = re.sub(r"(>.{2,8}?)\1{3,}", "", text)
    # 移除呼吸声、噪声、静音符、断点结束符等非语言标记
    text = re.sub(r"\[breath\]|\[noise\]|/sil|endofbreak|FFFF", "", text)
    # 将多个连续空白字符压缩为单个空格
    text = re.sub(r"\s+", " ", text)
    # 移除乱码字符（�）和行首的多余 > 符号
    text = text.replace("�", "").lstrip(">")
    return text.strip()


class FunASRNanoVLLMPipeline:
    """VAD + ASR(vLLM) + Speaker pipeline.
    （FunASR Nano vLLM 推理流水线：整合 VAD、ASR 和说话人识别的端到端解决方案）

    本类实现了完整的语音处理流水线，核心创新在于使用 vLLM 进行批量 ASR 推理，
    将传统逐段处理改为一次性批量生成，显著提升长音频的处理效率。

    Pipeline 完整流程:
        ┌─────────────┐
        │  输入音频    │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ Step 1: VAD │ ← 语音活动检测，将长音频切分为多个语音片段
        │ (fsmn-vad)  │   返回每个片段的起止时间戳（毫秒）
        └──────┬──────┘
               ▼
        ┌──────────────────────────────────┐
        │ Step 2: Audio Encoder            │ ← 对每个 VAD 片段进行音频编码
        │ (FunASR Audio Encoder)           │   提取声学特征并转换为嵌入向量
        └──────┬───────────────────────────┘
               ▼
        ┌──────────────────────────────────┐
        │ Step 3: Build EmbedsPrompt       │ ← 将音频嵌入包装为 vLLM 的
        │ (EmbedsPrompt)                   │   EmbedsPrompt 输入格式
        └──────┬───────────────────────────┘
               ▼
        ┌──────────────────────────────────┐
        │ Step 4: vLLM Generate (批量)     │ ← 核心：所有片段在一次调用中
        │ (vLLM Engine)                    │   并行完成 LLM 解码生成文本
        └──────┬───────────────────────────┘
               ▼
        ┌──────────────────────────────────┐
        │ Step 5: Speaker Diarization      │ ← 可选：提取说话人嵌入向量
        │ (Cam++ + ClusterBackend)         │   并进行聚类，区分不同说话人
        └──────┬───────────────────────────┘
               ▼
        ┌──────────────────────────────────┐
        │ Step 6: Merge & Output           │ ← 合并文本、时间戳、说话人标签
        │                                  │   输出结构化结果字典
        └──────────────────────────────────┘

    Args:
        model: Fun-ASR-Nano 模型名称或本地路径（如 "FunAudioLLM/Fun-ASR-Nano-2512"）
        vad_model: VAD 模型名称（如 "fsmn-vad"），设为 None 则禁用 VAD（整段音频作为单个片段）
        vad_kwargs: VAD 模型的额外配置参数（如 {"max_single_segment_time": 30000} 控制最大单段时长）
        spk_model: 说话人识别模型名称（如 "cam++"），设为 None 则禁用说话人分离功能
        spk_kwargs: 说话人模型的额外配置参数
        hub: 模型下载源，"ms" 表示 ModelScope，"hf" 表示 HuggingFace
        device: 音频编码器、VAD 和说话人模型运行的设备（如 "cuda:0"、"cpu"）
        dtype: ASR 计算的数据精度（"bf16"/"fp16"/"fp32"）
        tensor_parallel_size: vLLM 使用的 GPU 数量（张量并行大小），用于多卡推理加速
        gpu_memory_utilization: vLLM 占用的 GPU 显存比例（0~1 之间），预留部分显存给其他组件
        max_model_len: vLLM 引擎支持的最大序列长度（token 数）
        enforce_eager: 是否强制使用 Eager 模式（禁用 CUDA Graph 加速，便于调试）
    """

    def __init__(
        self,
        model: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        vad_model: str = None,
        vad_kwargs: dict = None,
        spk_model: str = None,
        spk_kwargs: dict = None,
        hub: str = "ms",
        device: str = "cuda:0",
        dtype: str = "bf16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 4096,
        enforce_eager: bool = False,
        **kwargs,
    ):
        """初始化 FunASR Nano vLLM 推理流水线。
        
        构建三个核心组件：
        1. ASR 引擎（基于 vLLM）- 负责音频到文本的转换
        2. VAD 模型（基于 PyTorch）- 负责语音活动检测与分段
        3. 说话人模型（基于 PyTorch）- 负责说话人特征提取与聚类
        """
        from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

        # ══════════════════════════════════════════════════════════════
        # 初始化 ASR 引擎（vLLM）
        # 这是整个流水线的核心组件，负责：
        #   - 加载 Fun-ASR-Nano 模型的音频编码器和 LLM 解码器
        #   - 创建 vLLM 推理引擎，支持高效批量生成
        #   - 管理 tokenizer、CTC 解码器等子模块
        # ══════════════════════════════════════════════════════════════
        self.asr_engine = FunASRNanoVLLM.from_pretrained(
            model=model, hub=hub, device=device, dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, enforce_eager=enforce_eager,
            **kwargs,
        )

        # ══════════════════════════════════════════════════════════════
        # 初始化 VAD（Voice Activity Detection）模型
        # VAD 用于检测音频中的语音段落边界，将长音频切分为多个短片段
        # 常用模型：fsmn-vad（基于 FSMN 网络的语音活动检测器）
        # 输出格式：[[start_ms, end_ms], ...]，每个元素是一个语音片段的起止时间（毫秒）
        # ══════════════════════════════════════════════════════════════
        self.vad_model = None
        if vad_model is not None:
            from funasr import AutoModel
            vad_kw = vad_kwargs or {}
            self.vad_model = AutoModel(
                model=vad_model, device=device, disable_update=True, **vad_kw
            )

        # ══════════════════════════════════════════════════════════════
        # 初始化说话人识别（Speaker Diarization）相关模型
        # 包含两个子组件：
        #   1. spk_model: 说话人嵌入提取模型（如 Cam++），用于从音频中提取
        #      能够表征说话人身份的特征向量（embedding）
        #   2. cb_model: 聚类后端（Cluster Backend），用于对所有片段的说话人
        #      嵌入进行聚类，自动判断有多少个不同的说话人
        # ══════════════════════════════════════════════════════════════
        self.spk_model = None
        self.cb_model = None
        if spk_model is not None:
            from funasr import AutoModel
            from funasr.models.campplus.cluster_backend import ClusterBackend
            spk_kw = spk_kwargs or {}
            # 初始化说话人嵌入提取模型（Cam++ 等）
            self.spk_model = AutoModel(
                model=spk_model, device=device, disable_update=True, **spk_kw
            )
            # 初始化聚类后端，用于对说话人嵌入进行聚类分析
            # 支持预设说话人数（oracle_num）或自动估计说话人数
            cb_kwargs = spk_kw.get("cb_kwargs", {})
            self.cb_model = ClusterBackend(**cb_kwargs).to(device)

        # 保存设备信息和采样率（默认 16kHz，符合大多数 ASR 模型的输入要求）
        self.device = device
        self.sample_rate = 16000  # 音频采样率：16kHz

    def generate(
        self,
        input: Union[str, List[str]],
        hotwords: List[str] = None,
        language: str = None,
        itn: bool = True,
        max_new_tokens: int = 512,
        batch_size_s: int = 300,
        return_spk_res: bool = True,
        **kwargs,
    ) -> List[dict]:
        """Run the full pipeline: VAD → ASR(vLLM) → Speaker.
        （执行完整的推理流水线：VAD 分段 → vLLM 批量 ASR 识别 → 说话人分离）

        这是流水线的主入口方法，对外提供统一的调用接口。
        支持单文件或多文件批量处理。

        处理流程概览:
            for each audio_file:
                1. 加载音频数据
                2. VAD 检测 → 获取语音分段列表
                3. 编码所有分段 → 构建 EmbedsPrompt 列表
                4. vLLM 一次 generate() 调用批量生成所有分段的文本
                5. 提取说话人嵌入并进行聚类（可选）
                6. 合并结果（文本 + 时间戳 + 说话人标签）

        Args:
            input: 音频文件路径（字符串）或路径列表（支持批量处理多个文件）
            hotwords: 热词列表，用于提升特定词汇的识别准确率（如专有名词、术语等）
            language: 语言提示（如 "中文"、"英文"、"auto" 等），帮助模型选择正确的语言
            itn: 是否执行逆文本标准化（Inverse Text Normalization）
                 True 时会将 "一百二十三" 转换为 "123"，"百分之五十" 转换为 "50%" 等
            max_new_tokens: 每个 VAD 分段最大生成的 token 数量
                 控制单段文本的最大长度，避免过长输出
            batch_size_s: 批次大小限制（秒），用于控制内存占用
                 当 VAD 分段总时长超过此值时，会分批处理
            return_spk_res: 是否返回说话人分离结果
                 设为 False 可跳过说话人识别步骤，加快推理速度

        Returns:
            结果列表，每个元素对应一个输入音频文件，格式为:
            [{
                "key": "文件名（不含扩展名）",
                "text": "完整转录文本（所有分段拼接）",
                "timestamp": [[start_ms, end_ms], ...],  # 各分段的时间戳
                "sentence_info": [                       # 仅当 return_spk_res=True 时存在
                    {
                        "start": 起始时间(ms),
                        "end": 结束时间(ms),
                        "text": "该段文本内容",
                        "timestamp": [[start_ms, end_ms]],
                        "spk_label": "说话人标识（如 SPK001）",  # 说话人标签
                    },
                    ...
                ]
            }]
        """
        # 统一输入格式：如果是单个文件路径字符串，转为单元素列表
        if isinstance(input, str):
            input = [input]

        # 依次处理每个音频文件，收集所有结果
        results_all = []
        for audio_path in input:
            result = self._process_one(
                audio_path, hotwords=hotwords, language=language,
                itn=itn, max_new_tokens=max_new_tokens,
                batch_size_s=batch_size_s, return_spk_res=return_spk_res,
                **kwargs,
            )
            results_all.append(result)
        return results_all

    def _process_one(self, audio_path, **kwargs):
        """Process a single audio file through the full pipeline.
        （处理单个音频文件，执行完整的 VAD→ASR→Speaker 流水线）

        这是每个音频文件的核心处理逻辑，按顺序执行以下步骤:

        Step 1 - 音频加载: 读取音频文件并转换为 numpy 数组
        Step 2 - VAD 分段: 使用 VAD 模型检测语音活动区域
        Step 3 - 音频切片: 按 VAD 边界切分音频为数个片段
        Step 4 - 批量 ASR: 通过 vLLM 一次性生成所有片段的文本
        Step 5 - 说话人识别: 提取说话人嵌入并聚类（可选）
        Step 6 - 结果合并: 整合文本、时间戳和说话人信息

        Args:
            audio_path: 音频文件的完整路径
            **kwargs: 从 generate() 传递的其他参数（hotwords, language, itn 等）

        Returns:
            包含完整推理结果的字典，含 key/text/timestamp/sentence_info 等字段
        """
        from funasr.utils.load_utils import load_audio_text_image_video
        from funasr.utils.vad_utils import slice_padding_audio_samples

        # 从文件名中提取 key（去掉扩展名），用作结果的唯一标识符
        key = os.path.splitext(os.path.basename(audio_path))[0]

        # ══════════════════════════════════════════════════════════════
        # Step 1: 加载音频数据
        # 使用 FunASR 的通用加载工具读取音频文件
        # 支持 wav/mp3 等常见格式，自动重采样到 sample_rate(16kHz)
        # 返回值为 torch.Tensor 或 numpy 数组
        # ══════════════════════════════════════════════════════════════
        audio_data = load_audio_text_image_video(audio_path, fs=self.sample_rate)
        if isinstance(audio_data, torch.Tensor):
            audio_np = audio_data.numpy()  # 转为 numpy 数组以便后续切片操作
        else:
            audio_np = np.array(audio_data)
        speech_length = len(audio_np)  # 记录音频总长度（采样点数）

        # ══════════════════════════════════════════════════════════════
        # Step 2: VAD（语音活动检测）
        # VAD 模型的作用是将长音频切分为多个有语音的片段
        # 输出格式: vad_segments = [[start_ms, end_ms], ...]
        #   - start_ms: 该段语音的起始时间（毫秒）
        #   - end_ms: 该段语音的结束时间（毫秒）
        # 例如: [[0, 2500], [3200, 5800], [6100, 12000]]
        # 表示检测到 3 个语音片段
        #
        # 如果未配置 VAD 模型，则将整段音频作为一个片段处理
        # ══════════════════════════════════════════════════════════════
        if self.vad_model is not None:
            # 调用 VAD 模型进行语音活动检测
            # cache={} 表示不使用缓存，is_final=True 表示这是最终推理（非流式中间结果）
            vad_res = self.vad_model.generate(input=audio_path, cache={}, is_final=True)
            # 提取 VAD 分段列表：每个元素为 [起始时间ms, 结束时间ms]
            vad_segments = vad_res[0]["value"]  # [[start_ms, end_ms], ...]
        else:
            # 无 VAD 模型时，将整段音频作为单个片段处理
            # 将采样点总数转换为毫秒（/sample_rate * 1000）
            vad_segments = [[0, int(speech_length / self.sample_rate * 1000)]]

        # 如果没有检测到任何语音片段，直接返回空结果
        if not vad_segments:
            return {"key": key, "text": "", "timestamp": []}

        n_segments = len(vad_segments)
        logger.info(f"VAD: {n_segments} segments for {key}")

        # ══════════════════════════════════════════════════════════════
        # Step 3: 按 VAD 分段边界切分音频
        # 将完整音频按照 VAD 检测到的边界切分为多个独立片段
        # 每个片段对应一个语音活动区域，后续将分别送入 ASR 模型
        # ══════════════════════════════════════════════════════════════
        segment_audios = []
        for seg in vad_segments:
            # 将毫秒时间转换为采样点索引（乘以采样率 / 1000）
            start_sample = int(seg[0] * self.sample_rate / 1000)  # 起始采样点
            end_sample = int(seg[1] * self.sample_rate / 1000)    # 结束采样点
            # 确保不超出音频实际长度（防止 VAD 边界略微越界）
            end_sample = min(end_sample, speech_length)
            # 切片获取该段的音频数据（numpy 数组）
            segment_audios.append(audio_np[start_sample:end_sample])

        # ══════════════════════════════════════════════════════════════
        # Step 4: 批量 ASR 识别（核心步骤 - 使用 vLLM）
        #
        # 【关键设计】传统做法是对每个 VAD 片段单独调用 ASR 模型，
        # 而 vLLM 方案将所有片段的音频嵌入打包为 EmbedsPrompt 列表，
        # 在一次 generate() 调用中同时处理所有片段，实现真正的批量推理。
        #
        # 详细流程:
        #   (a) 对每个 VAD 音频片段:
        #       i.   调用 _encode_audio() 提取声学特征（通过 Audio Encoder）
        #       ii.  调用 _build_input_embeds() 构建完整的输入嵌入序列
        #            （包含音频嵌入 + 任务提示前缀等）
        #       iii. 包装为 EmbedsPrompt 对象
        #   (b) 配置 SamplingParams（采样参数）
        #   (c) 调用 vllm_engine.generate() 一次性生成所有片段的文本
        #   (d) 解码生成的 token IDs 为可读文本
        # ══════════════════════════════════════════════════════════════
        from vllm import SamplingParams
        try:
            # 尝试导入 EmbedsPrompt（vLLM 的嵌入提示输入类型）
            # EmbedsPrompt 允许直接传入预计算的 embedding 张量而非文本 token
            # 这是实现音频到文本的关键桥梁——我们将音频编码器的输出直接喂给 LLM
            from vllm.inputs import EmbedsPrompt
        except ImportError:
            # 兼容旧版 vLLM 的导入路径
            from vllm.inputs.data import EmbedsPrompt

        # 【构建 EmbedsPrompt 列表】
        # 每个 VAD 音频片段对应一个 EmbedsPrompt
        # EmbedsPrompt 内部包含 prompt_embeds: 一个形状为 [seq_len, hidden_dim] 的张量
        # 这个张量就是音频经过编码器处理后得到的"音频语言"表示
        prompts = []
        for seg_audio in segment_audios:
            # 将 numpy 音频数据转为 PyTorch 张量（float32）
            seg_tensor = torch.from_numpy(seg_audio).float()

            # (a-i) 音频编码：通过 Audio Encoder 提取声学特征
            # _encode_audio 内部流程:
            #   1. 提取 FBank 特征（梅尔滤波器组特征，80 维）
            #   2. 通过音频编码器（通常为 Conformer/Transformer）编码
            #   3. 返回编码器输出 (adaptor_out) 及其有效长度 (adaptor_out_lens)
            adaptor_out, adaptor_out_lens, _, _ = self.asr_engine._encode_audio(seg_tensor)

            # (a-ii) 构建完整的输入嵌入序列
            # _build_input_embeds 内部流程:
            #   1. 将音频编码器输出投影到 LLM 的隐藏维度空间
            #   2. 添加任务特定的前缀 token（如 <asr> 任务标记）
            #   3. 可选地插入热词（hotwords）嵌入以提升关键词识别率
            #   4. 可选地添加语言标记（language）和 ITN 指示
            # 返回值 input_embeds 形状为 [total_seq_len, hidden_dim]
            input_embeds = self.asr_engine._build_input_embeds(
                adaptor_out, adaptor_out_lens,
                hotwords=kwargs.get("hotwords"),       # 热词：提升特定词识别率
                language=kwargs.get("language"),        # 语言提示：指导模型选择语言
                itn=kwargs.get("itn", True),            # ITN 开关：是否做逆文本标准化
            )

            # (a-iii) 将输入嵌入包装为 EmbedsPrompt 对象
            # EmbedsPrompt 是 vLLM 的特殊输入类型，允许跳过 tokenization 步骤
            # 直接使用预计算的 embedding 作为 LLM 的输入
            # .float() 确保数据类型为 float32（vLLM 要求）
            prompts.append(EmbedsPrompt(prompt_embeds=input_embeds.float()))

        # 【配置采样参数 SamplingParams】
        # 控制 vLLM 生成文本时的行为策略
        params = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", 512),  # 每个片段最多生成 512 个 token
            temperature=0.0,          # 温度设为 0：贪婪解码（确定性输出，每次结果相同）
            repetition_penalty=1.3,   # 重复惩罚：降低重复生成的概率（>1.0 表示惩罚）
            skip_special_tokens=True, # 跳过特殊 token：输出中不包含 <eos>、<pad> 等
        )

        # 【核心调用：vLLM 批量生成】
        # 这里是整个流水线最关键的步骤——一次性处理所有 VAD 片段！
        #
        # vllm_engine.generate() 内部工作原理:
        #   1. 将所有 EmbedsPrompt padding 到相同长度（或使用 PagedAttention 动态管理）
        #   2. 在 GPU 上并行运行 LLM 前向传播（利用 tensor parallelism 多卡加速）
        #   3. 对每个 prompt 自回归生成文本 token，直到满足停止条件
        #   4. 返回每个 prompt 对应的生成结果（RequestOutput 对象列表）
        #
        # 性能优势:
        #   - 相比逐段处理，batch 推理可将 GPU 利用率提升 3-10 倍
        #   - PagedAttention 显存优化允许处理更多并发请求
        #   - Continuous Batching 动态调度进一步减少等待时间
        t0 = time.perf_counter()
        outputs = self.asr_engine.vllm_engine.generate(prompts, params, use_tqdm=False)
        t1 = time.perf_counter()
        logger.info(f"vLLM batch ASR: {n_segments} segments in {t1-t0:.3f}s")

        # 【解码生成结果】
        # 将 vLLM 输出的 RequestOutput 对象转换为可读文本
        asr_results = []
        for output in outputs:
            # output.outputs[0].text 是 vLLM 生成的文本字符串（已跳过特殊 token）
            text = output.outputs[0].text
            # 如果 text 为空但存在 token_ids（某些边缘情况），手动解码
            if not text and output.outputs[0].token_ids:
                text = self.asr_engine.tokenizer.decode(
                    list(output.outputs[0].token_ids), skip_special_tokens=True
                )
            # 清理文本：移除残留的标签、填充词等（调用 _clean_text 函数）
            text = _clean_text(text)
            asr_results.append(text)

        # ══════════════════════════════════════════════════════════════
        # Step 5: 说话人嵌入提取（可选步骤）
        # 如果配置了说话人模型（spk_model），则为每个 VAD 片段提取说话人特征
        #
        # 说话人分离（Speaker Diarization）的目标是回答："这段话是谁说的？"
        # 实现方式:
        #   1. 对每个音频片段提取说话人嵌入向量（speaker embedding）
        #      - 这是一个固定维度的特征向量（如 192 维），能够表征说话人的声音特征
        #      - 同一个人的不同 utterance 应产生相似的嵌入向量
        #   2. 对所有嵌入向量进行聚类（clustering）
        #      - 使用谱聚类（Spectral Clustering）等方法
        #      - 自动将相似的嵌入归为一组，每组对应一个说话人
        #   3. 为每个片段分配说话人标签（如 SPK001, SPK002, ...）
        # ══════════════════════════════════════════════════════════════
        spk_embeddings = None
        if self.spk_model is not None and kwargs.get("return_spk_res", True):
            from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk

            all_segments = []      # 存储所有切片后的子片段信息
            all_spk_embs = []      # 存储所有子片段的说话人嵌入

            for i, seg_audio in enumerate(segment_audios):
                # 构造当前 VAD 片段的元数据：[起始时间(s), 结束时间(s), 音频数据]
                # 注意这里时间单位从毫秒转换为秒（/1000.0）
                vad_seg = [
                    [vad_segments[i][0] / 1000.0, vad_segments[i][1] / 1000.0, seg_audio]
                ]

                # sv_chunk: 将 VAD 片段进一步切分为更短的子块（chunks）
                # 原因：说话人嵌入模型通常需要较短的输入窗口（如 1-3 秒）
                #       长片段会被切成多个重叠或不重叠的 chunk，分别提取嵌入后平均
                chunks = sv_chunk(vad_seg)
                all_segments.extend(chunks)  # 收集所有子块的时间信息

                # 提取各子块的纯音频数据
                speech_chunks = [c[2] for c in chunks]

                # 调用说话人模型提取嵌入向量
                # spk_model.generate() 返回每个 chunk 的说话人嵌入（spk_embedding 字段）
                spk_res = self.spk_model.generate(input=speech_chunks, cache={}, is_final=True)

                # 将当前片段所有 chunk 的嵌入拼接为一个张量
                embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
                all_spk_embs.append(embs)

            # 将所有 VAD 片段的说话人嵌入合并为一个大张量
            # 形状: [total_chunks, embed_dim]，后续用于聚类
            if all_spk_embs:
                spk_embeddings = torch.cat(all_spk_embs, dim=0)

        # ══════════════════════════════════════════════════════════════
        # Step 6: 合并结果
        # 将 VAD 分段、ASR 文本和时间戳信息整合为最终输出格式
        # ══════════════════════════════════════════════════════════════
        # 合并所有分段的文本为完整转录（用空格连接）
        full_text = ""
        all_timestamps = []
        for i, (seg, text) in enumerate(zip(vad_segments, asr_results)):
            if text:  # 只处理有有效文本的分段（跳过空文本）
                if full_text:
                    full_text += " "  # 分段之间用空格分隔
                full_text += text
                # 记录粗粒度时间戳（基于 VAD 边界的段级时间戳）
                # 格式: [起始时间(ms), 结束时间(ms)]
                all_timestamps.append([int(seg[0]), int(seg[1])])

        # 构建基础结果字典
        result = {"key": key, "text": full_text}

        # ══════════════════════════════════════════════════════════════
        # CTC 时间戳计算（可选增强）
        # 如果 ASR 引擎配备了 CTC 解码器，可以计算更精细的字/词级时间戳
        # 否则使用上述基于 VAD 边界的粗粒度时间戳
        #
        # CTC 时间戳的优势:
        #   - 可以精确到每个字/词的起止时间
        #   - 基于 CTC（Connectionist Temporal Classification）的对齐算法
        #   - 通过 forced alignment（强制对齐）将文本 token 映射回音频帧
        # ══════════════════════════════════════════════════════════════
        if self.asr_engine.ctc_decoder is not None:
            try:
                # 调用 CTC 时间戳计算方法，获取细粒度的 token 级时间戳
                detailed_timestamps = self._compute_all_timestamps(
                    segment_audios, vad_segments, asr_results
                )
                if detailed_timestamps:
                    result["timestamp"] = detailed_timestamps
            except Exception as e:
                # CTC 时间戳计算失败时降级为粗粒度 VAD 时间戳
                logger.debug(f"Timestamp computation failed: {e}")
                result["timestamp"] = all_timestamps
        else:
            # 无 CTC 解码器时直接使用 VAD 边界时间戳
            result["timestamp"] = all_timestamps

        # ══════════════════════════════════════════════════════════════
        # 说话人聚类与结果整合
        # 如果成功提取了说话人嵌入，执行聚类并为每个分段分配说话人标签
        #
        # 聚类流程:
        #   1. cb_model(spk_embeddings): 对所有说话人嵌入进行聚类
        #      - 使用谱聚类（Spectral Clustering）或 AHC（层次聚类）
        #      - oracle_num 参数允许预设说话人数量（如果已知）
        #      - 返回 labels: 每个chunk对应的聚类标签（整数ID）
        #   2. postprocess(): 后处理聚类结果
        #      - 平滑标签序列（去除短暂的误分类）
        #      - 为每个连续的同标签段分配统一的说话人标识（SPK001, SPK002...）
        #   3. distribute_spk(): 将说话人标签分配到句子级别
        #      - 将 chunk 级别的说话人标签聚合到 VAD 分段级别
        #      - 每个分段根据其包含的 chunks 的多数投票决定说话人
        # ══════════════════════════════════════════════════════════════
        if spk_embeddings is not None and self.cb_model is not None:
            from funasr.models.campplus.utils import postprocess, distribute_spk

            # 按时间排序所有子片段（确保时间顺序正确）
            all_segments_sorted = sorted(all_segments, key=lambda x: x[0])

            # 执行说话人聚类
            # 输入: 所有 chunk 的说话人嵌入张量 [N, embed_dim]
            # oracle_num: 可选的预设说话人数量（None 则自动估计）
            # 输出: labels 数组，每个元素是对应 chunk 的聚类 ID（0, 1, 2, ...）
            labels = self.cb_model(
                spk_embeddings.cpu(),  # 移至 CPU 进行聚类（通常聚类在 CPU 上更快）
                oracle_num=kwargs.get("preset_spk_num", None),
            )

            # 后处理：平滑标签并生成说话人分离输出
            sv_output = postprocess(all_segments_sorted, None, labels, spk_embeddings.cpu())

            # 构建句子级信息列表（sentence_info）
            # 每个 VAD 分段对应一个 sentence 条目
            sentence_list = []
            for i, (seg, text) in enumerate(zip(vad_segments, asr_results)):
                if text:
                    sentence_list.append({
                        "start": seg[0],                          # 分段起始时间(ms)
                        "end": seg[1],                            # 分段结束时间(ms)
                        "text": text,                             # 该分段的转录文本
                        "timestamp": [[int(seg[0]), int(seg[1])]], # 时间戳（嵌套列表格式）
                        # 注意：spk_label 将在下面的 distribute_spk 中填入
                    })

            # 将 chunk 级别的说话人标签分配到句子（VAD 分段）级别
            # distribute_spk 内部逻辑:
            #   - 根据 chunk 的时间范围与 VAD 分段的对应关系
            #   - 对每个 VAD 分段内的所有 chunks 进行多数投票
            #   - 将胜出的说话人标签写入 sentence_list 中对应条目的 "spk_label" 字段
            distribute_spk(sentence_list, sv_output)

            # 将带说话人信息的句子列表加入最终结果
            result["sentence_info"] = sentence_list

        return result

    def _compute_all_timestamps(self, segment_audios, vad_segments, asr_results):
        """Compute CTC timestamps for all segments with VAD offsets.
        （使用 CTC 强制对齐算法计算所有分段的精细时间戳）

        本方法为每个 VAD 分段的文本计算 token 级别的精确时间戳，
        基于 Connectionist Temporal Classification (CTC) 的 forced alignment 技术。

        【CTC 时间戳计算原理】

        CTC 是一种用于序列建模的技术，特别适合变长输入输出对齐问题。
        在 ASR 中，CTC 的关键特性是它天然提供了帧级别的对齐信息：

        1. CTC 前向传播过程:
           音频帧序列 (T 帧) → Audio Encoder → Encoder 输出 (T, hidden_dim)
           → CTC Projection Layer → CTC Logits (T, vocab_size)
           → Log Softmax → CTC 概率分布

        2. Forced Alignment（强制对齐）:
           给定已知文本（ASR 识别结果），找到每个 token 在音频中的最佳位置
           - 使用动态规划（类似 Viterbi 算法）在 CTC 概率矩阵中搜索最优路径
           - 最优路径上每个 token 对应的帧位置即为该 token 的时间戳

        3. 时间戳计算公式:
           token_time = frame_index * frame_shift * 10ms + vad_offset
           其中:
           - frame_index: CTC 对齐得到的帧索引
           - frame_shift: 下采样因子（本模型中为 6，即每 6 个音频帧对应 1 个 CTC 帧）
           - 10ms: 每个音频帧的时长（16kHz, 10ms hop = 160 samples）
           - vad_offset: 当前 VAD 分段在原始音频中的起始偏移量

        Args:
            segment_audios: VAD 分段的音频数据列表（numpy 数组）
            vad_segments: VAD 分段的时间边界列表 [[start_ms, end_ms], ...]
            asr_results: 对应的 ASR 识别文本列表

        Returns:
            精细时间戳列表，每个元素为一个字典:
            [{
                "token": "字/词",
                "start_time": 起始时间(秒),
                "end_time": 结束时间(秒),
            }, ...]
            所有分段的时间戳按时间顺序拼接在一起
        """
        from funasr.models.fun_asr_nano.tools.utils import forced_align

        all_timestamps = []

        # 逐个处理每个 VAD 分段
        for seg_audio, vad_seg, text in zip(segment_audios, vad_segments, asr_results):
            if not text:
                continue  # 跳过空文本分段

            try:
                # 准备音频输入张量
                seg_tensor = torch.from_numpy(seg_audio).float()

                # 提取 FBank 声学特征
                # extract_fbank 内部:
                #   1. 预加重（Pre-emphasis）：高频增强
                #   2. 分帧加窗（Framing & Windowing）：25ms 窗口，10ms 步进
                #   3. FFT → 功率谱 → Mel 滤波器组 → 80 维 FBank 特征
                # 返回:
                #   speech: [batch, time_steps, feat_dim] 特征张量
                #   speech_lengths: [batch] 有效长度
                from funasr.utils.load_utils import extract_fbank
                speech, speech_lengths = extract_fbank(
                    seg_tensor, data_type="sound",
                    frontend=self.asr_engine.frontend, is_final=True
                )
                # 将特征移至目标设备并确保 float32 精度
                speech = speech.to(self.device, dtype=torch.float32)
                speech_lengths = speech_lengths.to(self.device)

                # ════════════════════════════════════════════════════
                # CTC 前向传播：获取帧级别的 token 概率分布
                # ════════════════════════════════════════════════════
                with torch.no_grad():  # 不需要梯度（推理模式）
                    # (1) Audio Encoder: 将 FBank 特征编码为高层语义表示
                    # enc_out: [batch, T, encoder_dim] 编码器输出
                    # enc_lens: [batch] 编码后的有效长度（可能因下采样而缩短）
                    enc_out, enc_lens = self.asr_engine.audio_encoder(speech, speech_lengths)

                    # (2) CTC Decoder: 线性投影层，将编码器维度映射到词表大小
                    # dec_out: [batch, T, vocab_size] 每个 CTC 帧对词表中每个 token 的 logits
                    dec_out, dec_lens = self.asr_engine.ctc_decoder(enc_out, enc_lens)

                    # (3) Log Softmax: 将 logits 转换为 log 概率
                    # ctc_logits: [batch, T, vocab_size] 用于后续强制对齐
                    ctc_logits = self.asr_engine.ctc.log_softmax(dec_out)

                # 取出第一个（也是唯一的）样本的 CTC 概率矩阵
                # x 形状: [T, vocab_size]，T 是 CTC 帧数，vocab_size 是词表大小
                # 每一行代表一个时间步上所有 token 的 log 概率
                x = ctc_logits[0, :enc_lens[0].item(), :]

                # 将识别出的文本编码为 token ID 序列
                # target_ids: [L] ，L 是文本的 token 数量
                target_ids = torch.tensor(
                    self.asr_engine.ctc_tokenizer.encode(text), dtype=torch.int64
                )
                if len(target_ids) == 0:
                    continue  # 无法编码的文本（如全特殊字符）跳过

                # ════════════════════════════════════════════════════
                # Forced Alignment（强制对齐）：核心时间戳计算
                # ════════════════════════════════════════════════════
                # forced_align 函数内部实现:
                #   使用动态规划在 CTC 概率矩阵中搜索最优对齐路径
                #   类似于 Viterbi 算法，但考虑了 CTC 的 blank token 特性
                #
                # 算法思路:
                #   1. 定义状态: (token_index, time_frame) 表示第 i 个 token 对齐到第 t 帧
                #   2. 转移规则:
                #      - 停留在当前 token: (i, t) → (i, t+1)
                #      - 前进到下一 token: (i, t) → (i+1, t+1)
                #      - 经过 blank: (i, t) → (i, t+1) [blank 不消耗 token]
                #   3. 目标: 找到使总概率最大的路径
                #   4. 回溯得到每个 token 的起止帧位置
                #
                # 返回 timestamps: [{"token": id, "start_time": frame, "end_time": frame}, ...]
                timestamps = forced_align(x, target_ids, self.asr_engine.blank_id)

                # ════════════════════════════════════════════════════
                # 时间戳后处理：帧索引 → 实际时间（秒）
                # ════════════════════════════════════════════════════
                # 获取当前 VAD 分段在原始音频中的起始偏移量（毫秒转秒）
                vad_offset_ms = int(vad_seg[0])
                for ts in timestamps:
                    # 将 token ID 解码为可读字符/词
                    ts["token"] = self.asr_engine.ctc_tokenizer.decode([ts["token"]])

                    # 【时间转换公式详解】
                    # ts["start_time"] 目前是 CTC 帧索引（整数）
                    # 需要转换为实际时间（秒），公式如下:
                    #
                    # 实际时间 = CTC帧索引 × 下采样因子 × 帧时长 + VAD偏移
                    #          = frame_idx × 6 × 10ms + vad_offset_sec
                    #
                    # 其中:
                    #   - 6: 下采样因子（Audio Encoder 的下采样倍率）
                    #     Conformer 编码器通常有 4 层下采样，总下采样率为 2^4=16 或自定义
                    #     此处 6 是模型特定的有效下采样率
                    #   - 10ms: 每帧时长（16kHz 采样率，160 采样点/帧 = 10ms）
                    #   - vad_offset_ms / 1000: 当前分段在原始音频中的起始位置（秒）
                    #
                    # 示例:
                    #   CTC 帧索引=100, VAD 偏移=2000ms
                    #   → start_time = 100 * 6 * 10 / 1000 + 2000 / 1000
                    #   → start_time = 6.0 + 2.0 = 8.0 秒
                    ts["start_time"] = ts["start_time"] * 6 * 10 / 1000 + vad_offset_ms / 1000
                    ts["end_time"] = ts["end_time"] * 6 * 10 / 1000 + vad_offset_ms / 1000

                # 将当前分段的时间戳追加到全局列表
                all_timestamps.extend(timestamps)

            except Exception as e:
                # CTC 时间戳计算失败时的降级处理
                # 使用 VAD 边界作为粗粒度时间戳（整段一个时间戳）
                logger.debug(f"Timestamp failed for segment: {e}")
                all_timestamps.append({
                    "start_time": vad_seg[0] / 1000,  # 起始时间（秒）
                    "end_time": vad_seg[1] / 1000,    # 结束时间（秒）
                    "token": text,                     # 整段文本作为一个 "token"
                })
        return all_timestamps

    @classmethod
    def from_pretrained(cls, model="FunAudioLLM/Fun-ASR-Nano-2512", **kwargs):
        """Convenience constructor.
        （便捷构造函数：从预训练模型创建流水线实例）

        提供类似 HuggingFace from_pretrained 风格的工厂方法，
        使得用户可以通过一行代码快速创建流水线实例。

        Args:
            model: 预训练模型名称或路径
            **kwargs: 传递给 __init__ 的其他参数

        Returns:
            初始化完成的 FunASRNanoVLLMPipeline 实例

        Example:
            pipeline = FunASRNanoVLLMPipeline.from_pretrained(
                "FunAudioLLM/Fun-ASR-Nano-2512",
                vad_model="fsmn-vad",
                spk_model="cam++",
            )
        """
        return cls(model=model, **kwargs)
