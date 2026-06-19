#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Fun-ASR-Nano vLLM 推理引擎模块

本模块实现了基于 vLLM 的高性能语音识别推理引擎。vLLM 是一个高吞吐量的 LLM 推理框架，
本模块利用 vLLM 进行语言模型解码，同时保持音频编码器和适配器使用 PyTorch 运行。

主要特性:
    - 支持批量推理，提高吞吐量
    - 支持张量并行(tensor-parallel)进行多 GPU 加速
    - 支持 CTC 强制对齐计算时间戳
    - 支持热词(hotwords)提升识别准确率

架构说明:
    Audio -> WavFrontend -> SenseVoiceEncoder -> AudioAdaptor -> 音频嵌入
    文本 tokens -> LLM embedding layer -> 文本嵌入
    合并嵌入 -> vLLM (Qwen3-0.6B) -> 生成文本

Usage:
    from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

    # 从预训练模型加载
    engine = FunASRNanoVLLM.from_pretrained(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        tensor_parallel_size=2,  # 使用 2 个 GPU 进行张量并行
    )

    # 批量推理
    results = engine.generate(["audio1.wav", "audio2.wav"])
    for r in results:
        print(f"识别结果: {r['text']}")
"""

# ===== 标准库导入 =====
import glob          # 文件路径匹配（用于查找 safetensors 文件）
import json          # JSON 序列化（用于保存模型索引文件）
import logging       # 日志记录
import os            # 操作系统接口（文件/目录操作）
import re            # 正则表达式（用于清理生成文本中的伪影）
import shutil        # 高级文件操作（用于复制配置文件）
import time          # 时间测量（用于性能统计）
from typing import List, Optional, Union  # 类型注解

# ===== 第三方库导入 =====
import numpy as np   # 数值计算（用于音频数组处理）
import torch         # PyTorch 深度学习框架
import torch.nn as nn  # PyTorch 神经网络模块（用于创建嵌入层）

# 配置模块级日志记录器
logger = logging.getLogger(__name__)

# 数据类型映射表：将字符串标识转换为 PyTorch 数据类型
# - "bf16": Brain Floating Point 16，训练和推理常用
# - "fp16": IEEE 754 Half Precision，推理加速常用
# - "fp32": IEEE 754 Single Precision，最高精度
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def prepare_vllm_model_dir(model_dir: str, output_dir: str = None) -> str:
    """准备 vLLM 模型目录：从 Fun-ASR-Nano 的 model.pt 中提取 LLM 权重并保存为 HuggingFace 格式。

    Fun-ASR-Nano 将所有权重（音频编码器 + 适配器 + LLM）存储在单个 model.pt 文件中。
    vLLM 需要 LLM 权重采用标准的 HuggingFace 格式。此函数从 model.pt 中提取 LLM 权重，
    并与 Qwen3-0.6B 子目录中的配置/分词器文件一起保存。

    Args:
        model_dir (str): Fun-ASR-Nano 模型目录的路径
        output_dir (str, optional): 提取的 LLM 保存位置。默认为 model_dir/Qwen3-0.6B-vllm

    Returns:
        str: 包含 vLLM 就绪的 LLM 模型的目录路径

    Raises:
        FileNotFoundError: 如果找不到 Qwen3-0.6B 配置目录或 model.pt 文件
        RuntimeError: 如果在 model.pt 中找不到 LLM 权重
    """
    # 设置默认输出目录
    if output_dir is None:
        output_dir = os.path.join(model_dir, "Qwen3-0.6B-vllm")

    # 检查是否已经准备过（避免重复处理，提高效率）
    safetensors_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    bin_files = glob.glob(os.path.join(output_dir, "model*.bin"))
    if safetensors_files or bin_files:
        logger.info(f"vLLM 模型已准备就绪: {output_dir}")
        return output_dir

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 从 Qwen3-0.6B 目录复制配置和分词器文件
    # 这些文件包含 tokenizer、config.json 等 vLLM 所需的元数据
    qwen_dir = os.path.join(model_dir, "Qwen3-0.6B")
    if not os.path.isdir(qwen_dir):
        raise FileNotFoundError(f"Qwen3-0.6B 配置目录未找到: {qwen_dir}")

    # 复制所有配置文件（仅当目标文件不存在时）
    for fname in os.listdir(qwen_dir):
        src = os.path.join(qwen_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # 加载 model.pt 并提取 LLM 权重
    model_pt = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(
            f"model.pt 未找到: {model_pt}。请确保模型已完全下载。"
        )

    logger.info(f"正在加载 model.pt: {model_pt}...")
    # 加载检查点到 CPU 以节省 GPU 内存（模型文件通常很大）
    checkpoint = torch.load(model_pt, map_location="cpu")
    # 获取 state_dict（兼容不同格式的检查点）
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 提取 LLM 权重（以 "llm." 为前缀的参数）
    # Fun-ASR-Nano 的 state_dict 包含 audio_encoder.*, audio_adaptor.*, llm.* 等前缀
    llm_state = {}
    for key, value in state_dict.items():
        if key.startswith("llm."):
            # 移除 "llm." 前缀，得到标准的 HuggingFace 参数名
            new_key = key[len("llm."):]
            llm_state[new_key] = value

    if not llm_state:
        raise RuntimeError("在 model.pt 中未找到 LLM 权重（期望前缀 'llm.'）")

    logger.info(f"已提取 {len(llm_state)} 个 LLM 权重张量")

    # 保存为 safetensors 格式（vLLM 推荐格式，加载更快、更安全）
    try:
        from safetensors.torch import save_file

        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(llm_state, save_path)
        logger.info(f"已保存 LLM 权重到: {save_path}")

        # 创建模型索引文件（支持多分片模型加载）
        index = {
            "metadata": {
                "total_size": sum(v.numel() * v.element_size() for v in llm_state.values())
            },
            "weight_map": {k: "model.safetensors" for k in llm_state.keys()},
        }
        index_path = os.path.join(output_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    except ImportError:
        # 如果 safetensors 未安装，回退到 PyTorch 的 pickle 格式
        save_path = os.path.join(output_dir, "model.bin")
        torch.save(llm_state, save_path)
        logger.info(f"已保存 LLM 权重到: {save_path}（安装 safetensors 可获得更快加载速度）")

    return output_dir


class FunASRNanoVLLM:
    """Fun-ASR-Nano vLLM 推理引擎类

    使用 vLLM 作为后端的高性能语音识别推理引擎。结合了 PyTorch 音频处理和 vLLM 的高效 LLM 推理能力。

    架构说明:
        音频 -> WavFrontend -> SenseVoiceEncoder -> AudioAdaptor -> 音频嵌入
        文本 tokens -> LLM embedding layer -> 文本嵌入
        合并嵌入 -> vLLM (Qwen3-0.6B) -> 生成文本

    工作流程:
        1. 音频编码器和适配器在单个 GPU 上使用 PyTorch 运行
        2. vLLM 处理 LLM 推理，支持可选的张量并行
        3. 支持批量推理，提高整体吞吐量

    Args:
        model_dir (str): Fun-ASR-Nano 模型目录的路径
        device (str): 音频编码器/适配器使用的设备（如 "cuda:0"）
        dtype (str): 音频处理的数据类型（"bf16", "fp16", "fp32"）
        tensor_parallel_size (int): vLLM 张量并行使用的 GPU 数量
        gpu_memory_utilization (float): vLLM KV 缓存使用的 GPU 内存比例（0-1）
        max_model_len (int): vLLM 的最大序列长度
        enforce_eager (bool): 是否禁用 CUDA graph（用于调试）

    Example:
        >>> engine = FunASRNanoVLLM(
        ...     model_dir="/path/to/Fun-ASR-Nano-2512",
        ...     tensor_parallel_size=2,  # 使用 2 个 GPU
        ... )
        >>> results = engine.generate(["audio1.wav", "audio2.wav"])
        >>> for r in results:
        ...     print(f"识别结果: {r['text']}")
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        dtype: str = "bf16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 2048,
        enforce_eager: bool = False,
        **kwargs,
    ):
        """初始化 FunASRNanoVLLM 推理引擎

        初始化过程包括四个主要步骤：
        1. 准备 vLLM 模型目录（从 model.pt 提取 LLM 权重）
        2. 加载音频组件（编码器 + 适配器 + 前端）
        3. 初始化 vLLM 引擎
        4. 加载分词器和 LLM 嵌入层
        """
        # 导入 vLLM 相关模块
        from vllm import LLM, SamplingParams
        try:
            from vllm.inputs import EmbedsPrompt
        except ImportError:
            # 兼容旧版本 vLLM
            from vllm.inputs.data import EmbedsPrompt

        # 保存配置参数
        self.device = device
        self.dtype = dtype
        self.torch_dtype = dtype_map.get(dtype, torch.bfloat16)  # 字符串转 PyTorch dtype
        self.model_dir = model_dir

        # 步骤 1: 准备 vLLM 模型目录（如果需要，从 model.pt 提取 LLM 权重）
        vllm_model_dir = prepare_vllm_model_dir(model_dir)

        # 步骤 2: 加载音频组件（编码器 + 适配器 + 前端）
        self._load_audio_components(model_dir, **kwargs)

        # 步骤 3: 初始化 vLLM 引擎
        logger.info(f"正在初始化 vLLM，模型路径: {vllm_model_dir}")
        logger.info(f"  张量并行大小: {tensor_parallel_size}")
        logger.info(f"  GPU 内存利用率: {gpu_memory_utilization}")

        # 获取额外的 vLLM 参数
        vllm_kwargs = kwargs.get("vllm_kwargs", {})
        
        # 创建 vLLM 引擎实例
        self.vllm_engine = LLM(
            enable_prompt_embeds=True,  # 启用嵌入提示功能（音频嵌入需要）
            model=vllm_model_dir,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "auto"}.get(dtype, dtype),
            trust_remote_code=True,  # 信任远程代码（Qwen3 需要此选项）
            **vllm_kwargs,
        )

        # 步骤 4: 获取分词器和 LLM 嵌入层
        self.tokenizer = self.vllm_engine.get_tokenizer()
        self._load_embedding_layer(model_dir)

    def _load_audio_components(self, model_dir: str, **kwargs):
        """加载音频组件：从检查点加载音频编码器、适配器、前端和 CTC 模块

        此方法负责加载所有音频处理相关的组件：
        - 音频前端（WavFrontend）：负责音频特征提取（如 Fbank）
        - 音频编码器（SenseVoiceEncoder）：将音频特征编码为高维表示
        - 音频适配器（AudioAdaptor）：将编码器输出转换为 LLM 可接受的嵌入
        - CTC 解码器（可选）：用于计算时间戳
        """
        # 导入配置管理工具和模型注册表
        from omegaconf import OmegaConf
        from funasr.register import tables

        # 加载配置文件
        config_path = os.path.join(model_dir, "config.yaml")
        config = OmegaConf.load(config_path)
        self._config = OmegaConf.to_container(config, resolve=True)

        # ========== 音频前端 ==========
        # 前端负责从原始音频提取特征（如 Fbank 特征）
        frontend_class = tables.frontend_classes.get(config["frontend"])
        frontend_conf = OmegaConf.to_container(config.get("frontend_conf", {}), resolve=True)
        # 处理 CMVN 文件路径（如果使用相对路径，转换为绝对路径）
        cmvn_file = frontend_conf.get("cmvn_file")
        if cmvn_file and not os.path.isabs(cmvn_file):
            frontend_conf["cmvn_file"] = os.path.join(model_dir, cmvn_file)
        self.frontend = frontend_class(**frontend_conf)
        self.frontend.eval()  # 设置为评估模式

        # ========== 音频编码器 ==========
        encoder_conf = OmegaConf.to_container(config.get("audio_encoder_conf", {}), resolve=True)
        hub = encoder_conf.get("hub", None)
        if hub == "ms":
            from funasr import AutoModel as FunAutoModel

            enc_model = FunAutoModel(
                model=config["audio_encoder"], model_revision="master", disable_update=True
            )
            self.audio_encoder_output_size = (
                enc_model.model.encoder_output_size
                if hasattr(enc_model.model, "encoder_output_size")
                else -1
            )
            self.audio_encoder = (
                enc_model.model.model.encoder
                if hasattr(enc_model.model, "model")
                else enc_model.model.encoder
            )
        else:
            encoder_class = tables.encoder_classes.get(config["audio_encoder"])
            input_size = self.frontend.output_size()
            self.audio_encoder = encoder_class(input_size=input_size, **encoder_conf)
            self.audio_encoder_output_size = self.audio_encoder.output_size()

        self.audio_encoder.eval()
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        # ========== 音频适配器 ==========
        # 适配器将编码器输出转换为 LLM 可接受的嵌入维度
        adaptor_conf = OmegaConf.to_container(config.get("audio_adaptor_conf", {}), resolve=True)
        adaptor_class = tables.adaptor_classes.get(config["audio_adaptor"])
        # 设置编码器维度
        if self.audio_encoder_output_size > 0:
            adaptor_conf["encoder_dim"] = self.audio_encoder_output_size
        self.audio_adaptor = adaptor_class(**adaptor_conf)
        self.audio_adaptor.eval()  # 设置为评估模式
        # 冻结适配器参数
        for p in self.audio_adaptor.parameters():
            p.requires_grad = False
        # 是否使用低帧率模式（影响时间戳计算）
        self.use_low_frame_rate = adaptor_conf.get("use_low_frame_rate", False)

        # ========== CTC 解码器（可选，用于时间戳） ==========
        # CTC 解码器用于计算字符级时间戳
        self.ctc_decoder = None
        self.ctc = None
        self.ctc_tokenizer = None
        self.blank_id = None

        ctc_decoder_name = self._config.get("ctc_decoder", None)
        if ctc_decoder_name:
            ctc_decoder_class = tables.adaptor_classes.get(ctc_decoder_name)
            ctc_decoder_conf = self._config.get("ctc_decoder_conf", {})
            if self.audio_encoder_output_size > 0:
                ctc_decoder_conf["encoder_dim"] = self.audio_encoder_output_size
            self.ctc_decoder = ctc_decoder_class(**ctc_decoder_conf)
            self.ctc_decoder.eval()
            for p in self.ctc_decoder.parameters():
                p.requires_grad = False

            from funasr.models.fun_asr_nano.ctc import CTC

            ctc_conf = self._config.get("ctc_conf", {})
            ctc_vocab_size = self._config.get("ctc_vocab_size", 60515)
            self.blank_id = ctc_conf.get("blank_id", ctc_vocab_size - 1)
            self.ctc = CTC(
                odim=ctc_vocab_size,
                encoder_output_size=self.audio_encoder_output_size,
                blank_id=self.blank_id,
                **ctc_conf,
            )

            # 加载 CTC 分词器（用于时间戳解码）
            ds_conf = self._config.get("dataset_conf", {})
            ctc_tokenizer_name = ds_conf.get("ctc_tokenizer", None)
            ctc_tokenizer_conf = ds_conf.get("ctc_tokenizer_conf", {})
            if ctc_tokenizer_name:
                ctc_tokenizer_class = tables.tokenizer_classes.get(ctc_tokenizer_name)
                vocab_path = ctc_tokenizer_conf.get("vocab_path")
                # 处理词表路径
                if vocab_path is None or not os.path.isabs(vocab_path):
                    # 尝试查找多语言词表
                    multilingual_path = os.path.join(model_dir, "multilingual.tiktoken")
                    if os.path.exists(multilingual_path):
                        ctc_tokenizer_conf["vocab_path"] = multilingual_path
                    elif vocab_path and not os.path.isabs(vocab_path):
                        ctc_tokenizer_conf["vocab_path"] = os.path.join(model_dir, vocab_path)
                self.ctc_tokenizer = ctc_tokenizer_class(**ctc_tokenizer_conf)

        # --- Load weights from model.pt ---
        model_pt = os.path.join(model_dir, "model.pt")
        if os.path.exists(model_pt):
            logger.info(f"Loading audio component weights from {model_pt}")
            checkpoint = torch.load(model_pt, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Audio encoder
            enc_state = {
                k[len("audio_encoder."):]: v
                for k, v in state_dict.items()
                if k.startswith("audio_encoder.")
            }
            if enc_state:
                self.audio_encoder.load_state_dict(enc_state, strict=False)
                logger.info(f"  Loaded audio_encoder: {len(enc_state)} params")

            # Audio adaptor
            adp_state = {
                k[len("audio_adaptor."):]: v
                for k, v in state_dict.items()
                if k.startswith("audio_adaptor.")
            }
            if adp_state:
                self.audio_adaptor.load_state_dict(adp_state, strict=False)
                logger.info(f"  Loaded audio_adaptor: {len(adp_state)} params")

            # CTC decoder
            if self.ctc_decoder is not None:
                ctc_dec_state = {
                    k[len("ctc_decoder."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("ctc_decoder.")
                }
                if ctc_dec_state:
                    self.ctc_decoder.load_state_dict(ctc_dec_state, strict=False)
                ctc_state = {
                    k[len("ctc."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("ctc.") and not k.startswith("ctc_decoder.")
                }
                if ctc_state:
                    self.ctc.load_state_dict(ctc_state, strict=False)

        # Move to device
        self.audio_encoder = self.audio_encoder.to(self.device, dtype=torch.float32)
        self.audio_adaptor = self.audio_adaptor.to(self.device, dtype=self.torch_dtype)
        if self.ctc_decoder is not None:
            self.ctc_decoder = self.ctc_decoder.to(self.device, dtype=torch.float32)
            self.ctc = self.ctc.to(self.device, dtype=torch.float32)

    def _load_embedding_layer(self, model_dir: str):
        """加载 LLM 嵌入层，用于计算文本 token 的嵌入表示

        嵌入层将文本 token ID 转换为稠密向量表示，用于与音频嵌入拼接后输入 LLM。
        """
        # 加载模型检查点
        model_pt = os.path.join(model_dir, "model.pt")
        checkpoint = torch.load(model_pt, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # 查找嵌入层权重（以 "llm." 开头，包含 "embed_tokens.weight"）
        embed_key = None
        for key in state_dict.keys():
            if "embed_tokens.weight" in key and key.startswith("llm."):
                embed_key = key
                break

        if embed_key is None:
            raise RuntimeError("在 model.pt 中未找到 LLM 嵌入权重")

        # 创建嵌入层并加载权重
        embed_weight = state_dict[embed_key]
        # freeze=True 表示嵌入层权重在推理时不会更新
        self.embed_tokens = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        # 移动到指定设备和数据类型
        self.embed_tokens = self.embed_tokens.to(self.device, dtype=self.torch_dtype)
        logger.info(f"已加载嵌入层，形状: {embed_weight.shape}")

    @torch.no_grad()  # 推理时禁用梯度计算，节省内存
    def _encode_audio(self, audio_input: Union[str, torch.Tensor, np.ndarray]):
        """通过前端 -> 编码器 -> 适配器编码音频

        Returns:
            tuple: (adaptor_out, adaptor_out_lens, encoder_out, encoder_out_lens)
                - adaptor_out: (1, T', D_llm) 适配器输出的音频嵌入
                - adaptor_out_lens: (1,) 有效长度
                - encoder_out: (1, T, D_enc) 编码器输出（用于 CTC 时间戳）
                - encoder_out_lens: (1,) 编码器输出长度
        """
        # 导入音频加载工具
        from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

        # 根据输入类型加载音频数据
        if isinstance(audio_input, str):
            # 从文件路径加载音频
            data_src = load_audio_text_image_video(audio_input, fs=self.frontend.fs)
        elif isinstance(audio_input, np.ndarray):
            # 从 NumPy 数组转换
            data_src = torch.from_numpy(audio_input).float()
        elif isinstance(audio_input, torch.Tensor):
            # 直接使用张量
            data_src = audio_input.float()
        else:
            raise ValueError(f"不支持的音频输入类型: {type(audio_input)}")

        # 提取 Fbank 特征
        speech, speech_lengths = extract_fbank(
            data_src, data_type="sound", frontend=self.frontend, is_final=True
        )
        # 将特征移动到指定设备
        speech = speech.to(self.device, dtype=torch.float32)
        speech_lengths = speech_lengths.to(self.device)

        # 通过音频编码器编码
        encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)
        # 转换为适配器所需的数据类型
        encoder_out_for_adaptor = encoder_out.to(dtype=self.torch_dtype)
        # 通过适配器转换为 LLM 嵌入
        adaptor_out, adaptor_out_lens = self.audio_adaptor(encoder_out_for_adaptor, encoder_out_lens)

        # 低帧率模式：计算有效 token 数量
        # 这个计算公式与 PyTorch model.py 中的 data_load_speech 完全一致
        if self.use_low_frame_rate:
            for i in range(adaptor_out.shape[0]):
                fbank_len = speech_lengths[i].item()
                # 两层卷积下采样（stride=2）
                olens = 1 + (fbank_len - 3 + 2 * 1) // 2
                olens = 1 + (olens - 3 + 2 * 1) // 2
                # 适配器下采样
                fake_token_len = (olens - 1) // 2 + 1
                adaptor_out_lens[i] = fake_token_len

        return adaptor_out, adaptor_out_lens, encoder_out, encoder_out_lens

    def _build_prompt_text(
        self,
        hotwords: List[str] = None,
        language: str = None,
        itn: bool = True,
    ) -> str:
        """构建 ASR 提示文本

        Args:
            hotwords: 热词列表，用于提升特定词汇的识别准确率
            language: 目标语言（如 "中文", "英文", "日文"）
            itn: 是否进行逆文本归一化（ITN）

        Returns:
            str: 构建好的提示文本
        """
        hotwords = hotwords or []
        # 构建热词提示（如果提供）
        if len(hotwords) > 0:
            hotwords_str = ", ".join(hotwords)
            prompt = (
                "请结合上下文信息，更加准确地完成语音转写任务。"
                "如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            )
            prompt += f"热词列表：[{hotwords_str}]\n"
        else:
            prompt = ""
        if language is None:
            prompt += "语音转写"
        else:
            prompt += f"语音转写成{language}"
        if not itn:
            prompt += "，不进行文本规整"
        return prompt + "："

    @torch.no_grad()
    def _build_input_embeds(
        self,
        audio_embeds: torch.Tensor,
        audio_embed_lens: torch.Tensor,
        hotwords: List[str] = None,
        language: str = None,
        itn: bool = True,
        system_prompt: str = "You are a helpful assistant.",
    ) -> torch.Tensor:
        """Build the full input embedding sequence with audio inserted.

        Returns:
            Tensor of shape (seq_len, D_llm)
        """
        prompt = self._build_prompt_text(hotwords, language, itn)

        # ChatML format with speech markers and thinking prefix
        prefix_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|startofspeech|>"
        )
        suffix_text = "<|endofspeech|><|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)

        # Embed text tokens
        prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=self.device)
        suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=self.device)
        prefix_embeds = self.embed_tokens(prefix_tensor)
        suffix_embeds = self.embed_tokens(suffix_tensor)

        # Audio embeddings
        audio_len = audio_embed_lens[0].item()
        audio_emb = audio_embeds[0, :audio_len, :]

        # Concat: [prefix_text_emb | audio_emb | suffix_text_emb]
        inputs_embeds = torch.cat([prefix_embeds, audio_emb, suffix_embeds], dim=0)
        return inputs_embeds

    def generate(
        self,
        inputs: Union[str, List[str], np.ndarray, torch.Tensor, List],
        hotwords: List[str] = None,
        language: str = None,
        itn: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> List[dict]:
        """使用 vLLM 运行批量 ASR 推理

        这是主要的推理入口方法。处理流程：
        1. 批量编码音频 -> 音频嵌入
        2. 构建嵌入提示（文本 + 音频）
        3. vLLM 批量生成
        4. 解码并清理结果

        Args:
            inputs: 音频输入，支持：
                - str: 单个文件路径
                - List[str]: 批量文件路径
                - np.ndarray / torch.Tensor: 原始音频采样（16kHz）
            hotwords: 热词列表，提升识别准确率
            language: 语言提示（如 "中文", "英文", "日文"）
            itn: 是否应用逆文本归一化（默认 True）
            max_new_tokens: 每个样本最大生成 token 数
            temperature: 采样温度（0 = 贪心解码）
            top_p: 核采样参数
            top_k: Top-k 采样（-1 = 禁用）
            repetition_penalty: 重复惩罚因子

        Returns:
            List[dict]: 结果列表: [{"key": str, "text": str, "timestamps": [...]}]
        """
        # 导入 vLLM 采样参数
        from vllm import SamplingParams
        try:
            from vllm.inputs import EmbedsPrompt
        except ImportError:
            from vllm.inputs.data import EmbedsPrompt

        if isinstance(inputs, (str, np.ndarray, torch.Tensor)):
            inputs = [inputs]

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=True,
        )

        # Batch encode audio and build embedding prompts
        prompts = []
        encoder_outputs = []

        t0 = time.perf_counter()

        # Pre-compute text embeddings (shared across batch)
        prompt_text = self._build_prompt_text(hotwords, language, itn)
        prefix_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}"
        suffix_text = "<|im_end|>\n<|im_start|>assistant\n"
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)
        prefix_emb = self.embed_tokens(torch.tensor(prefix_ids, dtype=torch.long, device=self.device))
        suffix_emb = self.embed_tokens(torch.tensor(suffix_ids, dtype=torch.long, device=self.device))

        # Batch encode audio (groups of 8 for memory efficiency)
        batch_size_enc = 8
        all_adaptor_outs = []
        all_adaptor_lens = []
        for i in range(0, len(inputs), batch_size_enc):
            batch_inputs = inputs[i:i+batch_size_enc]
            # Load and extract fbank for batch
            from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
            audio_tensors = []
            for audio_input in batch_inputs:
                if isinstance(audio_input, str):
                    data_src = load_audio_text_image_video(audio_input, fs=self.frontend.fs)
                elif isinstance(audio_input, np.ndarray):
                    data_src = torch.from_numpy(audio_input).float()
                elif isinstance(audio_input, torch.Tensor):
                    data_src = audio_input.float()
                else:
                    raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
                audio_tensors.append(data_src)

            speech, speech_lengths = extract_fbank(
                audio_tensors, data_type="sound", frontend=self.frontend, is_final=True
            )
            speech = speech.to(self.device, dtype=torch.float32)
            speech_lengths = speech_lengths.to(self.device)

            with torch.no_grad():
                enc_out, enc_lens = self.audio_encoder(speech, speech_lengths)
                adp_out, adp_lens = self.audio_adaptor(enc_out.to(dtype=self.torch_dtype), enc_lens)

            # Apply low frame rate token length correction
            if self.use_low_frame_rate:
                for j in range(len(batch_inputs)):
                    fbank_len = speech_lengths[j].item()
                    olens = 1 + (fbank_len - 3 + 2 * 1) // 2
                    olens = 1 + (olens - 3 + 2 * 1) // 2
                    adp_lens[j] = (olens - 1) // 2 + 1

            for j in range(len(batch_inputs)):
                all_adaptor_outs.append(adp_out[j, :adp_lens[j], :])
                all_adaptor_lens.append(adp_lens[j])
                encoder_outputs.append((enc_out[j:j+1, :enc_lens[j], :], enc_lens[j:j+1]))

        # Build prompts
        for audio_emb in all_adaptor_outs:
            input_embeds = torch.cat([prefix_emb, audio_emb, suffix_emb], dim=0)
            prompts.append(EmbedsPrompt(prompt_embeds=input_embeds.float()))

        t1 = time.perf_counter()
        logger.info(f"Audio encoding: {len(inputs)} samples in {t1 - t0:.3f}s")

        # vLLM batch generation
        outputs = self.vllm_engine.generate(prompts, sampling_params, use_tqdm=len(inputs) > 1)

        t2 = time.perf_counter()
        logger.info(f"vLLM generation: {t2 - t1:.3f}s")

        # Process results
        results = []
        for i, output in enumerate(outputs):
            token_ids = list(output.outputs[0].token_ids)
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            # Clean vLLM artifacts: remove garbage prefix/tags
            text = re.sub(r'<[^>]*>', '', text)
            text = re.sub(r'\[[^\]]*\]', '', text)
            text = re.sub(r'endofpatch|/sil|FFFF|</strong>', '', text)
            # Strip non-CJK/non-alnum prefix garbage
            text = re.sub(r'^[^\w一-鿿]+', '', text)
            text_clean = re.sub(r"\s+", " ", text).strip()

            key = (
                os.path.splitext(os.path.basename(inputs[i]))[0]
                if isinstance(inputs[i], str)
                else f"sample_{i}"
            )
            result = {"key": key, "text": text_clean}

            # Timestamps via CTC forced alignment
            if self.ctc_decoder is not None and self.ctc_tokenizer is not None:
                try:
                    timestamps = self._compute_timestamps(
                        encoder_outputs[i][0], encoder_outputs[i][1], text_clean
                    )
                    if timestamps:
                        result["timestamps"] = timestamps
                except Exception as e:
                    logger.debug(f"Timestamp computation failed for {key}: {e}")

            results.append(result)

        return results

    @torch.no_grad()
    def _compute_timestamps(self, encoder_out, encoder_out_lens, text):
        """通过 CTC 强制对齐计算字符级时间戳

        Args:
            encoder_out: 编码器输出张量
            encoder_out_lens: 编码器输出长度
            text: 识别出的文本

        Returns:
            list: 时间戳列表，每个元素包含 token, start_time, end_time
        """
        # 导入强制对齐工具
        from funasr.models.fun_asr_nano.tools.utils import forced_align

        decoder_out, decoder_out_lens = self.ctc_decoder(encoder_out, encoder_out_lens)
        ctc_logits = self.ctc.log_softmax(decoder_out)
        x = ctc_logits[0, : encoder_out_lens[0].item(), :]

        target_ids = torch.tensor(self.ctc_tokenizer.encode(text), dtype=torch.int64)
        if len(target_ids) == 0:
            return []

        timestamps = forced_align(x, target_ids, self.blank_id)
        for ts in timestamps:
            ts["token"] = self.ctc_tokenizer.decode([ts["token"]])
            ts["start_time"] = ts["start_time"] * 6 * 10 / 1000
            ts["end_time"] = ts["end_time"] * 6 * 10 / 1000
        return timestamps

    @classmethod
    def from_pretrained(
        cls,
        model: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        hub: str = "ms",
        device: str = "cuda:0",
        dtype: str = "bf16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 2048,
        **kwargs,
    ) -> "FunASRNanoVLLM":
        """从预训练模型加载（便捷入口方法）

        支持从 ModelScope 或 HuggingFace Hub 下载模型，也可直接指定本地目录。

        Args:
            model: 模型名称或本地目录路径
            hub: 模型仓库（"ms" = ModelScope, "hf" = HuggingFace）
            device: 音频编码器/适配器使用的设备
            dtype: 计算数据类型（"bf16", "fp16", "fp32"）
            tensor_parallel_size: vLLM 张量并行 GPU 数量
            gpu_memory_utilization: GPU 内存利用率（0-1）
            max_model_len: 最大序列长度

        Returns:
            FunASRNanoVLLM: 初始化好的推理引擎实例
        """
        # 判断是本地路径还是远程模型名
        if os.path.isdir(model):
            model_dir = model
        else:
            # 从远程仓库下载模型
            if hub in ("ms", "modelscope"):
                from modelscope.hub.snapshot_download import snapshot_download

                model_dir = snapshot_download(model, revision=kwargs.pop("revision", "master"))
            elif hub in ("hf", "huggingface"):
                from huggingface_hub import snapshot_download

                model_dir = snapshot_download(model)
            else:
                raise ValueError(f"不支持的 hub: {hub}。请使用 'ms' 或 'hf'。")

        logger.info(f"模型目录: {model_dir}")
        return cls(
            model_dir=model_dir,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **kwargs,
        )
