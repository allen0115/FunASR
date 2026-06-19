"""
FunASR Nano 模型实现

这是一个端到端的语音识别大模型，基于音频编码器 + 音频适配器 + LLM 的架构。
支持多语言、热词定制、字符级时间戳等功能。

主要组件:
- Audio Encoder: 音频编码器（如 Conformer），提取音频特征
- Audio Adaptor: 音频适配器，将音频特征投影到 LLM 维度
- LLM: 大型语言模型，进行多模态理解和文本生成
- CTC Decoder: （可选）CTC 解码器，用于字符级时间戳对齐
"""

import logging
import os
import random
import re
import string
import time
import traceback
from typing import Union

import torch
import torch.nn as nn

from funasr.metrics.compute_acc import compute_accuracy
from funasr.register import tables
from funasr.train_utils.device_funcs import force_gatherable, to_device
from funasr.utils.datadir_writer import DatadirWriter
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
try:
    # 尝试导入 HuggingFace Transformers 库，用于加载预训练语言模型
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError:
    # 如果未安装 Transformers，设置为 None
    AutoConfig = None
    AutoModelForCausalLM = None

from .ctc import CTC
from .tools.utils import forced_align

# 数据类型映射表：将字符串标识符映射到 PyTorch 的 dtype
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@tables.register("model_classes", "FunASRNano")
class FunASRNano(nn.Module):
    """Fun-ASR-Nano: End-to-End ASR Large Model.

    Trained on tens of millions of hours of real speech data.
    Supports 31 languages including Chinese dialects and regional accents.

    Features:
    - Character-level timestamps (via CTC forced alignment)
    - Hotword customization
    - Speaker diarization (when combined with spk_model)
    - Lyrics and rap recognition
    - Streaming chunk-by-chunk inference (demo2.py)

    Output: {"key": ..., "text": ..., "timestamps": [{"token", "start_time", "end_time"}, ...],
             "ctc_timestamps": [...]}

    Note: Outputs punctuation natively — punc_model is NOT needed.

    Requirements: pip install tiktoken huggingface_hub
    
    架构说明:
    - Audio Encoder: 音频编码器（如 Conformer），提取音频特征
    - Audio Adaptor: 音频适配器，将音频特征投影到 LLM 维度
    - LLM: 大型语言模型，进行多模态理解和文本生成
    - CTC Decoder: （可选）CTC 解码器，用于字符级时间戳对齐
    """

    def __init__(
        self,
        audio_encoder: str = None,
        audio_encoder_conf: dict = None,
        audio_adaptor: str = None,
        audio_adaptor_conf: dict = None,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        length_normalized_loss: bool = False,
        **kwargs,
    ):
        """Initialize FunASRNano.
        
            Args:
                audio_encoder: 音频编码器类型名称。
                audio_encoder_conf: 音频编码器的配置字典。
                audio_adaptor: 音频适配器类型名称。
                audio_adaptor_conf: 音频适配器的配置字典。
                llm: 语言模型类型名称。
                llm_conf: 语言模型的配置字典。
                input_size: 输入特征维度（默认 80，对应 FBank 特征）。
                length_normalized_loss: 是否对损失进行长度归一化。
                **kwargs: 其他关键字参数（包含 CTC、tokenizer 等配置）。
            """
        super().__init__()

        # ==================== 1. 音频编码器 (Audio Encoder) ====================
        # 从配置中获取模型仓库来源（如 "ms" 表示 ModelScope）
        hub = audio_encoder_conf.get("hub", None)
        # 是否启用激活检查点（Activation Checkpointing），用于节省显存
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get(
            "activation_checkpoint", False
        )
        
        if hub == "ms":
            # 如果从 ModelScope 加载，使用 AutoModel 自动构建模型
            from funasr import AutoModel

            model = AutoModel(model=audio_encoder, model_revision="master")
            # 获取编码器输出维度
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )
            # 提取实际的编码器模块
            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )
        else:
            # 否则从注册表中获取编码器类并实例化
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        
        # 是否冻结音频编码器参数（预训练模型通常冻结）
        freeze = audio_encoder_conf.get("freeze", True)

        if freeze:
            # 冻结所有参数，不参与梯度更新
            for _, param in audio_encoder.named_parameters():
                param.requires_grad = False
            # 设置为评估模式
            audio_encoder.eval()
        
        # 保存音频编码器实例
        self.audio_encoder = audio_encoder

        # ==================== 2. 大型语言模型 (LLM) ====================
        self.llm = None
        # 获取 LLM 初始化参数路径（包含模型配置文件）
        init_param_path = llm_conf.get("init_param_path", None)
        llm_dim = None  # LLM 的隐藏层维度

        # 获取加载模型的额外参数（如 trust_remote_code=True）
        llm_load_kwargs = llm_conf.get("load_kwargs", {})
        # 从配置文件加载模型配置
        config = AutoConfig.from_pretrained(init_param_path)
        # 根据配置创建因果语言模型（Causal LM）
        model = AutoModelForCausalLM.from_config(config, **llm_load_kwargs)

        # 是否冻结 LLM 参数
        freeze = llm_conf.get("freeze", True)
        if freeze:
            # 冻结所有 LLM 参数
            for _, param in model.named_parameters():
                param.requires_grad = False
            model.eval()
        
        # 是否启用梯度检查点（Gradient Checkpointing），用计算换显存
        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

        # 获取 LLM 的数据类型（fp16/bf16/fp32）
        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        # 将模型转换到指定数据类型
        self.llm = model.to(dtype_map[self.llm_dtype])
        # 获取 LLM 输入嵌入层的维度（即隐藏层大小）
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        # ==================== 3. 音频适配器 (Audio Adaptor) ====================
        # 从注册表中获取适配器类
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        
        # 如果音频编码器输出维度已知，传递给适配器配置
        if audio_encoder_output_size > 0:
            audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        
        # 设置 LLM 的维度（如果之前未获取到，则使用配置中的默认值）
        audio_adaptor_conf["llm_dim"] = (
            llm_dim if llm_dim is not None else audio_adaptor_conf["llm_dim"]
        )
        
        # 实例化音频适配器
        audio_adaptor = adaptor_class(**audio_adaptor_conf)
        
        # 是否冻结适配器参数
        freeze = audio_adaptor_conf.get("freeze", False)
        if freeze:
            for _, param in audio_adaptor.named_parameters():
                param.requires_grad = False
            audio_adaptor.eval()
        
        # 保存适配器实例
        self.audio_adaptor = audio_adaptor
        # 是否使用低帧率模式（降低音频特征的时间分辨率）
        self.use_low_frame_rate = audio_adaptor_conf.get("use_low_frame_rate", False)

        # ==================== 4. CTC 解码器 (CTC Decoder, 可选) ====================
        # CTC 解码器用于生成字符级时间戳（通过强制对齐）
        self.ctc_decoder = None
        # TODO: 修复表名（应该是 ctc_decoder_classes）
        ctc_decoder_class = tables.adaptor_classes.get(kwargs.get("ctc_decoder", None))
        
        if ctc_decoder_class is not None:
            # 获取 CTC tokenizer 及其配置
            ctc_tokenizer = (
                kwargs.get("ctc_tokenizer", None)
                if "ctc_tokenizer" in kwargs
                else kwargs["dataset_conf"]["ctc_tokenizer"]
            )
            ctc_tokenizer_conf = (
                kwargs.get("ctc_tokenizer_conf", None)
                if "ctc_tokenizer_conf" in kwargs
                else kwargs["dataset_conf"]["ctc_tokenizer_conf"]
            )
            
            # 如果提供了 tokenizer 配置，实例化 tokenizer
            if ctc_tokenizer is not None and ctc_tokenizer_conf is not None:
                ctc_tokenizer_class = tables.tokenizer_classes.get(ctc_tokenizer)
                ctc_tokenizer = ctc_tokenizer_class(**ctc_tokenizer_conf)
                self.ctc_tokenizer = ctc_tokenizer
            
            # 确保 tokenizer 已设置
            assert ctc_tokenizer is not None, f"ctc_tokenizer must be set"

            # 获取 CTC 词表大小（默认 60515）
            ctc_vocab_size = kwargs.get("ctc_vocab_size", 60515)
            # 获取 CTC 解码器配置
            ctc_decoder_conf = kwargs.get("ctc_decoder_conf", {})
            
            # 如果音频编码器输出维度已知，传递给 CTC 解码器
            if audio_encoder_output_size > 0:
                ctc_decoder_conf["encoder_dim"] = audio_encoder_output_size
            
            # 实例化 CTC 解码器
            self.ctc_decoder = ctc_decoder_class(**ctc_decoder_conf)
            
            # 如果有预训练权重路径，加载权重
            init_param_path = ctc_decoder_conf.get("init_param_path", None)
            if init_param_path is not None:
                src_state = torch.load(init_param_path, map_location="cpu")
                flag = self.ctc_decoder.load_state_dict(src_state, strict=False)
                logging.info(f"Loading ctc_decoder ckpt: {init_param_path}, status: {flag}")
            
            # 是否冻结 CTC 解码器参数
            freeze = ctc_decoder_conf.get("freeze", False)
            if freeze:
                for _, param in self.ctc_decoder.named_parameters():
                    param.requires_grad = False
                self.ctc_decoder.eval()

            # 获取 CTC 相关配置
            ctc_conf = kwargs.get("ctc_conf", {})
            # blank token 的 ID（默认为词表最后一个）
            self.blank_id = ctc_conf.get("blank_id", ctc_vocab_size - 1)
            # CTC 损失在总损失中的权重（默认 0.3）
            self.ctc_weight = kwargs.get("ctc_weight", 0.3)
            
            # 实例化 CTC 模块（用于计算 CTC 损失）
            self.ctc = CTC(
                odim=ctc_vocab_size,
                encoder_output_size=audio_encoder_output_size,
                blank_id=self.blank_id,
                **ctc_conf,
            )
            
            # 是否在计算 CTC 损失时detach梯度（防止梯度回传到编码器）
            self.detach_ctc_decoder = kwargs.get("detach_ctc_decoder", True)
            # 错误计算器（用于评估）
            self.error_calculator = None

        # 是否对损失进行长度归一化
        self.length_normalized_loss = length_normalized_loss
        # 获取当前进程的 rank（分布式训练时使用）
        rank = int(os.environ.get("RANK", 0))
        # 打印模型构建完成日志
        logging.info(f"rank: {rank}, model is builded.")

    def forward(
        self,
        speech: torch.Tensor = None,
        speech_lengths: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels_ids: torch.Tensor = None,
        fbank_beg: torch.Tensor = None,
        fbank_mask: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass for training.
        
            Args:
                speech: 语音音频张量，形状为 (batch, time)。
                speech_lengths: 每个语音样本的长度。
                input_ids: 输入 token IDs，形状为 (batch, seq_len)。
                attention_mask: 注意力掩码，1 表示有效位置，0 表示填充。
                labels_ids: 标签 token IDs，用于计算损失。
                fbank_beg: 每轮对话中音频特征应该插入的起始位置（token 索引）。
                fbank_mask: 音频特征的掩码，1 表示音频位置，0 表示文本位置。
                **kwargs: 其他关键字参数（包含 fake_token_len 等）。
            
            Returns:
                loss: 标量损失值。
                stats: 统计信息字典（包含准确率、batch size 等）。
                weight: batch size（用于数据并行）。
            """
        # 获取 batch size 和 token 数量
        batch_size, token_num = input_ids.shape
        stats = {}
        
        # 将无效的 token ID（< 0）替换为 0
        input_ids[input_ids < 0] = 0
        # 通过 LLM 的嵌入层将 token IDs 转换为嵌入向量
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        
        if speech is not None:
            # 如果提供了语音数据，处理音频特征
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]
            batch_size_speech, frames, _ = speech.shape

            # ==================== 音频编码器 ====================
            if self.audio_encoder_activation_checkpoint:
                # 如果启用激活检查点，使用 checkpoint 包装以减少显存占用
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
                # 否则直接调用编码函数
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # ==================== 音频适配器 ====================
            # 将音频特征从编码器维度投影到 LLM 维度
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

            # 获取 inputs_embeds 的形状
            batch_size, token_num, dims = inputs_embeds.shape
            # 获取每个音频段的 fake token 长度（占位符数量）
            fake_token_len = kwargs.get("fake_token_len")
            # 将无效值（< 0）替换为 0
            fake_token_len[fake_token_len < 0] = 0
            fbank_beg[fbank_beg < 0] = 0

            # 全局音频段计数器（encoder_out 是扁平化的所有音频段）
            speech_idx = 0
            
            # ==================== 双层循环：将音频特征注入到文本嵌入中 ====================
            # 遍历每个样本中的每轮对话，将音频特征覆盖到对应的文本嵌入位置
            for batch_idx in range(batch_size):
                for turn_id in range(fbank_beg.shape[1]):
                    # 获取当前轮次音频应该插入的起始位置（token 索引）
                    fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                    # fbank_beg_idx > 0 表示该位置有音频需要插入，= 0 表示该轮次无音频
                    if fbank_beg_idx > 0:
                        # 获取这段音频特征的长度（token 数量）
                        speech_token_len = fake_token_len[batch_idx, turn_id]
                        # 从 encoder 输出中提取对应的音频特征 (speech_token_len, dim)
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]

                        try:
                            # 核心操作：将音频特征覆盖到 inputs_embeds 的指定位置
                            # 相当于用音频特征"替换"掉原来占位的文本嵌入
                            inputs_embeds[
                                batch_idx,
                                fbank_beg_idx : fbank_beg_idx + speech_token_len,
                                :,
                            ] = speech_token
                        except Exception as e:
                            # 如果插入失败（如位置越界），记录错误信息和调试日志
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            # 降级处理：使用 encoder 实际输出长度作为音频特征长度
                            speech_token_len = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :speech_token_len, :]
                            inputs_embeds[
                                batch_idx,
                                fbank_beg_idx : fbank_beg_idx + speech_token_len,
                                :,
                            ] = speech_token

                        # 全局音频段计数器递增（encoder_out 是扁平化的，需要追踪当前处理到第几个音频段）
                        speech_idx += 1

            # 记录统计信息
            stats["batch_size_speech"] = batch_size_speech
            stats["batch_size_x_frames"] = frames * batch_size_speech
            stats["batch_size_real_frames"] = speech_lengths.sum().item()
            stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]

        # ==================== 混合精度训练 & LLM 前向传播 ====================
        # 获取当前模型所在设备类型（cuda/cpu/xpu/mps），用于后续混合精度计算
        device_type = next(self.parameters()).device.type
        
        # 使用 torch.autocast 进行混合精度训练，减少显存占用并加速计算
        with torch.autocast(
            # 仅在 GPU/XPU/MPS 上启用 autocast，CPU 上回退到 "cpu"
            device_type=device_type if device_type in ["cuda", "xpu", "mps"] else "cpu",
            # 只有当 llm_dtype 不是 fp32 时才启用混合精度（fp16/bf16 时启用）
            enabled=True if self.llm_dtype != "fp32" else False,
            # 指定混合精度使用的数据类型（fp16 或 bf16）
            dtype=dtype_map[self.llm_dtype],
        ):
            # 将标签中的 -1（填充/无效位置）替换为 -100，PyTorch CrossEntropyLoss 会忽略 -100 的位置
            labels_ids[labels_ids == -1] = -100
            # 将注意力掩码中的负值（无效位置）置为 0
            attention_mask[attention_mask < 0] = 0
            
            # 将混合了音频和文本的嵌入输入 LLM，计算生成损失
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                attention_mask=attention_mask,
                labels=labels_ids,
            )
            # 提取 LLM 计算的交叉熵损失
            loss = model_outputs.loss

        # ==================== 计算准确率（不记录梯度） ====================
        with torch.no_grad():
            # 对 LLM 输出的 logits 取 argmax，得到预测的 token ID
            preds = torch.argmax(model_outputs.logits, -1)
            # 计算准确率：预测 preds[:,:-1] vs 真实标签 labels_ids[:,1:]（偏移一位，因为是下一个 token 预测）
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc"] = acc_att

        # 记录损失值
        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        # 记录 token 统计信息
        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        # 统计对话轮次信息
        dialog_turns = (fbank_beg > 0).sum(-1)  # 每个样本的对话轮次数
        dialog_turns_max = torch.max(dialog_turns).int().item()  # 最大轮次数
        dialog_turns_avg = dialog_turns.sum().item() / batch_size  # 平均轮次数
        stats["dialog_turns_max"] = dialog_turns_max
        stats["dialog_turns_avg"] = dialog_turns_avg

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        # 如果是长度归一化损失，重新计算 batch_size
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
        # 将损失、统计信息和权重转换为适合数据并行的格式
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def forward_export(self, speech, speech_lengths, **kwargs):
        """Forward export.
        用于模型导出时的前向传播（仅包含音频编码和适配，不包含 LLM）。
        
            Args:
                speech: 语音音频张量，形状为 (batch, time)。
                speech_lengths: 每个语音样本的长度。
                **kwargs: 其他关键字参数。
            
            Returns:
                encoder_out: 经过适配器后的音频特征。
                encoder_out_lens: 音频特征的长度。
            """
        # 音频编码器
        x, olens = self.audio_encoder(speech, speech_lengths)
        # 音频适配器
        encoder_out, encoder_out_lens = self.audio_adaptor(x, olens)
        return encoder_out, encoder_out_lens

    def encode(self, speech, speech_lengths):
        """Encode.
        对语音进行编码。
        
            Args:
                speech: 语音音频张量，形状为 (batch, time)。
                speech_lengths: 每个语音样本的长度。
            
            Returns:
                encoder_out: 编码器输出的特征。
                encoder_out_lens: 输出特征的长度。
            """
        # 调用音频编码器
        encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

    def data_template(self, data):
        """Data template.
        将原始对话数据转换为系统、用户、助手三个列表。
        
            Args:
                data: 原始对话数据，格式为 [{"role": ..., "content": ...}, ...]。
            
            Returns:
                contents: 包含 system、user、assistant 三个列表的字典。
            """
        # 初始化三个角色的内容列表
        system, user, assistant = [], [], []
        
        # 遍历每条消息，按角色分类
        for i, item in enumerate(data):
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                # 如果用户消息中包含音频，保留音频信息
                if "audio" in item:
                    audio = item["audio"]
                    content = [content, audio]
                user.append(content)
            elif role == "assistant":
                assistant.append(content)

        # 复制 system prompt 以匹配 user 的数量
        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

    def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):
        """Data load speech.
        加载语音数据并构建多轮对话的输入序列。
        
            Args:
                contents: 包含 system、user、assistant 内容的字典。
                tokenizer: 用于文本编码/解码的 tokenizer 实例。
                frontend: 用于音频特征提取的前端模块。
                meta_data: 元数据字典，用于记录处理时间等信息。
                **kwargs: 其他关键字参数（包含 multiturn_num_max、max_token_length 等）。
            
            Returns:
                output: 包含 speech、input_ids、attention_mask、labels_ids 等的字典。
            """
        # 提取三个角色的内容
        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        
        # 正则表达式：匹配音频占位符 <|startofspeech|>...<|endofspeech|>
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        
        # 配置参数
        do_think = True  # 是否添加 <think> 标签
        sys_prompt = True  # 是否使用 system prompt
        if "dataset_conf" in kwargs:
            do_think = kwargs["dataset_conf"].get("do_think", True)
            sys_prompt = kwargs["dataset_conf"].get("sys_prompt", True)

        # 初始化输出列表
        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [],  # 输入 token IDs
            [],  # 标签 token IDs
            [],  # 音频特征（FBank）
            [],  # 音频特征长度
            [],  # 音频特征掩码
            [],  # 音频插入位置
            [],  # fake token 长度
        )
        input_source_ids = []  # 仅包含 source 部分的 IDs（不含 target）
        
        # ==================== 遍历多轮对话 ====================
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            # 如果超过最大轮次限制，跳出循环
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            # 如果 token 数量超过限制，跳出循环
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            
            # 如果 user_prompt 是列表/元组，说明包含音频
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            
            # ==================== 构建 source input（系统 + 用户 prompt） ====================
            if i == 0:
                # 第一轮对话
                if kwargs.get("infer_with_assistant_input", False):
                    # 如果在推理时也输入 assistant 的内容（teacher forcing）
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}"
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    # 正常情况：不包含 assistant 的输出
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    if not sys_prompt:
                        source_input = (
                            f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                        )
            else:
                # 后续轮次对话
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
            
            # 如果不启用思考模式，添加空的 <think> 标签
            if not do_think:
                source_input += "<think>\n\n</think>\n\n"
            
            # 如果有前置文本，添加到 source input
            if kwargs.get("prev_text", None) is not None:
                source_input += kwargs["prev_text"]

            # ==================== 处理音频占位符 ====================
            # 使用正则表达式分割 source_input，分离文本和音频部分
            splits = pattern.split(source_input)
            source_ids = []  # 当前轮次的 source IDs
            fbank_mask_i = []  # 当前轮次的 FBank 掩码
            fake_token_len_i = 0  # 当前音频段的 fake token 长度
            fbank_beg_i = -1  # 当前音频段的插入位置
            speech, speech_lengths = [], []  # 当前轮次的音频数据
                        
            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    # 如果是普通文本，直接编码
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)  # 标记为文本位置
                else:
                    # 如果是音频占位符，提取音频路径并加载
                    sub_str = sub_str.replace("<|startofspeech|>, "").replace(
                        "<|endofspeech|>", ""
                    )
                    # 如果以 ! 开头，表示是音频样本点
                    if sub_str.startswith("!"):
                        sub_str = sub_str[1:]
                        if sub_str.startswith("!"):  # !!: audio sample point
                            sub_str = audio
                                
                    try:
                        # 记录加载音频的时间
                        time1 = time.perf_counter()
                        data_src = load_audio_text_image_video(
                            sub_str, fs=frontend.fs, **kwargs
                        )
                        time2 = time.perf_counter()
                        meta_data["load_data"] = f"{time2 - time1:0.3f}"
                    except Exception as e:
                        # 如果加载失败，记录错误
                        logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")
            
                    # 提取 FBank 特征
                    speech, speech_lengths = extract_fbank(
                        data_src,
                        data_type=kwargs.get("data_type", "sound"),
                        frontend=frontend,
                        is_final=True,
                    )  # speech: [b, T, d]
            
                    # 记录特征提取时间
                    time3 = time.perf_counter()
                    meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                    # 记录音频时长（秒）
                    meta_data["batch_data_time"] = (
                        speech_lengths.sum().item()
                        * frontend.frame_shift
                        * frontend.lfr_n
                        / 1000
                    )
            
                    # 计算 fake token 长度（音频特征的占位符数量）
                    if self.use_low_frame_rate:
                        # 低帧率模式：经过两次下采样
                        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                        olens = 1 + (olens - 3 + 2 * 1) // 2
                        fake_token_len_i = (olens - 1) // 2 + 1
                    else:
                        # 正常模式：直接使用音频帧数
                        fake_token_len_i = speech_lengths[0].item()
                                
                    # 创建 fake token（占位符）
                    fake_token = [0] * fake_token_len_i
                    # 记录音频应该插入的位置
                    fbank_beg_i = len(source_ids)
                    # 将 fake token 添加到 source_ids
                    source_ids += fake_token
                    # 标记这些位置为音频（1 表示音频）
                    fbank_mask_i += [1] * len(fake_token)

            # ==================== 合并 source 和 target ====================
            # 记录当前音频段的插入位置（考虑之前已累积的 input_ids 长度）
            fbank_beg += [fbank_beg_i + len(input_ids)]
            # 记录当前音频段的 fake token 长度
            fake_token_len += [fake_token_len_i]
            
            # 创建 source 部分的掩码（-100 表示计算损失时忽略）
            source_mask = [-100] * len(source_ids)
            
            # 处理 target 输出（添加结束标记）
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            
            # 保存仅包含 source 的 IDs（用于推理时的 teacher forcing）
            input_source_ids = input_ids + source_ids
            
            # 将 source 和 target 拼接到 input_ids
            input_ids += source_ids + target_ids
            # labels 中 source 部分为 -100（忽略），target 部分为真实标签
            labels += source_mask + target_ids
            # 合并 FBank 掩码
            fbank_mask += fbank_mask_i
            
            # 如果有音频数据，添加到列表
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        # ==================== 转换为 Tensor ====================
        # 将所有列表转换为 PyTorch Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

        fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        # 如果有音频数据，进行 padding
        if len(fbank) > 0:
            # 对音频特征进行 padding（batch_first=True，填充值为 0.0）
            speech = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True, padding_value=0.0)
            # 对音频长度进行 padding（填充值为 -1）
            speech_lengths = torch.nn.utils.rnn.pad_sequence(
                fbank_lens, batch_first=True, padding_value=-1
            )
        else:
            speech = []
            speech_lengths = []
        
        # 构建输出字典
        output = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "fbank_mask": fbank_mask[None, :],  # 增加 batch 维度
            "fbank_beg": fbank_beg[None,],  # 增加 batch 维度
            "fake_token_len": fake_token_len[None, :],  # 增加 batch 维度
            "input_ids": input_ids[None,],  # 增加 batch 维度
            "attention_mask": attention_mask[None,],  # 增加 batch 维度
            "labels_ids": labels,
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        return output

    def inference_prepare(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        """Inference prepare.
        为推理准备数据：加载音频、编码、适配，并将音频特征注入到文本嵌入中。
        
            Args:
                data_in: 输入数据（音频样本、文件路径或文本）。
                data_lengths: batch 中每个输入样本的长度。
                key: 样本标识符列表。
                tokenizer: 用于文本编码/解码的 tokenizer 实例。
                frontend: 用于音频特征提取的前端模块。
                **kwargs: 其他关键字参数。
            
            Returns:
                inputs_embeds: 混合了音频和文本的嵌入向量。
                contents: 对话内容字典。
                batch: 包含所有输入数据的批处理字典。
                source_ids: 仅包含 source 部分的 token IDs。
                meta_data: 元数据字典（包含中间结果）。
            """
        meta_data = {}

        # 目前不支持批量解码
        if len(data_in) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        # 将原始数据转换为 system/user/assistant 格式
        contents = self.data_template(data_in[0])
        # 加载语音数据并构建输入序列
        output = self.data_load_speech(contents, tokenizer, frontend, meta_data=meta_data, **kwargs)
        # 将数据移动到指定设备
        batch = to_device(output, kwargs["device"])

        # ==================== 音频编码 & 适配 ====================
        speech = batch["speech"]

        if len(speech) > 0:
            # 如果提供了预计算的音频嵌入，直接使用
            if "audio_embedding" in kwargs and "audio_embedding_lens" in kwargs:
                encoder_out = kwargs["audio_embedding"]
                encoder_out_lens = kwargs["audio_embedding_lens"]
            else:
                # 否则进行音频编码
                speech_lengths = batch["speech_lengths"][:, 0]
                
                # 根据配置转换数据类型
                if kwargs.get("fp16", False):
                    speech = speech.to(torch.float16)
                elif kwargs.get("bf16", False):
                    speech = speech.to(torch.bfloat16)
                
                # 音频编码器
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

                # 音频适配器
                adaptor_out, adaptor_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
                
                # 保存中间结果到 meta_data
                meta_data["encoder_out"] = encoder_out
                meta_data["encoder_out_lens"] = encoder_out_lens
                meta_data["audio_adaptor_out"] = adaptor_out
                meta_data["audio_adaptor_out_lens"] = adaptor_out_lens

        # 获取输入 IDs 和相关掩码
        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]

        # 如果不是 teacher forcing 模式，只使用 source_ids（不包含 target）
        if not kwargs.get("teacherforcing", False):
            input_ids = source_ids

        # 将无效的 token ID 替换为 0
        input_ids[input_ids < 0] = 0
        # 通过 LLM 的嵌入层将 token IDs 转换为嵌入向量
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

        batch_size, token_num, dims = inputs_embeds.shape

        # 将无效值替换为 0
        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        # ==================== 将音频特征注入到文本嵌入中 ====================
        speech_idx = 0
        for batch_idx in range(batch_size):
            for turn_id in range(fbank_beg.shape[1]):
                fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                if fbank_beg_idx > 0:
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    speech_token = adaptor_out[speech_idx, :speech_token_len, :]

                    try:
                        # 将音频特征覆盖到 inputs_embeds 的指定位置
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token
                    except Exception as e:
                        # 如果插入失败，记录错误并使用降级方案
                        logging.error(f"{str(e)}, {traceback.format_exc()}")
                        logging.info(
                            f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, adaptor_out: {adaptor_out.shape}, adaptor_out_lens: {adaptor_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                        )
                        speech_token_len = adaptor_out_lens[speech_idx].item()
                        speech_token = adaptor_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token

                    speech_idx += 1
        
        return inputs_embeds, contents, batch, source_ids, meta_data

    def get_prompt(self, hotwords: list[str], language: str = None, itn: bool = True):
        """Get prompt.
        根据热词、语言和 ITN 配置生成提示词。
        
            Args:
                hotwords: 热词列表，用于提高特定词汇的识别准确率。
                language: 目标语言标识符（如 "中文"、"英文"）。
                itn: 是否进行文本规整（Inverse Text Normalization）。
            
            Returns:
                prompt: 生成的提示词字符串。
            """
        # 如果有热词，添加到提示词中
        if len(hotwords) > 0:
            hotwords = ", ".join(hotwords)
            prompt = f"请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            prompt += f"热词列表：[{hotwords}]\n"
        else:
            prompt = ""
        
        # 添加语言信息
        if language is None:
            prompt += "语音转写"
        else:
            prompt += f"语音转写成{language}"
        
        # 如果不进行文本规整，添加说明
        if not itn:
            prompt += "，不进行文本规整"
        
        return prompt + "："

    def generate_chatml(self, prompt: str, data: Union[str, torch.Tensor]):
        """Generate chatml.
        生成 ChatML 格式的对话数据。
        
            Args:
                prompt: 提示词字符串。
                data: 输入数据，可以是音频文件路径（str）或音频张量（torch.Tensor）。
            
            Returns:
                chatml_data: ChatML 格式的对话列表，包含 system、user、assistant 三条消息。
            """
        if isinstance(data, str):
            # 如果 data 是字符串（音频文件路径）
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}<|startofspeech|>!{data}<|endofspeech|>"},
                {"role": "assistant", "content": "null"},
            ]
        elif isinstance(data, torch.Tensor):
            # 如果 data 是 Tensor（音频数据）
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"{prompt}<|startofspeech|>!!<|endofspeech|>",
                    "audio": data,
                },
                {"role": "assistant", "content": "null"},
            ]

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        """Run inference on input data.
        对输入数据进行推理（主入口）。
        
            Args:
                data_in: 输入数据（音频样本、文件路径或文本）。
                data_lengths: batch 中每个输入样本的长度。
                key: 样本标识符列表。
                tokenizer: 用于文本编码/解码的 tokenizer 实例。
                frontend: 用于音频特征提取的前端模块。
                **kwargs: 其他关键字参数（包含 hotwords、language、itn 等）。
            
            Returns:
                results: 推理结果列表，包含 text、timestamps 等信息。
                meta_data: 元数据字典。
            """
        # 生成提示词
        prompt = self.get_prompt(
            kwargs.get("hotwords", []), kwargs.get("language", None), kwargs.get("itn", True)
        )
        # 将输入数据转换为 ChatML 格式
        data_in = [self.generate_chatml(prompt, data) for data in data_in]

        # 如果没有提供 key，生成随机 key
        if key is None:
            key = []
            for _ in data_in:
                chars = string.ascii_letters + string.digits
                key.append("rand_key_" + "".join(random.choice(chars) for _ in range(13)))

        # 调用 LLM 推理方法
        return self.inference_llm(
            data_in,
            data_lengths=data_lengths,
            key=key,
            tokenizer=tokenizer,
            frontend=frontend,
            **kwargs,
        )

    def inference_llm(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        """Inference llm.
        使用 LLM 进行推理，生成转录文本和时间戳。
        
            Args:
                data_in: 输入数据（ChatML 格式的对话列表）。
                data_lengths: batch 中每个输入样本的长度。
                key: 样本标识符列表。
                tokenizer: 用于文本编码/解码的 tokenizer 实例。
                frontend: 用于音频特征提取的前端模块。
                **kwargs: 其他关键字参数（包含 max_length、fp16、bf16 等）。
            
            Returns:
                results: 推理结果列表，包含 text、text_tn、timestamps、ctc_timestamps 等。
                meta_data: 元数据字典。
            """
        # ==================== 准备输入数据 ====================
        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )

        # ==================== CTC 解码（可选，用于时间戳对齐） ====================
        ctc_results = []
        if self.ctc_decoder is not None:
            # 获取编码器输出
            encoder_out = meta_data["encoder_out"]
            encoder_out_lens = meta_data["encoder_out_lens"]
            
            # CTC 解码器前向传播
            decoder_out, decoder_out_lens = self.ctc_decoder(encoder_out, encoder_out_lens)
            # 计算 CTC log softmax
            ctc_logits = self.ctc.log_softmax(decoder_out)

            b, n, d = encoder_out.size()
            # 处理 key 的格式
            if isinstance(key[0], (list, tuple)):
                key = key[0]
            if len(key) < b:
                key = key * b
            
            # 对每个样本进行 CTC 解码
            for i in range(b):
                x = ctc_logits[i, : encoder_out_lens[i].item(), :]
                # 取 argmax 得到最可能的 token 序列
                yseq = x.argmax(dim=-1)
                # 去除连续重复的 token
                yseq = torch.unique_consecutive(yseq, dim=-1)
                # 过滤掉 blank token
                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                # 将 token IDs 解码为文本
                text = self.ctc_tokenizer.decode(token_int)
                ctc_results.append({"key": key[i], "text": text, "ctc_logits": x})

        # ==================== LLM 推理 ====================
        # 确定 LLM 的数据类型
        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        # 获取设备类型
        device_type = torch.device(kwargs.get("device", "cuda")).type
        
        # 使用混合精度进行推理
        with torch.autocast(
            device_type=device_type if device_type in ["cuda", "xpu", "mps"] else "cpu",
            enabled=True if llm_dtype != "fp32" else False,
            dtype=dtype_map[llm_dtype],
        ):
            # 获取真实标签（用于评估）
            label = contents["assistant"][-1]
            # 将 LLM 和 inputs_embeds 转换到指定数据类型
            self.llm = self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])
            
            # 获取 LLM 的额外参数
            llm_kwargs = kwargs.get("llm_kwargs", {})
            
            if not kwargs.get("teacherforcing", False):
                # ==================== 正常推理模式：使用 generate 生成文本 ====================
                attention_mask = batch.get("attention_mask", None)
                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=kwargs.get("max_length", 512),
                    pad_token_id=self.llm.config.pad_token_id or self.llm.config.eos_token_id,
                    **llm_kwargs,
                )

                # 解码生成的 token IDs 为文本
                response = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]

                loss = None
            else:
                # ==================== Teacher Forcing 模式：使用真实标签计算损失 ====================
                labels_ids = batch["labels_ids"]
                labels_ids[labels_ids == -1] = -100
                attention_mask = batch.get("attention_mask", None)
                model_outputs = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels_ids,
                    pad_token_id=self.llm.config.pad_token_id or self.llm.config.eos_token_id,
                    **llm_kwargs,
                )

                # 提取预测结果（跳过 source 部分）
                preds = torch.argmax(model_outputs.logits, -1)[:, source_ids.shape[1] :]
                response = tokenizer.batch_decode(
                    preds,
                    add_special_tokens=False,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]
                loss = model_outputs.loss.item()
        
        # 如果有前置文本，拼接到响应前面
        response = kwargs.get("prev_text", "") + response

        # ==================== 结果处理 & 输出 ====================
        # 初始化结果写入器（如果指定了输出目录）
        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

        results = []
        # 清理响应文本：移除标点符号，保留中文、英文、数字和空白
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        
        # 构建结果字典
        result_i = {
            "key": key[0],
            # 将 /sil 替换为空格，并合并多余空格
            "text": re.sub(r"\s+", " ", response.replace("/sil", " ")),
            # 清理后的文本（无标点）
            "text_tn": response_clean,
            # 真实标签
            "label": label,
        }
        if loss is not None:
            result_i["loss"] = loss
        results.append(result_i)

        # ==================== CTC 时间戳对齐 ====================
        # 如果有 CTC 结果，进行强制对齐以生成字符级时间戳
        for ctc_result, result in zip(ctc_results, results):
            # 获取 CTC 解码的文本
            result["ctc_text"] = ctc_result["text"].replace("<|nospeech|>", "")
            
            # 对 CTC 文本进行强制对齐
            target_ids = torch.tensor(
                self.ctc_tokenizer.encode(result["ctc_text"]), dtype=torch.int64
            )
            result["ctc_timestamps"] = forced_align(
                ctc_result["ctc_logits"], target_ids, self.blank_id
            )
            
            # 对 LLM 生成的文本进行强制对齐
            target_ids = torch.tensor(self.ctc_tokenizer.encode(result["text"]), dtype=torch.int64)
            result["timestamps"] = forced_align(ctc_result["ctc_logits"], target_ids, self.blank_id)
            
            # 转换时间戳格式：将 token ID 转换为字符，并计算实际时间（秒）
            for timestamps in [result["timestamps"], result["ctc_timestamps"]]:
                for timestamp in timestamps:
                    # 将 token ID 解码为字符
                    timestamp["token"] = self.ctc_tokenizer.decode([timestamp["token"]])
                    # 将帧索引转换为时间（秒）：frame_idx * 6 * 10 / 1000
                    timestamp["start_time"] = timestamp["start_time"] * 6 * 10 / 1000
                    timestamp["end_time"] = timestamp["end_time"] * 6 * 10 / 1000

        # 如果指定了输出目录，写入结果文件
        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response.replace("\n", " ")
            ibest_writer["label"][key[0]] = label.replace("\n", " ")
            ibest_writer["text_tn"][key[0]] = response_clean

        return results, meta_data

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        """From pretrained.
        从预训练模型加载 FunASRNano 模型。
        
            Args:
                model: 模型实例或模型名称（如 ModelScope/HuggingFace 上的模型 ID）。
                **kwargs: 其他关键字参数（传递给 AutoModel.build_model）。
            
            Returns:
                model: 加载的 FunASRNano 模型实例。
                kwargs: 模型配置字典。
            """
        from funasr import AutoModel

        # 使用 AutoModel 自动构建和加载模型
        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)

        return model, kwargs
