#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import types
import torch
from funasr.utils.torch_function import sequence_mask


def export_rebuild_model(model, **kwargs):
    """重建模型用于 ONNX 导出。

    将模型的 forward 方法替换为导出友好的版本，并绑定导出相关的元数据方法。

    Args:
        model: 原始 SenseVoiceSmall 模型实例。
        **kwargs: 导出配置，需包含 device 和 max_seq_len。

    Returns:
        重建后的模型，支持 ONNX 导出。
    """
    model.device = kwargs.get("device")
    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)
    model.forward = types.MethodType(export_forward, model)
    model.export_dummy_inputs = types.MethodType(export_dummy_inputs, model)
    model.export_input_names = types.MethodType(export_input_names, model)
    model.export_output_names = types.MethodType(export_output_names, model)
    model.export_dynamic_axes = types.MethodType(export_dynamic_axes, model)
    model.export_name = types.MethodType(export_name, model)
    return model

def export_forward(
    self,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    language: torch.Tensor,
    textnorm: torch.Tensor,
    **kwargs,
):
    """ONNX 导出专用的前向传播。

    与标准 forward 的区别：
    - 直接接收语言和文本归一化 token ID（而非从 text 中解析）
    - 不计算损失，直接输出 CTC logits
    - 不使用 SpecAugment 和归一化

    Args:
        speech (torch.Tensor): 音频特征，形状为 (batch, time, feat_dim)。
        speech_lengths (torch.Tensor): 每个样本的有效长度。
        language (torch.Tensor): 语言 token ID。
        textnorm (torch.Tensor): 文本归一化风格 token ID。

    Returns:
        tuple: (ctc_logits, encoder_out_lens)
            - ctc_logits: CTC 输出 logits，形状为 (batch, time, vocab_size)。
            - encoder_out_lens: 编码器输出的有效长度。
    """
    language_query = self.embed(language.to(speech.device)).unsqueeze(1)
    textnorm_query = self.embed(textnorm.to(speech.device)).unsqueeze(1)
    
    speech = torch.cat((textnorm_query, speech), dim=1)
    
    event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
        speech.size(0), 1, 1
    )
    input_query = torch.cat((language_query, event_emo_query), dim=1)
    speech = torch.cat((input_query, speech), dim=1)
    
    speech_lengths_new = speech_lengths + 4
    encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths_new)
    
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]

    ctc_logits = self.ctc.ctc_lo(encoder_out)
    
    return ctc_logits, encoder_out_lens

def export_dummy_inputs(self):
    """Export dummy inputs."""
    speech = torch.randn(2, 30, 560)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    language = torch.tensor([0, 0], dtype=torch.int32)
    textnorm = torch.tensor([15, 15], dtype=torch.int32)
    return (speech, speech_lengths, language, textnorm)

def export_input_names(self):
    """Export input names."""
    return ["speech", "speech_lengths", "language", "textnorm"]

def export_output_names(self):
    """Export output names."""
    return ["ctc_logits", "encoder_out_lens"]

def export_dynamic_axes(self):
    """Export dynamic axes."""
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {0: "batch_size"},
        "language": {0: "batch_size"},
        "textnorm": {0: "batch_size"},
        "ctc_logits": {0: "batch_size", 1: "logits_length"},
        "encoder_out_lens":  {0: "batch_size"},
    }

def export_name(self):
    """Export name."""
    return "model.onnx"
