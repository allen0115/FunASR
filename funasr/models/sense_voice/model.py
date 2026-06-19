from typing import Iterable, Optional
import types
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.cuda.amp import autocast
from funasr.metrics.compute_acc import compute_accuracy, th_accuracy
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.train_utils.device_funcs import force_gatherable

from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.ctc.ctc import CTC

from funasr.register import tables


from funasr.models.paraformer.search import Hypothesis
from .utils.ctc_alignment import ctc_forced_align


class SinusoidalPositionEncoder(torch.nn.Module):
    """正弦位置编码器。

    使用正弦和余弦函数生成位置编码，为序列中的每个位置生成唯一的向量表示。
    与 Transformer 原论文中的位置编码方式一致。
    """

    def __init__(self, d_model=80, dropout_rate=0.1):
        super().__init__()

    def encode(
        self, positions: torch.Tensor = None, depth: int = None, dtype: torch.dtype = torch.float32
    ):
        """将位置索引编码为正弦位置向量。

        Args:
            positions (torch.Tensor): 位置索引张量，形状为 (batch_size, seq_len)。
            depth (int): 编码维度（必须为偶数）。
            dtype (torch.dtype): 输出数据类型。

        Returns:
            torch.Tensor: 位置编码，形状为 (batch_size, seq_len, depth)。
        """
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (
            depth / 2 - 1
        )
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        """将正弦位置编码加到输入张量上。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, timesteps, input_dim)。

        Returns:
            torch.Tensor: 添加位置编码后的张量，形状与输入相同。
        """
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)

        return x + position_encoding


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        """前向 FSMN（Feedforward Sequential Memory Network）模块。

        FSMN 是一种轻量级的序列建模模块，通过 1D 卷积捕获局部上下文信息。
        残差连接确保梯度流畅传播。

        Args:
            inputs (torch.Tensor): 输入张量，形状为 (batch, time, dim)。
            mask (torch.Tensor, optional): 输入掩码。
            mask_shfit_chunk (torch.Tensor, optional): 流式推理时的块移位掩码。

        Returns:
            torch.Tensor: FSMN 输出，形状与输入相同。
        """
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """SANM 多头注意力前向传播。

        融合标准多头注意力和 FSMN 模块的输出，实现全局和局部信息的互补。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch, time, size)。
            mask (torch.Tensor): 输入掩码。
            mask_shfit_chunk (torch.Tensor, optional): 流式推理时的块移位掩码。
            mask_att_chunk_encoder (torch.Tensor, optional): 编码器注意力掩码。

        Returns:
            torch.Tensor: 注意力输出 + FSMN 输出，形状为 (batch, time, d_model)。
        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """流式推理时的分块前向传播。

        支持增量式推理，通过 KV 缓存复用历史计算结果，提高流式推理效率。

        Args:
            x (torch.Tensor): 当前块的输入张量。
            cache (dict, optional): KV 缓存字典，包含历史的 key 和 value。
            chunk_size (tuple, optional): 分块大小配置。
            look_back (int): 回看块数，-1 表示全局回看。

        Returns:
            tuple: (输出张量, 更新后的 KV 缓存)。
        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        if chunk_size is not None and look_back > 0 or look_back == -1:
            if cache is not None:
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)

                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache_tmp = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
                cache = cache_tmp
        fsmn_memory = self.forward_fsmn(v, None)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, None)
        return att_outs + fsmn_memory, cache


class LayerNorm(nn.LayerNorm):
    """改进的层归一化，将输入转为 float32 进行计算后再转回原始类型。

    这确保了归一化的数值稳定性，同时保持与输入相同的数据类型（如 fp16）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """前向传播：float32 归一化后转回原始类型。"""
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    """生成序列掩码。

    根据每个序列的有效长度生成布尔掩码，有效位置为 True，填充位置为 False。

    Args:
        lengths (torch.Tensor): 每个序列的有效长度，形状为 (batch,)。
        maxlen (int, optional): 最大序列长度，默认为 lengths 的最大值。
        dtype (torch.dtype): 输出掩码的数据类型。
        device (torch.device, optional): 输出掩码的设备。

    Returns:
        torch.Tensor: 序列掩码，形状为 (batch, maxlen)。
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class EncoderLayerSANM(nn.Module):
    """SANM 编码器层。

    每个编码器层包含：
    1. SANM 自注意力（多头注意力 + FSMN）
    2. 前馈网络（PositionwiseFeedForward）
    3. 残差连接和层归一化

    支持 Pre-Norm 和 Post-Norm 两种归一化方式，以及随机深度正则化。
    """

    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """编码器层前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch, time, size)。
            mask (torch.Tensor): 输入掩码。
            cache (torch.Tensor, optional): 缓存张量（用于流式推理）。
            mask_shfit_chunk (torch.Tensor, optional): 块移位掩码。
            mask_att_chunk_encoder (torch.Tensor, optional): 编码器注意力掩码。

        Returns:
            tuple: (输出张量, 掩码, 缓存, 块移位掩码, 编码器注意力掩码)。
        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """流式推理时的分块前向传播。

        Args:
            x (torch.Tensor): 当前块的输入张量。
            cache (dict, optional): KV 缓存。
            chunk_size (tuple, optional): 分块大小配置。
            look_back (int): 回看块数。

        Returns:
            tuple: (输出张量, 更新后的 KV 缓存)。
        """

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, cache


@tables.register("encoder_classes", "SenseVoiceEncoderSmall")
class SenseVoiceEncoderSmall(nn.Module):
    """SenseVoice 小型编码器，基于 SANM (Streaming Chunk-Aware Multihead Attention) 架构。

    采用两阶段编码结构：
    - encoders0 + encoders: 主编码器，负责将音频特征编码为高层表示
    - tp_encoders: 任务感知编码器，进一步处理以适应下游任务

    参考论文: SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01713

    Author: Speech Lab of DAMO Academy, Alibaba Group
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        tp_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        selfattention_layer_type: str = "sanm",
        **kwargs,
    ):
        """初始化 SenseVoice 编码器。

        Args:
            input_size (int): 输入特征维度（如 FBank 特征维度）。
            output_size (int): 编码器输出维度。
            attention_heads (int): 多头注意力的头数。
            linear_units (int): 前馈网络的隐藏层维度。
            num_blocks (int): 主编码器层数。
            tp_blocks (int): 任务感知编码器层数。
            dropout_rate (float): Dropout 比率。
            positional_dropout_rate (float): 位置编码的 Dropout 比率。
            attention_dropout_rate (float): 注意力权重的 Dropout 比率。
            stochastic_depth_rate (float): 随机深度比率（训练时随机跳过层）。
            kernel_size (int): FSMN 的卷积核大小。
            sanm_shfit (int): SANM 的偏移量。
        """
        super().__init__()
        self._output_size = output_size

        self.embed = SinusoidalPositionEncoder()

        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        encoder_selfattn_layer = MultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )

        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    input_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(1)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(num_blocks - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(tp_blocks)
            ]
        )

        self.after_norm = LayerNorm(output_size)

        self.tp_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        """Output size."""
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ):
        """Embed positions in tensor."""
        maxlen = xs_pad.shape[1]
        masks = sequence_mask(ilens, maxlen=maxlen, device=ilens.device)[:, None, :]

        xs_pad *= self.output_size() ** 0.5

        xs_pad = self.embed(xs_pad)

        # forward encoder1
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad)

        # forward encoder2
        olens = masks.squeeze(1).sum(1).int()

        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens


@tables.register("model_classes", "SenseVoiceSmall")
class SenseVoiceSmall(nn.Module):
    """SenseVoice 小型语音理解模型。

    基于 CTC 的端到端语音识别模型，支持：
    - 多语言语音识别（中文、英文、粤语、日语、韩语等）
    - 语音情感识别（开心、悲伤、愤怒、中性）
    - 音频事件检测（语音、背景音乐、掌声、笑声）
    - 字符级时间戳

    模型结构：音频特征 -> 语言/情感/事件查询嵌入拼接 -> 编码器 -> CTC 解码
    """

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        ctc_conf: dict = None,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        """初始化 SenseVoice 模型。

        Args:
            specaug (str, optional): SpecAugment 数据增强类名。
            specaug_conf (dict, optional): SpecAugment 配置。
            normalize (str, optional): 特征归一化类名（如 CMVN）。
            normalize_conf (dict, optional): 归一化配置。
            encoder (str, optional): 编码器类名。
            encoder_conf (dict, optional): 编码器配置。
            ctc_conf (dict, optional): CTC 解码器配置。
            input_size (int): 输入特征维度（默认 80 维 FBank）。
            vocab_size (int): 词表大小。
            ignore_id (int): 计算损失时忽略的标签 ID。
            blank_id (int): CTC blank token 的 ID。
            sos (int): 序列起始 token ID。
            eos (int): 序列结束 token ID。
            length_normalized_loss (bool): 是否对损失进行长度归一化。
        """
        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if ctc_conf is None:
            ctc_conf = {}
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.error_calculator = None

        self.ctc = ctc

        self.length_normalized_loss = length_normalized_loss
        self.encoder_output_size = encoder_output_size

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.embed = torch.nn.Embedding(
            7 + len(self.lid_dict) + len(self.textnorm_dict), input_size
        )
        self.emo_dict = {
            "unk": 25009,
            "happy": 25001,
            "sad": 25002,
            "angry": 25003,
            "neutral": 25004,
        }

        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        """From pretrained.
        
            Args:
                model: Model instance or model name.
                **kwargs: Additional keyword arguments.
            """
        from funasr import AutoModel

        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)

        return model, kwargs

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, text)

        loss_ctc, cer_ctc = None, None
        loss_rich, acc_rich = None, None
        stats = dict()

        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out[:, 4:, :], encoder_out_lens - 4, text[:, 4:], text_lengths - 4
        )

        loss_rich, acc_rich = self._calc_rich_ce_loss(encoder_out[:, :4, :], text[:, :4])

        loss = loss_ctc + loss_rich
        # Collect total loss stats
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None
        stats["loss_rich"] = torch.clone(loss_rich.detach()) if loss_rich is not None else None
        stats["loss"] = torch.clone(loss.detach()) if loss is not None else None
        stats["acc_rich"] = acc_rich

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        **kwargs,
    ):
        """音频编码：前端特征增强 + 语言/风格/事件查询嵌入拼接 + 编码器。

        处理流程：
        1. 可选的 SpecAugment 数据增强
        2. 可选的特征归一化（如 CMVN）
        3. 从 text 中提取语言 ID 和文本归一化风格，生成查询嵌入
        4. 拼接 [语言查询, 事件查询, 风格查询, 音频特征] 作为编码器输入
        5. 通过 SANM 编码器编码

        Args:
            speech (torch.Tensor): 音频特征，形状为 (batch, time, feat_dim)。
            speech_lengths (torch.Tensor): 每个样本的有效长度。
            text (torch.Tensor): 包含语言 ID 和风格信息的 token 序列。

        Returns:
            tuple: (encoder_out, encoder_out_lens)
                - encoder_out: 编码器输出，形状为 (batch, time+4, output_size)。
                - encoder_out_lens: 输出的有效长度。
        """

        # Data augmentation
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        lids = torch.LongTensor(
            [
                [
                    (
                        self.lid_int_dict[int(lid)]
                        if torch.rand(1) > 0.2 and int(lid) in self.lid_int_dict
                        else 0
                    )
                ]
                for lid in text[:, 0]
            ]
        ).to(speech.device)
        language_query = self.embed(lids)

        styles = torch.LongTensor(
            [[self.textnorm_int_dict[int(style)]] for style in text[:, 3]]
        ).to(speech.device)
        style_query = self.embed(styles)
        speech = torch.cat((style_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        """计算 CTC 损失和 CER（字符错误率）。

        Args:
            encoder_out (torch.Tensor): 编码器输出。
            encoder_out_lens (torch.Tensor): 编码器输出的有效长度。
            ys_pad (torch.Tensor): 填充后的目标序列。
            ys_pad_lens (torch.Tensor): 目标序列的有效长度。

        Returns:
            tuple: (loss_ctc, cer_ctc)
                - loss_ctc: CTC 损失值。
                - cer_ctc: 字符错误率（仅在评估时计算）。
        """
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rich_ce_loss(
        self,
        encoder_out: torch.Tensor,
        ys_pad: torch.Tensor,
    ):
        """计算富标签交叉熵损失（用于情感/事件/语言等辅助任务）。

        对编码器输出的前 4 个位置（语言、事件、情感、风格）计算交叉熵损失。

        Args:
            encoder_out (torch.Tensor): 编码器输出，形状为 (batch, 4, output_size)。
            ys_pad (torch.Tensor): 富标签序列（包含语言、事件、情感、风格 token）。

        Returns:
            tuple: (loss_rich, acc_rich)
                - loss_rich: 交叉熵损失值。
                - acc_rich: 预测准确率。
        """
        decoder_out = self.ctc.ctc_lo(encoder_out)
        # 2. Compute attention loss
        loss_rich = self.criterion_att(decoder_out, ys_pad.contiguous())
        acc_rich = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad.contiguous(),
            ignore_label=self.ignore_id,
        )

        return loss_rich, acc_rich

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = ["wav_file_tmp_name"],
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        """Run inference on input data.
        
            Args:
                data_in: Input data (audio samples, file paths, or text).
                data_lengths: Lengths of each input sample in the batch.
                key: Sample identifiers.
                tokenizer: Tokenizer instance for text encoding/decoding.
                frontend: Audio frontend for feature extraction.
                **kwargs: Additional keyword arguments.
            """
        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        language = kwargs.get("language", "auto")
        language_query = self.embed(
            torch.LongTensor([[self.lid_dict[language] if language in self.lid_dict else 0]]).to(
                speech.device
            )
        ).repeat(speech.size(0), 1, 1)

        use_itn = kwargs.get("use_itn", False)
        textnorm = kwargs.get("text_norm", None)
        output_timestamp = kwargs.get("output_timestamp", False)

        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # Encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # c. Passed the encoder result and the beam search
        ctc_logits = self.ctc.log_softmax(encoder_out)
        if kwargs.get("ban_emo_unk", False):
            ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")

        results = []
        b, n, d = encoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            x = ctc_logits[i, : encoder_out_lens[i].item(), :]
            yseq = x.argmax(dim=-1)
            yseq = torch.unique_consecutive(yseq, dim=-1)

            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"1best_recog"]

            mask = yseq != self.blank_id
            token_int = yseq[mask].tolist()

            # Change integer-ids to tokens
            text = tokenizer.decode(token_int)

            # result_i = {"key": key[i], "text": text}
            # results.append(result_i)

            if ibest_writer is not None:
                ibest_writer["text"][key[i]] = text

            if output_timestamp:
                from itertools import groupby

                timestamp = []
                tokens = tokenizer.text2tokens(text)[4:]
                token_back_to_id = tokenizer.tokens2ids(tokens)
                token_ids = []
                for tok_ls in token_back_to_id:
                    if tok_ls: token_ids.extend(tok_ls)
                    else: token_ids.append(124)

                if len(token_ids) == 0:
                    result_i = {"key": key[i], "text": text}
                    results.append(result_i)
                    continue

                logits_speech = self.ctc.log_softmax(encoder_out)[i, 4 : encoder_out_lens[i].item(), :]
                pred = logits_speech.argmax(-1).cpu()
                logits_speech[pred == self.blank_id, self.blank_id] = 0
                align = ctc_forced_align(
                    logits_speech.unsqueeze(0).float(),
                    torch.Tensor(token_ids).unsqueeze(0).long().to(logits_speech.device),
                    (encoder_out_lens[i] - 4).long(),
                    torch.tensor(len(token_ids)).unsqueeze(0).long().to(logits_speech.device),
                    ignore_id=self.ignore_id,
                )
                pred = groupby(align[0, : encoder_out_lens[i]])
                _start = 0
                token_id = 0
                ts_max = encoder_out_lens[i] - 4
                for pred_token, pred_frame in pred:
                    _end = _start + len(list(pred_frame))
                    if pred_token != 0:
                        ts_left = max((_start * 60 - 30) / 1000, 0)
                        ts_right = min((_end * 60 - 30) / 1000, (ts_max * 60 - 30) / 1000)
                        timestamp.append([tokens[token_id], ts_left, ts_right])
                        token_id += 1
                    _start = _end
                timestamp, words = self.post(timestamp)
                result_i = {"key": key[i], "text": text, "timestamp": timestamp, "words": words}
                results.append(result_i)
            else:
                result_i = {"key": key[i], "text": text}
                results.append(result_i)
        return results, meta_data

    def post(self, timestamp):
        """后处理时间戳：合并子词为完整单词，并将秒转为毫秒。

        处理逻辑：
        1. 跳过纯空格 token（"▁"）
        2. 以 "▁" 开头的 token 作为新单词的开始
        3. 连续的英文字符合并为一个单词
        4. 其他情况（中文等）每个 token 独立成词

        Args:
            timestamp (list): 原始时间戳列表，每个元素为 [token, start_sec, end_sec]。

        Returns:
            tuple: (timestamp_new, words_new)
                - timestamp_new: 时间戳列表，每个元素为 [start_ms, end_ms]。
                - words_new: 对应的单词/字符列表。
        """
        timestamp_new = []
        words_new = []
        prev_word = None
        for i, t in enumerate(timestamp):
            word, start, end = t
            start = int(start * 1000)
            end = int(end * 1000)
            if word == "▁":
                continue
            if i == 0:
                # timestamp_new.append([word, start, end])
                timestamp_new.append([start, end])
                words_new.append(word)
            elif word.startswith("▁"):
                word = word[1:]
                timestamp_new.append([start, end])
                words_new.append(word)
            elif prev_word is not None and prev_word.isalpha() and prev_word.isascii() and word.isalpha() and word.isascii():
                word = prev_word + word
                timestamp_new[-1][1] = end
                words_new[-1] = word
            else:
                # timestamp_new[-1][0] += word
                timestamp_new.append([start, end])
                words_new.append(word)
            prev_word = word
        return timestamp_new, words_new

    def export(self, **kwargs):
        """Export.
        
            Args:
                **kwargs: Additional keyword arguments.
            """
        from .export_meta import export_rebuild_model

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models

        return results, meta_data
