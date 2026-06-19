"""Fun-ASR-Nano 工具函数模块

提供音频加载和 CTC 强制对齐等基础工具函数。
"""

from itertools import groupby

import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F


def load_audio(wav_path, rate: int = None, offset: float = 0, duration: float = None):
    """加载音频文件并可选地进行重采样。

    使用 soundfile 读取音频文件，支持指定偏移位置和时长。
    如果目标采样率与原始采样率不同，会自动进行重采样。

    Args:
        wav_path (str): 音频文件路径。
        rate (int, optional): 目标采样率（Hz）。如果为 None 则使用原始采样率。
        offset (float): 读取起始偏移时间（秒），默认为 0。
        duration (float, optional): 读取时长（秒）。如果为 None 则读取到文件末尾。

    Returns:
        tuple: (audio_tensor, sample_rate)
            - audio_tensor (torch.Tensor): 音频波形张量，float32 格式。
            - sample_rate (int): 实际采样率。
    """
    with sf.SoundFile(wav_path) as f:
        start_frame = int(offset * f.samplerate)
        if duration is None:
            frames_to_read = f.frames - start_frame
        else:
            frames_to_read = int(duration * f.samplerate)
        f.seek(start_frame)
        audio_data = f.read(frames_to_read, dtype="float32")
    audio_tensor = torch.from_numpy(audio_data)
    if rate is not None and f.samplerate != rate:
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio_tensor.T
        resampler = torchaudio.transforms.Resample(orig_freq=f.samplerate, new_freq=rate)
        audio_tensor = resampler(audio_tensor)
        if audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.squeeze(0)
    return audio_tensor, rate if rate is not None else f.samplerate


def forced_align(log_probs: torch.Tensor, targets: torch.Tensor, blank: int = 0):
    """CTC 强制对齐：将 CTC log 概率与目标文本进行对齐，生成字符级时间戳。

    使用 torchaudio 的 forced_align 接口，将 CTC 输出与目标 token 序列对齐，
    返回每个非 blank token 的起止帧索引和置信度分数。

    Args:
        log_probs (torch.Tensor): CTC log 概率张量，形状为 (T, vocab_size)。
        targets (torch.Tensor): 目标 token ID 序列，形状为 (target_len,)。
        blank (int): blank token 的 ID，默认为 0。

    Returns:
        list[dict]: 对齐结果列表，每个元素包含：
            - token (int): token ID。
            - start_time (int): 起始帧索引。
            - end_time (int): 结束帧索引。
            - score (float): 该 token 的最大置信度（保留 3 位小数）。
    """
    items = []
    try:
        # 当前版本仅支持 batch_size == 1
        log_probs, targets = log_probs.unsqueeze(0).cpu(), targets.unsqueeze(0).cpu()
        assert log_probs.shape[1] >= targets.shape[1]
        # 执行强制对齐，获取对齐序列和分数
        alignments, scores = F.forced_align(log_probs, targets, blank=blank)
        alignments, scores = alignments[0], torch.exp(scores[0]).tolist()
        # 按 token 值分组，合并连续相同 token 的帧范围
        for token, group in groupby(enumerate(alignments), key=lambda item: item[1]):
            if token == blank:
                continue
            group = list(group)
            start = group[0][0]
            end = start + len(group)
            score = max(scores[start:end])
            items.append(
                {
                    "token": token.item(),
                    "start_time": start,
                    "end_time": end,
                    "score": round(score, 3),
                }
            )
    except:
        pass
    return items
