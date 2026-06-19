"""
声纹验证模块 (Speaker Verification Module)

用于解决：将用户录音的声纹与会议录音中的说话人进行匹配识别。

核心流程：
    1. 声纹注册：从用户录音中提取说话人嵌入向量作为"声纹模板"
    2. 声纹验证：从会议音频中提取说话人嵌入，与模板计算相似度
    3. 阈值判定：根据余弦相似度判断是否为同一说话人

关键技术：
    - 使用 Cam++ 模型提取 192 维说话人嵌入
    - 余弦相似度 (Cosine Similarity) 衡量说话人相似度
    - CMS (Cepstral Mean Subtraction) 信道补偿，缓解域偏移
    - 多模板策略：取多段音频嵌入的均值作为更鲁棒的声纹

典型用法：
    >>> from funasr.models.fun_asr_nano.speaker_verifier import SpeakerVerifier
    >>> verifier = SpeakerVerifier()
    >>> # 注册用户声纹
    >>> verifier.register_user("user_001", "user_audio.wav")
    >>> # 验证会议音频
    >>> scores = verifier.verify("meeting.wav", "user_001")
    >>> # scores 包含每个说话人片段的相似度分数
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SpeakerVerifier:
    """说话人验证器

    基于 FunASR 的 Cam++ 声纹模型，实现跨录音环境的说话人验证。

    Args:
        model: Cam++ 模型路径，默认 "cam++"
        device: 计算设备，默认 "cuda"（无 GPU 时回退 "cpu"）
        sample_rate: 音频采样率，默认 16000
        threshold: 余弦相似度阈值，高于此值判定为同一说话人
                   默认 0.6，可调范围 [0.3, 0.8]
    """

    def __init__(
        self,
        model: str = "cam++",
        device: str = "cuda",
        sample_rate: int = 16000,
        threshold: float = 0.6,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.threshold = threshold

        # 懒加载声纹模型（首次调用时初始化，节省启动时间）
        self._spk_model = None
        # 声纹模板存储：{user_id: [embedding1, embedding2, ...]}
        self._templates: Dict[str, List[torch.Tensor]] = {}

    @property
    def spk_model(self):
        """懒加载声纹模型"""
        if self._spk_model is None:
            from funasr import AutoModel
            self._spk_model = AutoModel(
                model="cam++", device=self.device, disable_update=True
            )
        return self._spk_model

    def _extract_embedding(self, audio: np.ndarray) -> torch.Tensor:
        """从单段音频提取说话人嵌入向量

        Args:
            audio: numpy 数组，单声道音频数据 (16kHz)

        Returns:
            归一化的 192 维说话人嵌入向量
        """
        with torch.no_grad():
            result = self.spk_model.generate(
                input=[audio], cache={}, is_final=True
            )
            # result[0]["spk_embedding"]: shape (1, 192)
            emb = result[0]["spk_embedding"]
            # L2 归一化，使余弦相似度计算更稳定
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def _extract_multi_embeddings(
        self, audio: np.ndarray, chunk_duration: float = 2.0
    ) -> List[torch.Tensor]:
        """从长音频提取多段说话人嵌入

        将长音频切分为多个 chunk，分别提取嵌入，取平均提高鲁棒性。
        这是缓解域偏移的关键策略：多段统计平均比单段更稳定。

        Args:
            audio: numpy 数组，音频数据
            chunk_duration: 每个 chunk 的时长（秒），默认 2 秒

        Returns:
            多个归一化的说话人嵌入列表
        """
        sr = self.sample_rate
        chunk_samples = int(chunk_duration * sr)
        hop_samples = chunk_samples // 2  # 50% 重叠

        embeddings = []
        for start in range(0, max(0, len(audio) - chunk_samples), hop_samples):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) >= chunk_samples * 0.5:
                emb = self._extract_embedding(chunk)
                embeddings.append(emb)

        # 如果音频太短，直接提取整段的嵌入
        if not embeddings:
            embeddings.append(self._extract_embedding(audio))

        return embeddings

    def _cms_normalize(self, embedding: torch.Tensor) -> torch.Tensor:
        """CMS 归一化：消除信道效应，缓解域偏移

        CMS (Cepstral Mean Subtraction) 是经典的信道补偿方法：
        - 减去均值：消除慢变的信道偏移
        - 除以标准差：归一化嵌入分布

        Args:
            embedding: 原始说话人嵌入

        Returns:
            CMS 归一化后的嵌入
        """
        mean = embedding.mean(dim=-1, keepdim=True)
        std = embedding.std(dim=-1, keepdim=True) + 1e-8
        return (embedding - mean) / std

    def register_user(
        self,
        user_id: str,
        audio: np.ndarray,
        multi_chunk: bool = True,
    ) -> torch.Tensor:
        """注册用户声纹模板

        Args:
            user_id: 用户唯一标识（如 "user_001"）
            audio: 用户录音的 numpy 数组（16kHz 单声道）
            multi_chunk: 是否使用多 chunk 策略，默认 True
                        True: 切分为多段取平均（更鲁棒）
                        False: 直接使用整段嵌入（更快速）

        Returns:
            注册后的平均声纹嵌入向量
        """
        if multi_chunk:
            embeddings = self._extract_multi_embeddings(audio)
        else:
            embeddings = [self._extract_embedding(audio)]

        # 取所有嵌入的平均作为模板
        template = torch.stack(embeddings).mean(dim=0)
        # 再次 L2 归一化
        template = template / template.norm(dim=-1, keepdim=True)
        # CMS 归一化
        template = self._cms_normalize(template)

        self._templates[user_id] = [template]
        return template

    def register_user_from_file(
        self,
        user_id: str,
        audio_path: str,
        multi_chunk: bool = True,
    ) -> torch.Tensor:
        """从音频文件注册用户声纹

        Args:
            user_id: 用户唯一标识
            audio_path: 音频文件路径（支持 mp3/wav/m4a 等）
            multi_chunk: 是否使用多 chunk 策略

        Returns:
            注册后的平均声纹嵌入向量
        """
        audio = self._load_audio(audio_path)
        return self.register_user(user_id, audio, multi_chunk)

    def compute_similarity(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> float:
        """计算两个嵌入向量的余弦相似度

        余弦相似度范围 [-1, 1]，值越大表示越相似。
        经验阈值：
            - > 0.7: 极高置信度（同一说话人）
            - 0.6 ~ 0.7: 中等置信度（可能同一说话人）
            - < 0.6: 低置信度（可能不同说话人）

        Args:
            embedding_a: 说话人嵌入 A
            embedding_b: 说话人嵌入 B

        Returns:
            余弦相似度分数
        """
        # 确保都是 L2 归一化的嵌入
        a = embedding_a / embedding_a.norm(dim=-1, keepdim=True)
        b = embedding_b / embedding_b.norm(dim=-1, keepdim=True)
        # 余弦相似度 = 点积（已归一化）
        similarity = (a * b).sum(dim=-1).item()
        return float(similarity)

    def verify_segment(
        self,
        segment_audio: np.ndarray,
        user_id: str,
    ) -> Tuple[float, bool]:
        """验证单个音频片段是否为指定用户

        Args:
            segment_audio: 待验证的音频片段 (numpy array)
            user_id: 已注册的用户 ID

        Returns:
            (相似度分数, 是否匹配) 元组
        """
        if user_id not in self._templates:
            raise ValueError(f"用户 {user_id} 未注册，请先调用 register_user")

        # 提取待验证片段的嵌入
        test_embs = self._extract_multi_embeddings(segment_audio)

        # 计算与模板的最大相似度
        template = self._templates[user_id][0]
        max_score = -1.0
        for emb in test_embs:
            score = self.compute_similarity(emb, template)
            max_score = max(max_score, score)

        return max_score, max_score >= self.threshold

    def verify_meeting_audio(
        self,
        meeting_audio: np.ndarray,
        user_ids: List[str],
        vad_segments: Optional[List[Tuple]] = None,
    ) -> List[Dict]:
        """验证会议音频中的说话人身份

        这是主要的业务接口：输入会议音频和多个用户ID，
        返回每个说话人片段的身份识别结果。

        Args:
            meeting_audio: 会议录音的 numpy 数组
            user_ids: 要验证的用户 ID 列表
            vad_segments: VAD 分段列表，格式 [[start_ms, end_ms], ...]
                          如果为 None，将使用固定 2 秒分段

        Returns:
            说话人验证结果列表，每项包含：
            {
                "segment_idx": 片段索引,
                "start_ms": 开始时间,
                "end_ms": 结束时间,
                "speaker_embedding": 说话人嵌入,
                "scores": {user_id: similarity_score, ...},
                "identified_user": 最可能的用户 ID（或 None）,
                "is_match": 是否匹配（超过阈值）
            }
        """
        sr = self.sample_rate
        results = []

        # 如果没有 VAD 分段，使用固定 2 秒分段
        if vad_segments is None:
            chunk_ms = 2000  # 2 秒
            total_ms = int(len(meeting_audio) / sr * 1000)
            vad_segments = [
                (i * chunk_ms, min((i + 1) * chunk_ms, total_ms))
                for i in range(total_ms // chunk_ms)
            ]

        for seg_idx, (start_ms, end_ms) in enumerate(vad_segments):
            start_sample = int(start_ms / 1000 * sr)
            end_sample = int(end_ms / 1000 * sr)
            segment = meeting_audio[start_sample:end_sample]

            if len(segment) < sr * 0.5:  # 跳过 < 0.5 秒的片段
                continue

            # 提取该片段的说话人嵌入
            seg_embs = self._extract_multi_embeddings(segment)
            seg_emb = torch.stack(seg_embs).mean(dim=0)
            seg_emb = seg_emb / seg_emb.norm(dim=-1, keepdim=True)
            seg_emb = self._cms_normalize(seg_emb)

            # 与每个注册用户计算相似度
            scores = {}
            best_user = None
            best_score = -1.0
            for uid in user_ids:
                if uid in self._templates:
                    score = self.compute_similarity(seg_emb, self._templates[uid][0])
                    scores[uid] = score
                    if score > best_score:
                        best_score = score
                        best_user = uid

            results.append({
                "segment_idx": seg_idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker_embedding": seg_emb,
                "scores": scores,
                "identified_user": best_user if best_score >= self.threshold else None,
                "is_match": best_score >= self.threshold,
            })

        return results

    def set_threshold(self, threshold: float):
        """调整验证阈值

        Args:
            threshold: 新阈值，范围 [0, 1]
                       - 更严格 (0.7+): 降低误报，但可能漏检
                       - 更宽松 (0.4-0.5): 提高召回，但可能误匹配
        """
        self.threshold = max(0.0, min(1.0, threshold))

    def _load_audio(self, path: str) -> np.ndarray:
        """加载音频文件并转换为 16kHz 单声道

        Args:
            path: 音频文件路径

        Returns:
            16kHz 单声道 numpy 数组
        """
        import soundfile as sf
        audio, orig_sr = sf.read(path, dtype="float32")

        # 转单声道
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 重采样到 16kHz
        if orig_sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)

        return audio

    def list_registered_users(self) -> List[str]:
        """列出所有已注册的用户"""
        return list(self._templates.keys())

    def remove_user(self, user_id: str) -> bool:
        """删除已注册的用户"""
        if user_id in self._templates:
            del self._templates[user_id]
            return True
        return False
