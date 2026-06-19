"""SCP 转 JSONL 数据转换工具。

将 SCP 格式的音频路径文件和转录文本文件转换为 Fun-ASR-Nano 训练所需的
ChatML JSONL 格式。支持本地文件和 HTTP URL 音频路径，使用多线程并行处理。

输入格式：
    scp_file:       utterance_id  /path/to/audio.wav
    transcript_file: utterance_id  转录文本

输出格式（JSONL）：
    {"messages": [...], "speech_length": N, "text_length": M}
"""

"""SCP + Transcript 转 JSONL 训练数据工具。

将 Kaldi 格式的 scp 文件（音频路径列表）和 transcript 文件（对应文本）
转换为 Fun-ASR-Nano 训练所需的 JSONL 格式。

输入格式:
    scp_file:       utt_id  audio_path
    transcript_file: utt_id  text

输出格式 (JSONL):
    {"messages": [...], "speech_length": int, "text_length": int}
"""

import hydra
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Dict, Optional, Tuple
from urllib.request import urlopen

import soundfile as sf
from modelscope import AutoTokenizer
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf, ListConfig


class LineProcessor:
    """单行数据处理器，负责将一对 scp + transcript 行转换为 JSONL 格式。

    线程安全：通过 threading.Lock 保护共享的 tokenizer。
    """

    def __init__(self, tokenizer):
        """初始化处理器。

        Args:
            tokenizer: 用于计算文本 token 长度的分词器实例。
        """
        self.tokenizer = tokenizer
        self.lock = threading.Lock()

    def process_line(self, line_pair: Tuple[str, str]) -> Optional[Dict]:
        """处理一对 scp + transcript 行，转换为 JSONL 训练数据格式。

        校验 utt_id 一致性，加载音频获取时长，构建 ChatML 格式的消息列表。

        Args:
            line_pair (tuple): (scp_line, transcript_line)，每行格式为 "utt_id content"。

        Returns:
            dict: 成功时返回 {"success": data_dict, "utt": utt_id}，
                  失败时返回 {"error": error_message}，跳过空行时返回 None。
        """
        line1, line2 = line_pair

        line1, line2 = line1.strip(), line2.strip()
        if not line1 or not line2:
            return None

        parts1, parts2 = line1.split(maxsplit=1), line2.split(maxsplit=1)
        if len(parts1) != 2 or len(parts2) != 2:
            return None

        utt1, utt2 = parts1[0], parts2[0]
        wav_path, text = parts1[1], parts2[1]

        # 校验两个文件中的 utt_id 是否一致
        if utt1 != utt2:
            return {"error": f"UTT mismatch: {utt1} vs {utt2}"}

        try:
            # 加载音频并获取时长（支持 HTTP URL 和本地路径）
            if wav_path.startswith("http"):
                response = urlopen(wav_path)
                if response.status != 200:
                    return {"error": f"WAV not found: {wav_path}"}
                audio_file = BytesIO(response.read())
                duration = sf.info(audio_file).duration
            else:
                if not os.path.exists(wav_path):
                    return {"error": f"WAV not found: {wav_path}"}
                duration = sf.info(wav_path).duration

            # 构建 ChatML 格式的训练数据
            # speech_length 计算公式：(duration_ms - 25) // 10 + 1，对应前端特征帧数
            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"语音转写：<|startofspeech|>!{wav_path}<|endofspeech|>",
                    },
                    {"role": "assistant", "content": text},
                ],
                "speech_length": int((duration * 1000 - 25) // 10 + 1),
                "text_length": len(self.tokenizer.tokenize(text)),
            }
            return {"success": data, "utt": utt1}

        except Exception as e:
            return {"error": f"Error processing {wav_path}: {str(e)}"}


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    """Hydra 入口函数：并行读取 scp + transcript 文件，转换为 JSONL 训练数据。

    使用线程池并行处理每一行数据，处理完成后统计成功/失败数量并输出摘要。

    Args:
        cfg (DictConfig): Hydra 配置，需包含 scp_file, transcript_file, jsonl_file，
                          可选 max_workers（默认为 CPU 核心数）。
    """
    def to_plain_list(cfg_item):
        """递归地将 OmegaConf 配置对象转换为纯 Python 类型。"""
        if isinstance(cfg_item, ListConfig):
            return OmegaConf.to_container(cfg_item, resolve=True)
        elif isinstance(cfg_item, DictConfig):
            return {k: to_plain_list(v) for k, v in cfg_item.items()}
        else:
            return cfg_item

    kwargs = to_plain_list(cfg)
    scp_file = kwargs["scp_file"]
    transcript_file = kwargs["transcript_file"]
    max_workers = kwargs.get("max_workers", os.cpu_count())
    jsonl_file = kwargs["jsonl_file"]

    with open(scp_file, "r") as f1, open(transcript_file, "r") as f2:
        scp_lines = f1.readlines()
        transcript_lines = f2.readlines()

    if len(scp_lines) != len(transcript_lines):
        print(f"Warning: Line count mismatch - scp: {len(scp_lines)}, transcript: {len(transcript_lines)}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    processor = LineProcessor(tokenizer)

    data_pairs = list(zip(scp_lines, transcript_lines))

    processed_count = 0
    failed_count = 0
    error_messages = []

    with tqdm(total=len(data_pairs), desc="Processing") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with open(jsonl_file, "w") as f_out:
                futures = {executor.submit(processor.process_line, pair): i for i, pair in enumerate(data_pairs)}

                for future in as_completed(futures):
                    result = future.result()

                    if result and "success" in result:
                        with processor.lock:
                            json.dump(result["success"], f_out, ensure_ascii=False)
                            f_out.write("\n")
                        processed_count += 1
                    elif result and "error" in result:
                        failed_count += 1
                        error_messages.append(result["error"])

                    pbar.update(1)
                    pbar.set_postfix({"processed": processed_count, "failed": failed_count})

    print(f"\nProcessing completed:")
    print(f"  Total lines: {len(data_pairs)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Failed: {failed_count}")

    if error_messages and len(error_messages) <= 10:
        print(f"\nSample errors:")
        for error in error_messages[:10]:
            print(f"  - {error}")
    elif error_messages:
        print(f"\nFirst 10 errors:")
        for error in error_messages[:10]:
            print(f"  - {error}")
        print(f"  ... and {len(error_messages) - 10} more errors")


if __name__ == "__main__":
    main_hydra()
