# -*- coding: utf-8 -*-
#!/usr/bin/python
# Author: Mengze Chen
"""
Whisper 混合文本归一化工具。

结合 Whisper 的英文归一化器和中文文本归一化工具（cn_tn, cn_itn），
对多语言（中英日混合）语音识别结果进行统一的文本归一化处理。

处理流程:
    1. 按语言分段（纯英文、中英混合、其他语言）
    2. 英文段使用 Whisper EnglishTextNormalizer
    3. 中英混合段：英文归一化 + 中文 NSW 归一化 + ITN + 繁简转换
    4. 其他语言段：BasicTextNormalizer + 中文 NSW 归一化 + ITN + 繁简转换

依赖:
    pip install whisper-normalizer zhconv pyopenjtalk
"""

import re
import sys

import cn_tn as cn_tn
import format5res as cn_itn
import pyopenjtalk
import zhconv
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

# 初始化 Whisper 归一化器
basic_normalizer = BasicTextNormalizer()
english_normalizer = EnglishTextNormalizer()


def is_only_chinese_and_english(s):
    """判断字符串是否仅包含中文、英文、数字和基本标点。"""
    pattern = r"^[\u4e00-\u9fa5A-Za-z0-9,\.!\?:;，。！？：；、%\'\s\-\~]+$"
    return re.match(pattern, s) is not None


def is_only_english(s):
    """判断字符串是否仅包含英文、数字和基本标点（不含中文）。"""
    pattern = r"^[A-Za-z0-9,\.!\?:;，。！？：；、%\'\s\-\~]+$"
    return re.match(pattern, s) is not None


def is_number(s):
    """判断字符串是否仅包含数字和基本标点。"""
    pattern = r"^[0-9,\.!\?:;，。！？：；、%\'\s]+$"
    return re.match(pattern, s) is not None


def is_only_english(s):
    """检查字符串是否仅包含英文、数字和常见标点（不含中文）。

    Args:
        s (str): 待检查的字符串。

    Returns:
        bool: 如果仅包含英文/数字/标点则返回 True。
    """
    pattern = r"^[A-Za-z0-9,\.!\?:;，。！？：；、%\'\s\-\~]+$"
    return re.match(pattern, s) is not None


def is_number(s):
    """检查字符串是否仅包含数字和常见标点。

    Args:
        s (str): 待检查的字符串。

    Returns:
        bool: 如果仅包含数字和标点则返回 True。
    """
    pattern = r"^[0-9,\.!\?:;，。！？：；、%\'\s]+$"
    return re.match(pattern, s) is not None


def safe_ja_g2p(text, kana=True, max_length=100):
    """安全的日语音素转换（Grapheme-to-Phoneme）。

    对过长的文本进行分段处理，避免 pyopenjtalk 处理超长文本时出错。
    转换失败时返回原文本，保证不会因异常而中断流程。

    Args:
        text (str): 输入日语文本。
        kana (bool): 是否返回假名注音，默认 True。
        max_length (int): 单次处理的最大字符数，默认 100。

    Returns:
        str: 音素转换后的文本，失败时返回原文本。
    """
    if len(text) > max_length:
        # 文本过长时分段处理，每段最多 max_length 个字符
        parts = []
        for i in range(0, len(text), max_length):
            part = text[i : i + max_length]
            try:
                converted = pyopenjtalk.g2p(part, kana=kana)
                parts.append(converted)
            except:
                parts.append(part)
        return " ".join(parts)
    else:
        try:
            return pyopenjtalk.g2p(text, kana=kana)
        except:
            return text


def normalize_text(srcfn, dstfn, kana=False):
    """对识别结果文件进行多语言混合文本归一化。

    逐行读取源文件，按语言类型（纯英文/中英混合/其他）分段处理：
    - 英文段：使用 Whisper EnglishTextNormalizer + cn_itn 格式化
    - 中英混合段：Whisper 英文归一化 + cn_tn NSW 归一化 + cn_itn ITN + 繁简转换
    - 其他语言段：Whisper BasicTextNormalizer + cn_tn + cn_itn + 繁简转换

    Args:
        srcfn (str): 源文件路径，每行格式为 "key text"。
        dstfn (str): 输出文件路径。
        kana (bool): 是否对日语文本进行假名注音转换，默认 False。
    """
    with open(srcfn, "r") as f_read, open(dstfn, "w") as f_write:
        all_lines = f_read.readlines()
        for line in all_lines:
            line = line.strip()
            line_arr = line.split(maxsplit=1)
            if len(line_arr) < 1:
                continue
            if len(line_arr) == 1:
                line_arr.append("")
            key = line_arr[0]
            # 移除等号和括号（常见 ASR 伪影）
            line_arr[1] = re.sub(r"=", " ", line_arr[1])
            line_arr[1] = re.sub(r"\(", " ", line_arr[1])
            line_arr[1] = re.sub(r"\)", " ", line_arr[1])
            # 可选：对日语文本进行假名注音转换
            if kana:
                line_arr[1] = safe_ja_g2p(line_arr[1], kana=True, max_length=100)

            line_arr = f"{key}\t{line_arr[1]}".split()
            conts = []
            language_bak = ""
            part = []
            # 按语言类型分段处理
            for i in range(1, len(line_arr)):
                out_part = ""
                chn_eng_bool = is_only_chinese_and_english(line_arr[i])
                eng_bool = is_only_english(line_arr[i])
                num_bool = is_number(line_arr[i])
                # 判断当前词的语言类型
                if eng_bool and not num_bool:
                    language = "en"
                elif chn_eng_bool:
                    language = "chn_en"
                else:
                    language = "not_chn_en"
                # 语言类型相同则归入同一段
                if language == language_bak or language_bak == "":
                    part.append(line_arr[i])
                    language_bak = language
                else:
                    # 语言类型切换，处理已积累的段
                    if language_bak == "en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part = cn_itn.scoreformat("", out_part1)
                    elif language_bak == "chn_en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    else:
                        out_part1 = basic_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    conts.append(out_part)
                    language_bak = language
                    part = []
                    part.append(line_arr[i])
                # 处理最后一段
                if i == len(line_arr) - 1:
                    if language == "en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part = cn_itn.scoreformat("", out_part1)
                    elif language == "chn_en":
                        out_part1 = english_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    else:
                        out_part1 = basic_normalizer(" ".join(part))
                        out_part2 = cn_tn.normalize_nsw(out_part1)
                        out_part3 = cn_itn.all_convert(out_part2)
                        out_part = zhconv.convert(out_part3, "zh-cn")
                    conts.append(out_part)

            f_write.write("{0}\t{1}\n".format(key, " ".join(conts).strip()))


if __name__ == "__main__":
    srcfn = sys.argv[1]
    dstfn = sys.argv[2]
    normalize_text(srcfn, dstfn, True if len(sys.argv) > 3 else False)
