# -*- coding: utf-8 -*-
#!/usr/bin/python
# Author: Mengze Chen
"""
中文语音识别结果格式化工具。

提供一系列文本后处理函数，用于将 ASR 识别结果中的中文数字、特殊符号等
转换为标准中文文本格式。处理流程：
    recoformat -> numbersingle -> ch_number2digit -> special -> scoreformat
"""

import re
import sys


def scoreformat(name, line, flag=1):
    """对识别结果进行评分格式化：在中英文之间插入空格，并添加说话人标识。

    遍历文本中的每个字符，判断是否为英文字符（包括拉丁字母、西里尔字母等），
    在中英文交界处插入空格。根据 flag 参数决定说话人名称的拼接方式。

    Args:
        name (str): 说话人/音频名称标识。
        line (str): 待格式化的文本行。
        flag (int): 格式标志。1 = name<tab>text, 0/-1 = text (name), 其他 = text。

    Returns:
        str: 格式化后的文本。
    """
    newline = ""
    for i in range(0, len(line)):
        curr = line[i]
        currEn = False
        if curr == "":
            continue
        # 判断当前字符是否为英文类字符（拉丁、西里尔、扩展拉丁等），且不是数字
        if (
            (curr >= "\u0041" and curr <= "\u005a")  # 大写英文字母
            or (curr >= "\u0061" and curr <= "\u007a")  # 小写英文字母
            or (curr >= "\u0000" and curr <= "\u007f")  # 基本 ASCII（德法西意等）
            or (curr >= "\u0400" and curr <= "\u04ff")  # 西里尔字母（俄语等）
            or (curr >= "\u0100" and curr <= "\u017f")  # 拉丁扩展 A（越南语等）
            or (curr >= "\u0080" and curr <= "\u00ff")  # 拉丁补充（法语等）
            or curr == "'"
        ) and (curr < "\u0030" or curr > "\u0039"):
            currEn = True
        if i == 0:
            newline = newline + curr
        else:
            # 在中英文交界处插入空格
            if lastEn == True and currEn == True:
                newline = newline + curr
            else:
                newline = newline + " " + curr
        # flag == -1 时每个字符都视为非英文（强制每个字符间加空格）
        if flag == -1:
            lastEn = False
        else:
            lastEn = currEn
    # 合并连续空格
    ret = re.sub("[ ]{1,}", " ", newline)
    ret = ret
    if name == "":
        ret = ret
    else:
        # 根据 flag 决定说话人名称的拼接位置
        if flag <= 0:
            ret = ret + " " + "(" + name + ")"
        else:
            ret = name + "\t" + ret
    return ret


def recoformat(line):
    """格式化识别结果：在中文、数字与英文单词之间添加空格分隔。

    遍历文本，根据字符类型（中文/数字 vs 英文）在交界处插入空格，
    使中英文混合文本更易读。

    Args:
        line (str): 待格式化的文本行。

    Returns:
        str: 格式化后的文本，中英文之间已添加空格。
    """
    newline = ""
    en_flag = 0  # 0: 非英文状态, 1: 英文状态
    for i in range(0, len(line)):
        word = line[i]
        if ord(word) == 32:  # 空格
            if en_flag == 0:
                continue
            else:
                en_flag = 0
                newline += " "
        # 中文字符或数字：直接拼接
        if (word >= "\u4e00" and word <= "\u9fa5") or (word >= "\u0030" and word <= "\u0039"):
            if en_flag == 1:
                newline += " " + word  # 从英文切换到中文，加空格
            else:
                newline += word
            en_flag = 0
        elif (
            (word >= "\u0041" and word <= "\u005a")  # 大写英文
            or (word >= "\u0061" and word <= "\u007a")  # 小写英文
            or (word >= "\u0000" and word <= "\u007f")  # 基本 ASCII
            or (word >= "\u0400" and word <= "\u04ff")  # 西里尔字母
            or (word >= "\u0100" and word <= "\u017f")  # 拉丁扩展 A
            or (word >= "\u0080" and word <= "\u00ff")  # 拉丁补充
            or word == "'"
        ):
            if en_flag == 0:
                newline += " " + ("" if (word == "'") else word)  # 从中文切换到英文，加空格
            else:
                newline += word
            en_flag = 1
        else:
            newline += " " + word
    newline = newline
    # 合并连续空格
    newline = re.sub("[ ]{1,}", " ", newline)
    newline = newline
    return newline


def numbersingle(line):
    """将单个阿拉伯数字字符转换为对应的中文数字。

    逐字符遍历文本，将 '0'-'9' 转换为 '零'-'九'，
    将小数点 '.' 转换为 '点'。对连续数字中的 '0' 进行特殊处理。

    Args:
        line (str): 包含阿拉伯数字的文本。

    Returns:
        str: 数字已转换为中文的文本。
    """
    chnu = ["零", "一", "二", "两", "三", "四", "五", "六", "七", "八", "九", "点"]
    newline = ""
    for id in range(len(line)):
        if re.findall(r"\.", line[id]):
            # 小数点：如果后面还有内容则转为 '点'，否则保留原样
            if re.findall(r"\.\s*$", line[id]):
                newline += "."
            else:
                newline += chnu[10]
        elif re.search(r"0", line[id]):
            # 对 '0' 的特殊处理：根据前后字符决定是否添加量词
            if id > 0 and id < len(line) - 1:
                if (
                    re.search(r"\d", line[id - 1])
                    and (not re.search(r"\d", line[id + 1]))
                    and (not re.search(r"0", line[id - 1]))
                ):
                    if id > 2 and len(line) > 2 and (not re.search(r"\d", line[id - 1])):
                        newline = newline[:-1]
                        newline += chnu[int(line[id - 1])] + "十"
                    else:
                        newline += chnu[int(line[id])]
                else:
                    newline += chnu[int(line[id])]
            else:
                newline += chnu[int(line[id])]
        elif re.search(r"\d", line[id]):
            # 普通数字直接转换
            newline += chnu[int(line[id])]
        else:
            # 非数字字符保留原样
            newline += line[id]
    return newline


def ch_number2digit(line):
    """将中文数字表达转换为阿拉伯数字。

    识别文本中的中文数字（如 "一百二十三"、"三千零五"），
    将其转换为对应的阿拉伯数字（如 "123"、"3005"）。
    支持十百千万等量词以及复合量词（十万、百万、千万）。

    Args:
        line (str): 包含中文数字的文本。

    Returns:
        str: 中文数字已转换为阿拉伯数字的文本。
    """
    # number_flag 状态机：
    #   0: 初始状态（未遇到数字）
    #   1: 遇到数字字符（一-九、两、幺）
    #   2: 以 '十' 开头（如 "十五"）
    #   3: 遇到量词（十百千万）
    #  -1: 数字序列结束，准备输出
    #  -2: 数字序列结束，当前字符也需要输出
    number_flag = 0
    zero_flag = 0
    # 量词到数位的映射（'十' 对应第 2 位，'百' 对应第 3 位，以此类推）
    bits = {
        "零": "1",
        "十": "2",
        "百": "3",
        "千": "4",
        "万": "5",
        "十万": "6",
        "百万": "7",
        "千万": "8",
    }
    # 中文数字到阿拉伯数字的映射
    chsh = {
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "两": "2",
        "幺": "1",
    }
    # 特殊单位（"里"、"克"、"米"前的 "千" 不识别为量词）
    unit = {"里": "1", "克": "1", "米": "1"}
    newline = ""
    digit = []  # 存储数字字符
    bit = []    # 存储对应的量词
    onebit = ""
    for i in range(len(line)):
        if ord(line[i]) == 32:
            newline += " "
            continue
        if line[i] in chsh:
            number_flag = 1
            # '两' 在句末或后跟非数字非量词时，不作为数字处理（如 "两天"）
            if line[i] == "两":
                if (i == len(line) - 1) or ((line[i + 1] not in chsh.keys()) and (line[i + 1] not in bits.keys())):
                    number_flag = -1
            if number_flag == 1:
                digit.append(chsh[line[i]])

        elif "十" == line[i] and number_flag == 0:
            # 以 '十' 开头的数字（如 "十五"），隐含前面有个 '一'
            number_flag = 2
            digit.append("1")
            bit.append(line[i])
        elif "十" == line[i] and number_flag == 3:
            # 量词后跟 '十'（如 "百十五" -> 复合量词）
            digit.append("1")
            bit.append(line[i])
        elif ("零" == line[i]) and (number_flag == 0 or number_flag == 1):
            digit.append("0")
        elif ("零" == line[i]) and number_flag == 3:
            zero_flag = 1
        elif number_flag == 1 and line[i] in bits:
            # 数字后跟量词，进入量词状态
            number_flag = 3
            if line[i] == "千":
                # "千米"、"千克" 中的 "千" 不是量词
                if i < len(line) - 1:
                    if line[i + 1] in unit:
                        number_flag = -1
            if number_flag == 3:
                onebit = line[i]
                bit.append(onebit)
        elif number_flag == 3 and line[i] in bits:
            # 处理复合量词（如 "十万"、"百万"、"千万"）
            onebit = bit[-1] + line[i]
            if onebit in bits:
                bit[-1] = onebit
            else:
                number_flag = -2
        else:
            number_flag = -1
        if len(digit) > 0 and number_flag == -1:
            number_flag = -2
        if i == (len(line) - 1) and number_flag >= 0:
            number_flag = -1
        # 数字序列结束，将收集的数字和量词组合为阿拉伯数字
        if number_flag < 0:
            newdigit = ""
            if len(digit) > 0:
                # 补全量词：如 "一百八" -> "一百八十"（补 '十'）
                if len(bit) == 1 and zero_flag == 0 and bit[0] == "百" and len(bit) != len(digit):
                    bit.append("十")
                # 补零：如 "一百五" -> digit=[1,5], bit=[百,零]
                if len(digit) == (len(bit) + 1):
                    bit.append("零")
                if len(digit) == len(bit):
                    # 从低位到高位组装数字
                    for m in range(len(digit))[-1::-1]:
                        if int(bits[bit[m]]) == int(len(newdigit) + 1):
                            newdigit += digit[m]
                        else:
                            # 中间有缺失的位，补零
                            nu = int(bits[bit[m]]) - len(newdigit) - 1
                            for n in range(nu):
                                newdigit += "0"
                            newdigit += digit[m]
                    # 反转得到正确的数字字符串
                    for z in range(len(newdigit))[-1::-1]:
                        newline += newdigit[z]
                else:
                    # 无法匹配量词，直接拼接数字
                    newline += "".join(digit)
                bit = []
                digit = []
                zero_flag = 0
            else:
                newline += line[i]
            if number_flag == -2:
                newline += line[i]
            number_flag = 0
    return newline


def special(line):
    """将 Unicode 特殊符号转换为中文文字。

    处理数学运算符（÷×=+-）、温度单位（℃）、面积单位（㎡）、
    千分号（‰）、百分号（%）、小数点（.）、度分符号（°′）等。

    Args:
        line (str): 包含特殊符号的文本。

    Returns:
        str: 特殊符号已转换为中文的文本。
    """
    newline = ""
    angel = 0  # 用于判断 '°' 后面的 '′' 是否为角度的"分"
    for e in range(len(line)):
        if ord(line[e]) == 247:      # ÷ -> 除以
            newline += "除以"
        elif ord(line[e]) == 215:    # × -> 乘以
            newline += "乘以"
        elif ord(line[e]) == 61:     # = -> 等于
            newline += "等于"
        elif ord(line[e]) == 43:     # + -> 加
            newline += "加"
        elif ord(line[e]) == 45:     # - -> 负
            newline += "负"
        elif ord(line[e]) == 8451:   # ℃ -> 摄氏度
            newline += "摄氏度"
        elif ord(line[e]) == 13217:  # ㎡ -> 平方米
            newline += "平方米"
        elif ord(line[e]) == 8240 or ord(line[e]) == 65130:  # ‰ -> %
            newline += "%"
        elif ord(line[e]) == 46:     # . -> 点
            newline += "点"
        elif ord(line[e]) == 176:    # ° -> 度
            newline += "度"
            angel = 1
        elif ord(line[e]) == 8242 and angel == 1:  # ′ -> 分（角度）
            newline += "分"
        else:
            newline += line[e]
    return newline


def all_convert(content):
    """执行完整的文本格式化流程。

    依次调用 recoformat（中英文分词）、numbersingle（数字转中文）、
    ch_number2digit（中文数字转阿拉伯数字）、special（特殊符号转换）、
    scoreformat（最终格式化）。

    Args:
        content (str): 原始识别结果文本。

    Returns:
        str: 完全格式化后的文本。
    """
    content = recoformat(content)
    content = numbersingle(content)
    content = ch_number2digit(content)
    content = special(content)
    content = scoreformat("", content)
    return content


if __name__ == "__main__":
    if len(sys.argv[1:]) < 1:
        sys.stderr.write("Usage:\n .py  reco.result\n")
        sys.stderr.write(" reco.result:   id<tab>recoresult\n")
        sys.exit(1)
    f = open(sys.argv[1])
    flag = 0
    if len(sys.argv[1:]) > 1:
        flag = int(sys.argv[2])
    for line in f.readlines():
        if not line:
            continue
        line = line.rstrip()
        tmp = line.split("\t")
        if len(tmp) < 2:
            tmp = line.split(",")
            if len(tmp) < 2:
                tmp = line.split(" ", 1)
                if len(tmp) < 2:
                    name = tmp[0]
                    content = ""
                    print(content)
                    continue
        name = tmp[0]
        content = tmp[1]
        name = re.sub("\.pcm", "", name)
        name = re.sub("\.wav", "", name)
        content = recoformat(content)
        content = numbersingle(content)
        content = ch_number2digit(content)
        content = special(content)
        content = scoreformat(name, content, flag)
        print(content)
    f.close()
