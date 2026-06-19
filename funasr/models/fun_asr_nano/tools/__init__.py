"""Fun-ASR-Nano 工具包。

包含以下工具模块:
    - cn_tn: 中文文本归一化（NSW -> 中文表达）
    - format5res: ASR 识别结果格式化（数字、标点等后处理）
    - scp2jsonl: Kaldi scp + transcript 转 JSONL 训练数据
    - utils: 音频加载和 CTC 强制对齐工具函数
    - whisper_mix_normalize: 多语言混合文本归一化（中英日）
"""