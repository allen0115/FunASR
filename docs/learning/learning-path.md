# FunASR 学习路线

## 第一站：从 CLI 入口理解调用链路

核心入口文件：
- `funasr/auto/auto_model.py` — AutoModel，所有推理的统一入口
- `funasr/bin/inference.py` — CLI 推理入口
- `funasr/bin/server.py` — OpenAI 兼容 API 服务

先跑通一个推理（`funasr model="iic/SenseVoiceSmall" input="test.wav"`），然后顺着 `AutoModel.generate()` 读下去，理解 下载模型 → 注册发现 → 实例化 → 推理 的完整链路。

## 第二站：理解注册系统

- `funasr/register.py` — 装饰器注册表，整个框架的依赖注入核心

所有模型、前端、tokenizer 都通过 `@tables.register("model_classes", "MyModel")` 注册，`AutoModel` 运行时从注册表发现并实例化。

## 第三站：挑一个简单模型读懂结构

推荐从最小的模型开始：
- `funasr/models/fun_asr_nano/` — 轻量级 ASR 模型，代码量小，适合入门
- `funasr/models/sense_voice/` — SenseVoice，FunASR 核心模型，工业级实现

每个模型目录一般包含：
- `model.py` — `nn.Module`，架构定义 + `forward()` + `from_pretrained()`
- `template.yaml` — 默认配置/超参

## 第四站：按需深入子系统

| 感兴趣方向 | 对应目录 |
|-----------|---------|
| 音频特征提取 | `funasr/frontends/` |
| 文本 token 化 | `funasr/tokenizer/` |
| 训练流程 | `funasr/train_utils/`（支持 DeepSpeed 分布式训练） |
| 数据集加载 | `funasr/datasets/` |
| VAD/标点/说话人分离 | `funasr/utils/` → 查 postprocess、VAD、diarization 相关 |
| 生产部署 | `runtime/`（gRPC/WebSocket/ONNX/Android/iOS/HTML5） |

## 第五站：看 examples 和 tests

- `examples/` — 各场景的使用示例
- `tests/` — 测试用例，也是 API 用法参考
- `tests_models/` — 模型级测试

## 建议入手顺序

`register.py` → `auto_model.py` → `fun_asr_nano/model.py` → `sense_voice/model.py`

读完这四个文件，基本就能理解 FunASR 的核心设计了。
