import logging
import inspect
from dataclasses import dataclass
import re
from typing import Any, Callable


@dataclass
class RegisterTables:
    """
    通用注册表（Registry）系统。

    核心设计思想：
    1. **集中式管理**：将项目中不同类型的组件（如模型、前端、编码器、解码器等）通过字典统一注册管理。
    2. **解耦与扩展**：通过装饰器 `@tables.register(...)` 的形式，将类的定义与其使用位置解耦。新增算法或模型时，只需在类定义上方添加装饰器即可完成注册，无需手动修改工厂类或配置映射，极大地方便了代码的扩展和维护。
    3. **键值对存储**：以类名（或指定的别名）作为 Key，类对象本身作为 Value。运行时可通过 Key 快速查找并实例化对应的类。
    4. **元数据追踪**：除了存储类本身，还额外记录了类所在的文件路径和行号（_meta 字典），便于调试和打印注册表详情。

    常见用途：
    - 模型架构切换：根据配置文件中的 "model" 字段动态加载对应的 Model 类。
    - 算法模块替换：方便替换不同的 Encoder、Decoder 或 Backend 实现。
    """

    # -------------------------------------------------------------------------
    # 注册表字典：用于存放不同类型的组件类。Key 为类名或别名，Value 为类对象。
    # -------------------------------------------------------------------------
    model_classes = {}          # 模型类（如 Paraformer, SenseVoice 等）
    frontend_classes = {}       # 前端处理类（如 WavFrontend 提取特征）
    specaug_classes = {}        # 数据增强类（如 SpecAugment）
    normalize_classes = {}      # 文本归一化类
    encoder_classes = {}        # 编码器类（如 Conformer, Transformer）
    decoder_classes = {}        # 解码器类（如 ParaformerDecoder）
    joint_network_classes = {}  # 连接网络类（用于 Encoder-Decoder 连接）
    predictor_classes = {}      # 预测器类（如 CIF Predictor）
    stride_conv_classes = {}    # 步长卷积类（用于降采样）
    tokenizer_classes = {}      # 分词器类
    dataloader_classes = {}    # 数据加载器类
    batch_sampler_classes = {} # Batch 采样器类
    dataset_classes = {}       # 数据集类
    index_ds_classes = {}      # 索引数据集类

    def print(self, key: str = None) -> None:
        """
        打印已注册的类信息（调试用）。

        Args:
            key: 如果提供，则只打印名称中包含该 key 的注册表；否则打印所有表。
        """
        print("\ntables: \n")
        fields = vars(self)
        headers = ["register name", "class name", "class location"]
        for classes_key, classes_dict in fields.items():
            # 只打印带有 "_meta" 后缀的字典（即元数据字典），因为它包含了可读的信息
            if classes_key.endswith("_meta") and (key is None or key in classes_key):
                print(f"-----------    ** {classes_key.replace('_meta', '')} **    --------------")
                metas = []
                for register_key, meta in classes_dict.items():
                    metas.append(meta)
                metas.sort(key=lambda x: x[0])
                data = [headers] + metas
                # 计算每列的最大宽度，用于格式化表格输出
                col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]

                for row in data:
                    print(
                        "| "
                        + " | ".join(str(item).ljust(width) for item, width in zip(row, col_widths))
                        + " |"
                    )
        print("\n")

    def register(self, register_tables_key: str, key: str = None) -> Callable[..., Any]:
        """
        装饰器工厂方法。用于将一个类注册到指定的注册表中。

        Args:
            register_tables_key: 目标注册表的属性名（如 'model_classes', 'encoder_classes'）。
            key: 注册后使用的 Key，默认为类名（target_class.__name__）。

        Returns:
            一个装饰器函数，该函数接收一个类作为参数并将其注册。

        使用示例:
            @tables.register("model_classes", key="Paraformer")
            class Paraformer:
                ...
        """

        def decorator(target_class):
            """
            实际的装饰器函数。
            """
            # 1. 检查对应的注册表是否存在，如果不存在则动态创建
            if not hasattr(self, register_tables_key):
                # 动态创建新的注册表字典：如果指定的注册表属性不存在，
                # 则在当前实例上创建一个空字典，用于存储该类型的组件类
                setattr(self, register_tables_key, {})
                # 记录日志：通知开发者有新的注册表被动态创建，便于调试和追踪
                logging.debug(f"New registry table added: {register_tables_key}")

            # 2. 获取注册表，并确定要使用的 Key
            # registry 是一个字典（dict）类型，用于存储注册表中的键值对，Key 为注册名，Value 为类对象
            registry = getattr(self, register_tables_key)
            registry_key = key if key is not None else target_class.__name__

            # 3. 如果 Key 已经存在，打印调试日志（允许覆盖）
            if registry_key in registry:
                logging.debug(
                    f"Key {registry_key} already exists in {register_tables_key}, re-register"
                )

            # 4. 将类注册到字典中
            # 将目标类以 registry_key 为键存入注册表字典中，完成类的注册
            # 实际效果：建立从注册名到类对象的映射，使得后续可以通过注册名查找并实例化该类
            registry[registry_key] = target_class
            # 5. 注册元数据（_meta），用于记录类的位置等信息
            register_tables_key_meta = register_tables_key + "_meta"
            if not hasattr(self, register_tables_key_meta):
                setattr(self, register_tables_key_meta, {})
            registry_meta = getattr(self, register_tables_key_meta)

            # 通过 inspect 获取类定义所在的文件路径和行号
            class_file = inspect.getfile(target_class)
            class_line = inspect.getsourcelines(target_class)[1]
            # 将绝对路径中的绝对前缀替换为 "funasr/"，使其更具可读性
            pattern = r"^.+/funasr/"
            class_file = re.sub(pattern, "funasr/", class_file)
            meta_data = [
                registry_key,                   # 注册名
                target_class.__name__,         # 类名
                f"{class_file}:{class_line}",  # 类所在位置（文件:行号）
            ]
            registry_meta[registry_key] = meta_data
            
            # 返回原类，保证装饰器不改变类的行为
            return target_class

        return decorator


# 全局单例实例：整个项目通过此实例进行组件注册
# 
# 注册表结构示意图：
# 
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                           RegisterTables (tables)                           │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
# │  │  model_classes  │    │ encoder_classes │    │ decoder_classes │  ...     │
# │  │  (dict)         │    │  (dict)         │    │  (dict)         │         │
# │  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤         │
# │  │ "Paraformer" ──►│───►│ Paraformer类     │    │                 │         │
# │  │ "SenseVoice"──►│───►│ SenseVoice类     │    │                 │         │
# │  │     ...         │    │     ...         │    │                 │         │
# │  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
# │                                                                             │
# │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
# │  │model_classes_meta│   │encoder_classes_ │    │decoder_classes_ │  ...     │
# │  │    (dict)        │   │    meta (dict)  │    │    meta (dict)  │         │
# │  ├─────────────────┤    ├─────────────────┤    ├─────────────────┤         │
# │  │ "Paraformer" ──►│───►│[注册名,类名,位置]│    │                 │         │
# │  │ "SenseVoice"──►│───►│[注册名,类名,位置]│    │                 │         │
# │  │     ...         │    │     ...         │    │                 │         │
# │  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
# │                                                                             │
# │  说明：                                                                      │
# │  1. 每个 xxx_classes 字典存储 注册名 → 类对象 的映射                          │
# │  2. 每个 xxx_classes_meta 字典存储 注册名 → 元数据列表[注册名,类名,文件位置]    │
# │  3. 通过 @tables.register("model_classes") 装饰器完成注册                    │
# │  4. 运行时通过 tables.model_classes["Paraformer"] 获取类并实例化             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
tables = RegisterTables()
