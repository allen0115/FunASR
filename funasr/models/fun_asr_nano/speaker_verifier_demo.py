"""
声纹验证演示脚本 (Speaker Verification Demo)

演示如何使用 SpeakerVerifier 解决：
"用户录音提取声纹 → 与会议录音进行声纹识别" 的场景。

核心步骤：
    1. 初始化 SpeakerVerifier
    2. 注册用户声纹模板
    3. 对会议录音进行说话人身份验证
    4. 输出验证结果（相似度分数 + 是否匹配）

运行前准备：
    pip install soundfile librosa
    # 确保 FunASR 已安装
"""

import numpy as np
from funasr.models.fun_asr_nano.speaker_verifier import SpeakerVerifier


def demo_basic_usage():
    """演示基本用法：注册用户声纹 → 验证会议音频"""
    print("=" * 60)
    print("声纹验证：基本用法演示")
    print("=" * 60)

    # 1. 初始化验证器
    verifier = SpeakerVerifier(
        model="cam++",           # Cam++ 声纹模型
        device="cuda",           # 使用 GPU（无 GPU 时改为 "cpu"）
        threshold=0.6,           # 相似度阈值：高于此值判定为同一人
    )

    # 2. 注册用户声纹模板
    #    建议：每个用户用 3-5 秒安静环境录音，效果最佳
    print("\n📝 注册用户声纹...")
    verifier.register_user_from_file("user_001", "user_zhang_speech.wav")
    verifier.register_user_from_file("user_002", "user_li_speech.wav")

    print(f"   已注册用户: {verifier.list_registered_users()}")

    # 3. 验证会议音频
    #    输入会议录音，返回每个说话人片段的身份识别结果
    print("\n🔍 验证会议音频...")
    results = verifier.verify_meeting_audio(
        meeting_audio=verifier._load_audio("meeting_recording.wav"),
        user_ids=["user_001", "user_002"],
        vad_segments=None,  # 可选：传入 VAD 分段提高精度
    )

    # 4. 输出结果
    print("\n📊 验证结果:")
    for r in results:
        scores_str = ", ".join(f"{k}: {v:.3f}" for k, v in r["scores"].items())
        print(f"   [{r['start_ms']}ms - {r['end_ms']}ms] "
              f"最可能: {r['identified_user']} "
              f"(匹配: {r['is_match']})")
        print(f"      相似度: {scores_str}")


def demo_single_segment():
    """演示单片段验证"""
    print("\n" + "=" * 60)
    print("声纹验证：单片段验证演示")
    print("=" * 60)

    verifier = SpeakerVerifier(threshold=0.6)

    # 注册用户
    verifier.register_user_from_file("user_001", "user_zhang_speech.wav")

    # 加载会议音频
    meeting_audio = verifier._load_audio("meeting_recording.wav")

    # 取一个片段（2-4 秒）进行验证
    segment = meeting_audio[16000 * 10 : 16000 * 13]  # 第 10-13 秒

    print("\n🔍 验证单个片段...")
    score, is_match = verifier.verify_segment(segment, "user_001")

    print(f"   相似度: {score:.4f}")
    print(f"   阈值: {verifier.threshold}")
    print(f"   匹配结果: {'✅ 是同一人' if is_match else '❌ 不是同一人'}")


def demo_threshold_tuning():
    """演示阈值调优"""
    print("\n" + "=" * 60)
    print("声纹验证：阈值调优演示")
    print("=" * 60)

    verifier = SpeakerVerifier()
    verifier.register_user_from_file("user_001", "user_zhang_speech.wav")

    meeting_audio = verifier._load_audio("meeting_recording.wav")
    segment = meeting_audio[16000 * 10 : 16000 * 13]

    # 尝试不同阈值
    for threshold in [0.4, 0.5, 0.6, 0.7, 0.8]:
        verifier.set_threshold(threshold)
        score, is_match = verifier.verify_segment(segment, "user_001")
        status = "✅" if is_match else "❌"
        print(f"   阈值={threshold}: {status} 匹配 (相似度={score:.4f})")

    # 恢复默认
    verifier.set_threshold(0.6)


def demo_improve_accuracy_tips():
    """提高识别准确率的技巧"""
    print("\n" + "=" * 60)
    print("📈 提高识别准确率的关键技巧")
    print("=" * 60)

    tips = [
        ("1. 声纹注册音频质量",
         "用户注册时使用安静环境、近讲麦克风、3-5 秒清晰语音，"
         "避免电话音质、混响、背景噪声。"),
        ("2. 多环境注册",
         "同一用户在不同环境（安静办公室、在家）各录一段，"
         "将所有嵌入都注册为模板，提高泛化性。"),
        ("3. 足够长度",
         "注册音频至少 3 秒，验证音频至少 2 秒。"
         "太短会导致嵌入不稳定，太长会混入噪声。"),
        ("4. 避免重叠语音",
         "注册和验证音频都应避免多人同时说话（重叠语音），"
         "这会污染说话人嵌入。"),
        ("5. 阈值调优",
         "根据应用场景调整阈值："
         "安全敏感场景用 0.7+（更严格），"
         "便捷场景用 0.5-0.6（更宽松）。"),
        ("6. CMS 归一化",
         "SpeakerVerifier 已内置 CMS 归一化，"
         "可有效缓解不同录音设备和环境造成的域偏移。"),
        ("7. 多段平均",
         "注册时自动切分为多段取平均，"
         "比单段嵌入更稳定，是对抗域偏移的核心策略。"),
    ]

    for title, desc in tips:
        print(f"\n   {title}")
        print(f"   {desc}")


if __name__ == "__main__":
    print("FunASR 声纹验证模块演示")
    print("=" * 60)
    print("\n⚠️  运行前请准备以下音频文件：")
    print("    - user_zhang_speech.wav: 用户张的 3-5 秒录音")
    print("    - user_li_speech.wav: 用户李的 3-5 秒录音")
    print("    - meeting_recording.wav: 会议录音文件")
    print("\n📂 所有文件应为 16kHz 单声道 WAV 格式")

    # 取消注释以运行演示（需要实际音频文件）
    # demo_basic_usage()
    # demo_single_segment()
    # demo_threshold_tuning()

    # 始终显示技巧
    demo_improve_accuracy_tips()
