"""
傅里叶变换 (Fourier Transform) 可视化演示

目标：让阅读者直观理解傅里叶变换的本质——
      "任何周期信号都可以分解为一系列不同频率的正弦波之和"

通过动画和静态图展示：
    1. 时域 → 频域的转换过程
    2. 叠加正弦波如何合成方波
    3. 傅里叶变换如何"抽取"频率成分
    4. 真实语音信号的频谱分析

运行: python fourier_visualization.py
依赖: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，保存图片
from matplotlib.animation import FuncAnimation, PillowWriter

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "fourier_output"  # 图片输出目录
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# 图 1: 傅里叶变换的本质——叠加正弦波合成方波
# ═══════════════════════════════════════════════════════════════════
def plot_fourier_synthesis():
    """演示：用一系列正弦波合成方波

    方波的傅里叶级数公式：
        square(t) = (4/π) * Σ[sin((2n-1)ωt) / (2n-1)]  n=1,2,3,...

    当叠加的正弦波越多，合成波形越接近理想方波
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("傅里叶变换的本质：用正弦波合成方波", fontsize=16, fontweight="bold")

    t = np.linspace(0, 2 * np.pi, 1000)

    # 子图 1: 只叠加 1 个正弦波（基频）
    ax1 = axes[0, 0]
    n1 = 1
    wave1 = np.sin(n1 * t) / n1
    ax1.plot(t, wave1, "b-", linewidth=1.5)
    ax1.set_title(f"叠加 1 个正弦波 (n={n1})", fontsize=12)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_xlabel("时间 t")

    # 子图 2: 叠加 3 个正弦波
    ax2 = axes[0, 1]
    n_values = [1, 3, 5]
    wave3 = sum(np.sin(n * t) / n for n in n_values)
    for n in n_values:
        ax2.plot(t, np.sin(n * t) / n, "--", alpha=0.4, linewidth=0.8)
    ax2.plot(t, wave3, "r-", linewidth=2)
    ax2.set_title(f"叠加 3 个正弦波 (n={n_values})", fontsize=12)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_xlabel("时间 t")

    # 子图 3: 叠加 10 个正弦波
    ax3 = axes[1, 0]
    n_values = list(range(1, 20, 2))
    wave10 = sum(np.sin(n * t) / n for n in n_values)
    for n in n_values:
        ax3.plot(t, np.sin(n * t) / n, "--", alpha=0.2, linewidth=0.5)
    ax3.plot(t, wave10, "r-", linewidth=2)
    ax3.set_title(f"叠加 10 个正弦波", fontsize=12)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_xlabel("时间 t")

    # 子图 4: 叠加 100 个正弦波 → 接近理想方波
    ax4 = axes[1, 1]
    n_values = list(range(1, 100, 2))
    wave100 = sum(np.sin(n * t) / n for n in n_values)
    ax4.plot(t, wave100, "r-", linewidth=2.5, label="合成波形")
    ax4.set_title("叠加 100 个正弦波 ≈ 理想方波", fontsize=12)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_xlabel("时间 t")

    # 添加方波参考线
    square_wave = np.sign(np.sin(t))
    ax4.plot(t, square_wave, "k--", linewidth=1, alpha=0.5, label="理想方波")
    ax4.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_fourier_synthesis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 已保存: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 图 2: 时域 → 频域 的转换
# ═══════════════════════════════════════════════════════════════════
def plot_time_frequency_domain():
    """演示：同一个信号在时域和频域的不同表现

    时域：展示信号随时间的变化
    频域：展示信号包含哪些频率成分
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("时域 → 频域：同一段信号的两种视角", fontsize=16, fontweight="bold")

    # 生成一个复合信号：3Hz + 7Hz + 12Hz
    fs = 100  # 采样率 100Hz
    duration = 2  # 2 秒
    t = np.linspace(0, duration, fs * duration)

    signal = (
        np.sin(2 * np.pi * 3 * t) * 1.0 +      # 3Hz 频率，幅度 1.0
        np.sin(2 * np.pi * 7 * t) * 0.7 +      # 7Hz 频率，幅度 0.7
        np.sin(2 * np.pi * 12 * t) * 0.5       # 12Hz 频率，幅度 0.5
    )

    # 时域图
    ax1 = axes[0]
    ax1.plot(t, signal, "b-", linewidth=1.2)
    ax1.set_xlim(0, 1)  # 只显示前 1 秒
    ax1.set_xlabel("时间 (秒)", fontsize=12)
    ax1.set_ylabel("幅度", fontsize=12)
    ax1.set_title("时域信号：只能看到复杂波形，看不出频率成分", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 频域图（FFT）
    ax2 = axes[1]
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_mag = np.abs(np.fft.rfft(signal)) * 2 / n  # 归一化幅度
    fft_mag[0] /= 2  # DC 分量特殊处理

    markerline, stemlines, baseline = ax2.stem(freqs, fft_mag, linefmt="r-", basefmt=" ")
    plt.setp(markerline, markersize=4)
    ax2.set_xlim(0, 25)
    ax2.set_xlabel("频率 (Hz)", fontsize=12)
    ax2.set_ylabel("幅度", fontsize=12)
    ax2.set_title("频域信号：清晰看到 3Hz、7Hz、12Hz 三个频率成分", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 标注三个频率峰
    for freq, amp in [(3, 1.0), (7, 0.7), (12, 0.5)]:
        ax2.annotate(
            f"{freq}Hz\n幅度={amp}",
            xy=(freq, amp),
            xytext=(freq + 1, amp + 0.1),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
        )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_time_frequency_domain.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 已保存: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 图 3: 傅里叶变换如何"抽取"频率成分
# ═══════════════════════════════════════════════════════════════════
def plot_frequency_extraction():
    """演示：傅里叶变换如何"筛选"出特定频率的正弦波

    核心思想：将信号与不同频率的正弦波相乘，
             如果信号中包含该频率，乘积的平均值就不为零
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("傅里叶变换如何'抽取'频率成分", fontsize=16, fontweight="bold")

    t = np.linspace(0, 2, 200)
    # 复合信号：3Hz + 7Hz
    signal = np.sin(2 * np.pi * 3 * t) + 0.7 * np.sin(2 * np.pi * 7 * t)

    # 子图 1: 原始信号
    ax1 = axes[0, 0]
    ax1.plot(t, signal, "b-", linewidth=1.5)
    ax1.set_title("原始信号 (3Hz + 7Hz)", fontsize=12)
    ax1.set_xlabel("时间 (秒)")
    ax1.set_ylabel("幅度")
    ax1.set_xlim(0, 1)

    # 子图 2: 用 3Hz 正弦波"探测" → 乘积有规律 → 说明包含 3Hz
    ax2 = axes[0, 1]
    probe_3hz = np.sin(2 * np.pi * 3 * t)
    product_3hz = signal * probe_3hz
    ax2.plot(t, product_3hz, "r-", linewidth=1.5)
    ax2.axhline(y=np.mean(product_3hz), color="green", linestyle="--", linewidth=2)
    ax2.set_title("信号 × 3Hz 正弦波\n(乘积均值 ≠ 0 → 包含 3Hz)", fontsize=12)
    ax2.set_xlabel("时间 (秒)")
    ax2.set_xlim(0, 1)

    # 子图 3: 用 5Hz 正弦波"探测" → 乘积正负抵消 → 说明不包含 5Hz
    ax3 = axes[1, 0]
    probe_5hz = np.sin(2 * np.pi * 5 * t)
    product_5hz = signal * probe_5hz
    ax3.plot(t, product_5hz, "gray", linewidth=1.5)
    ax3.axhline(y=np.mean(product_5hz), color="green", linestyle="--", linewidth=2)
    ax3.set_title("信号 × 5Hz 正弦波\n(乘积均值 ≈ 0 → 不含 5Hz)", fontsize=12)
    ax3.set_xlabel("时间 (秒)")
    ax3.set_xlim(0, 1)

    # 子图 4: 用 7Hz 正弦波"探测" → 乘积有规律 → 说明包含 7Hz
    ax4 = axes[1, 1]
    probe_7hz = np.sin(2 * np.pi * 7 * t)
    product_7hz = signal * probe_7hz
    ax4.plot(t, product_7hz, "r-", linewidth=1.5)
    ax4.axhline(y=np.mean(product_7hz), color="green", linestyle="--", linewidth=2)
    ax4.set_title("信号 × 7Hz 正弦波\n(乘积均值 ≠ 0 → 包含 7Hz)", fontsize=12)
    ax4.set_xlabel("时间 (秒)")
    ax4.set_xlim(0, 1)

    # 添加说明文本
    fig.text(
        0.5, 0.01,
        "核心思想：将信号与某频率正弦波相乘，若信号含该频率则乘积均值≠0，\n"
        "若不含则正负抵消均值≈0。对所有频率做此操作→得到频谱。",
        ha="center", fontsize=11, style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(OUTPUT_DIR, "03_frequency_extraction.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 已保存: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 图 4: 语音信号的频谱分析（FunASR 实际用到的）
# ═══════════════════════════════════════════════════════════════════
def plot_speech_spectrum():
    """演示：真实语音信号的时域和频域特征

    展示 FunASR 前端如何使用傅里叶变换提取声学特征（fbank）
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("语音信号的频谱分析：FunASR 如何'听懂'声音", fontsize=16, fontweight="bold")

    # 模拟一段语音信号（用几个基频 + 谐波模拟）
    fs = 16000  # 16kHz 采样率（FunASR 使用的采样率）
    duration = 0.5  # 0.5 秒
    t = np.linspace(0, duration, int(fs * duration))

    # 模拟元音 /a/ 的频谱：基频 100Hz + 谐波
    f0 = 100  # 基频
    signal = sum(
        (1 / n) * np.sin(2 * np.pi * f0 * n * t + np.random.uniform(-0.3, 0.3))
        for n in range(1, 15)
    )
    # 添加轻微的幅度包络
    envelope = np.exp(-((t - 0.25) ** 2) / 0.02)
    signal *= envelope

    # 子图 1: 时域波形
    ax1 = axes[0]
    ax1.plot(t * 1000, signal, "b-", linewidth=0.8)
    ax1.set_xlabel("时间 (毫秒)", fontsize=12)
    ax1.set_ylabel("幅度", fontsize=12)
    ax1.set_title("时域波形：看到的是复杂的声波振动", fontsize=12)
    ax1.set_xlim(0, duration * 1000)

    # 子图 2: 完整频谱（FFT）
    ax2 = axes[1]
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_mag = np.abs(np.fft.rfft(signal))
    # 取对数（模拟人耳对声音的感知）
    log_mag = np.log(fft_mag + 1e-10)

    ax2.plot(freqs, log_mag, "r-", linewidth=1)
    ax2.set_xlim(0, 4000)  # 语音主要能量在 4kHz 以下
    ax2.set_xlabel("频率 (Hz)", fontsize=12)
    ax2.set_ylabel("对数幅度", fontsize=12)
    ax2.set_title("频域频谱：清晰看到基频 100Hz 及谐波结构", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 标注基频和谐波
    for n in range(1, 6):
        freq = f0 * n
        if freq <= 4000:
            ax2.axvline(x=freq, color="green", linestyle="--", alpha=0.3, linewidth=0.8)
            ax2.annotate(
                f"{freq}Hz",
                xy=(freq, log_mag[min(int(freq / (fs / 2) * n), len(log_mag) - 1)]),
                fontsize=8,
                color="green",
            )

    # 子图 3: 梅尔频谱（fbank 的核心步骤）
    ax3 = axes[2]
    # 简化的梅尔频谱：将频率轴转换为梅尔刻度
    mel_filterbank = np.zeros((40, len(freqs)))
    for i in range(40):
        center_freq = 700 * (10 ** (i * 0.5) - 1) / 2595
        bandwidth = center_freq * 0.3
        mel_filterbank[i, :] = np.exp(-0.5 * ((freqs - center_freq) / bandwidth) ** 2)

    mel_energy = mel_filterbank @ fft_mag
    log_mel_energy = np.log(mel_energy + 1e-10)

    # 绘制梅尔滤波器组
    for i in range(0, 40, 4):
        ax3.plot(freqs, mel_filterbank[i, :], "--", alpha=0.4, linewidth=0.8)

    ax3.set_xlim(0, 4000)
    ax3.set_xlabel("频率 (Hz)", fontsize=12)
    ax3.set_ylabel("滤波器响应", fontsize=12)
    ax3.set_title("梅尔滤波器组 (40 个滤波器)：模拟人耳感知", fontsize=12)

    # 添加说明
    fig.text(
        0.5, 0.01,
        "FunASR 的特征提取流程：\n"
        "时域波形 → [傅里叶变换] → 频域频谱 → [梅尔滤波器] → fbank 特征 → 送入模型",
        ha="center", fontsize=11, style="italic",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUTPUT_DIR, "04_speech_spectrum.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 已保存: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 图 5: 动画——叠加正弦波合成方波的过程
# ═══════════════════════════════════════════════════════════════════
def create_animation():
    """创建动画：逐步叠加正弦波，观察波形如何趋近方波"""
    print("\n🎬 正在创建动画...")

    fig, ax = plt.subplots(figsize=(12, 6))
    t = np.linspace(0, 2 * np.pi, 1000)

    def update(frame):
        ax.clear()
        n_waves = frame * 2 + 1  # 1, 3, 5, 7, ...
        n_values = list(range(1, n_waves + 1, 2))
        wave = sum(np.sin(n * t) / n for n in n_values)

        # 绘制各正弦波分量
        for n in n_values:
            ax.plot(t, np.sin(n * t) / n, "--", alpha=0.3, linewidth=0.8, color="gray")

        # 绘制合成波形
        ax.plot(t, wave, "r-", linewidth=2.5, label=f"合成波形 ({n_waves} 个正弦波)")

        # 绘制理想方波
        square = np.sign(np.sin(t))
        ax.plot(t, square, "k--", linewidth=1, alpha=0.6, label="理想方波")

        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xlabel("时间 t", fontsize=12)
        ax.set_ylabel("幅度", fontsize=12)
        ax.set_title(f"叠加 {n_waves} 个正弦波 → 逼近方波", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # 创建动画（10 帧，间隔 300ms）
    anim = FuncAnimation(fig, update, frames=10, interval=300, repeat=True)

    # 保存 GIF
    path = os.path.join(OUTPUT_DIR, "05_fourier_animation.gif")
    writer = PillowWriter(fps=3)
    anim.save(path, writer=writer)
    plt.close()
    print(f"✅ 已保存: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 主函数：生成所有可视化
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("傅里叶变换可视化演示")
    print("=" * 60)
    print(f"\n📁 图片输出目录: {OUTPUT_DIR}/")
    print("\n生成中...\n")

    plot_fourier_synthesis()
    plot_time_frequency_domain()
    plot_frequency_extraction()
    plot_speech_spectrum()

    # 动画需要 Pillow，可选生成
    try:
        create_animation()
    except ImportError:
        print("\n⚠️  跳过动画生成（需要 Pillow: pip install Pillow）")

    print("\n" + "=" * 60)
    print("🎉 所有图片生成完毕！")
    print(f"📂 查看: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)

    # 打印阅读指南
    print("\n📖 阅读指南：")
    print("  1. 01_fourier_synthesis.png  — 核心思想：正弦波叠加合成方波")
    print("  2. 02_time_frequency_domain.png — 时域 vs 频域的对比")
    print("  3. 03_frequency_extraction.png  — 傅里叶变换的'探测'原理")
    print("  4. 04_speech_spectrum.png       — 语音信号的实际频谱分析")
    print("  5. 05_fourier_animation.gif    — 动画演示叠加过程")


if __name__ == "__main__":
    main()
