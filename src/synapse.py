# -*- coding: utf-8 -*-
"""
步骤①②：有机光电突触器件电导模型 + 非线性度/不对称比提取
==========================================================

器件电导更新模型（指数模型）::

    G_{n+1} = G_n ± α · exp[ −β · (G_n − G_min) / (G_max − G_min) ]

其中
  α  : 单脉冲步长幅度（自动标定，使 N 个脉冲恰好扫满电导窗口）
  β  : 非线性因子。β > 0 → 增强曲线（先快后慢，饱和）；
       β < 0 → 抑制曲线（电导越高步长越大）。
  n  : 已施加的电脉冲个数

提取的两个器件指标：
  非线性度 NL = max_n |G(n) − G_lin(n)| / (G_max − G_min)
      G_lin(n) 为从 G(0) 到 G(N) 的理想直线，NL 越接近 0 越线性；
  不对称比 AR：由于增强/抑制两条曲线分别从窗口两端出发，
      直接按 max|G_p(n) − G_d(n)| / (G_p(N) − G_d(N)) 计算恒等于 1（端点处
      电导差总等于窗口值），因此文献通行做法是把抑制曲线镜像到增强方向后
      再比较。本脚本两种都给出，训练中采用的是镜像版本的物理含义。

运行（在项目根目录下）::

    python src/synapse.py

输出：终端打印 NL / AR，figures/ 下生成 fig1、fig2 两张图。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无窗口环境下也能出图
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")


# ----------------------------------------------------------------------------
# ① 电导更新曲线 G(n)
# ----------------------------------------------------------------------------
def pulse_sweep(alpha, beta, g_min, g_max, n_pulses, potentiate):
    """施加 n_pulses 个同极性脉冲，返回长度为 n_pulses+1 的电导序列 G(n)。

    potentiate=True  : 增强（电导从 g_min 出发向上），β 应 > 0
    potentiate=False : 抑制（电导从 g_max 出发向下），β 应 < 0
    """
    R = g_max - g_min
    G = np.empty(n_pulses + 1)
    G[0] = g_min if potentiate else g_max
    for i in range(n_pulses):
        x = (G[i] - g_min) / R                     # 归一化电导 ∈ [0, 1]
        dG = alpha * np.exp(-beta * x)             # 指数模型的单脉冲步长
        G[i + 1] = np.clip(G[i] + (dG if potentiate else -dG), g_min, g_max)
    return G


def calibrate_alpha(beta, g_min, g_max, n_pulses, potentiate):
    """二分法标定步长 α，使 n_pulses 个脉冲恰好扫满 [g_min, g_max] 窗口。"""
    target = g_max if potentiate else g_min
    lo, hi = 1e-12, (g_max - g_min)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        G = pulse_sweep(mid, beta, g_min, g_max, n_pulses, potentiate)
        if potentiate:               # 还没升到 G_max → α 太小
            reached = G[-1] >= target - 1e-12
        else:                        # 还没降到 G_min → α 太小
            reached = G[-1] <= target + 1e-12
        if not reached:
            lo = mid
        else:                        # 提前到头（被截断）→ α 太大
            hi = mid
    return 0.5 * (lo + hi)


def measure_curves(g_min, g_max, beta_p, beta_d, n_pulses=100):
    """返回 (G_p, G_d) 两条实测曲线，α 已自动标定。"""
    a_p = calibrate_alpha(beta_p, g_min, g_max, n_pulses, potentiate=True)
    a_d = calibrate_alpha(beta_d, g_min, g_max, n_pulses, potentiate=False)
    G_p = pulse_sweep(a_p, beta_p, g_min, g_max, n_pulses, potentiate=True)
    G_d = pulse_sweep(a_d, beta_d, g_min, g_max, n_pulses, potentiate=False)
    return G_p, G_d


# ----------------------------------------------------------------------------
# ② 指标提取
# ----------------------------------------------------------------------------
def nonlinearity(G, g_min, g_max):
    """非线性度 NL：与端点连线的最大偏差 / 电导窗口。"""
    G_lin = np.linspace(G[0], G[-1], len(G))
    return np.max(np.abs(G - G_lin)) / (g_max - g_min)


def asymmetry_ratio(G_p, G_d, g_min, g_max):
    """不对称比 AR。

    镜像版（文献常用，也是本项目训练中使用的定义）：
        把抑制曲线翻转到增强方向 G_d'(n) = G_max + G_min − G_d(n)，
        AR = max_n |G_p(n) − G_d'(n)| / (G_max − G_min)
    完美对称的器件 AR = 0；AR 越大表示增强/抑制越不对称。
    """
    G_d_mirror = (g_max + g_min) - G_d
    return np.max(np.abs(G_p - G_d_mirror)) / (g_max - g_min)


def asymmetry_ratio_raw(G_p, G_d):
    """题述原始公式 AR = max|G_p(n)−G_d(n)| / (G_p(N)−G_d(N))。

    注意：两条曲线分别从窗口两端出发时，n=0 处电导差恒等于窗口值，
    该式对理想扫满窗口的曲线恒返回 1，仅当曲线未扫满窗口时才有区分度。
    """
    return np.max(np.abs(G_p - G_d)) / abs(G_p[-1] - G_d[-1])


# ----------------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------------
def plot_fig1(G_p, G_d, g_min, g_max, path):
    """图1：(a) 增强/抑制曲线与理想直线；(b) 单脉冲步长 ΔG 随脉冲数的变化。"""
    n = np.arange(len(G_p))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(n, G_p, "o-", ms=3, color="tab:red", label="Potentiation $G_p(n)$")
    ax.plot(n, G_d, "s-", ms=3, color="tab:blue", label="Depression $G_d(n)$")
    ax.plot(n, np.linspace(G_p[0], G_p[-1], len(n)), "--", color="gray",
            lw=1, label="ideal linear")
    ax.set_xlabel("Pulse number $n$")
    ax.set_ylabel("Conductance $G$ (µS)")
    nl_p, nl_d = nonlinearity(G_p, g_min, g_max), nonlinearity(G_d, g_min, g_max)
    ar = asymmetry_ratio(G_p, G_d, g_min, g_max)
    ax.set_title(f"(a) Conductance update curves\n"
                 f"NL$_p$={nl_p*100:.1f}%, NL$_d$={nl_d*100:.1f}%, AR={ar:.2f}")
    ax.legend()

    ax = axes[1]
    dG_p = np.diff(G_p)
    dG_d = -np.diff(G_d)
    ax.plot(n[1:], dG_p, "o-", ms=3, color="tab:red", label="Δ$G_p$ (POT)")
    ax.plot(n[1:], dG_d, "s-", ms=3, color="tab:blue", label="|Δ$G_d$| (DEP)")
    ax.set_xlabel("Pulse number $n$")
    ax.set_ylabel("Step size |ΔG| (µS / pulse)")
    ax.set_title("(b) Pulse-by-pulse step size")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fig2(g_min, g_max, betas, n_pulses, path):
    """图2：不同非线性因子 β 下的增强曲线族。"""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(betas):
        G_p, _ = measure_curves(g_min, g_max, b, -abs(b), n_pulses)
        nl = nonlinearity(G_p, g_min, g_max)
        ax.plot(np.arange(len(G_p)), G_p, color=cmap(i / max(len(betas) - 1, 1)),
                label=f"$\\beta$={b}, NL={nl*100:.0f}%")
    ax.set_xlabel("Pulse number $n$")
    ax.set_ylabel("Conductance $G$ (µS)")
    ax.set_title("Potentiation curves vs. nonlinearity factor $\\beta$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- 器件参数（示例值；有实测数据时替换 g_min/g_max/beta 即可）----
    g_min, g_max = 0.1, 9.9        # 电导窗口 (µS)
    beta_p, beta_d = 2.0, -4.0     # 增强为正、抑制为负
    n_pulses = 100                 # 每条曲线的脉冲数

    G_p, G_d = measure_curves(g_min, g_max, beta_p, beta_d, n_pulses)

    nl_p = nonlinearity(G_p, g_min, g_max)
    nl_d = nonlinearity(G_d, g_min, g_max)
    ar = asymmetry_ratio(G_p, G_d, g_min, g_max)
    ar_raw = asymmetry_ratio_raw(G_p, G_d)

    print("=" * 60)
    print("有机光电突触器件电导特性（指数更新模型）")
    print("-" * 60)
    print(f"参数: G_min={g_min} µS, G_max={g_max} µS, N={n_pulses} pulses")
    print(f"      beta_p={beta_p}, beta_d={beta_d}")
    print("-" * 60)
    print(f"非线性度   NL_p = {nl_p*100:6.2f} %   (增强曲线)")
    print(f"           NL_d = {nl_d*100:6.2f} %   (抑制曲线)")
    print(f"不对称比   AR   = {ar:6.3f}      (镜像定义, 0=完全对称)")
    print(f"           AR'  = {ar_raw:6.3f}      (题述原始公式)")
    print("=" * 60)

    plot_fig1(G_p, G_d, g_min, g_max, os.path.join(FIG_DIR, "fig1_conductance_curves.png"))
    plot_fig2(g_min, g_max, betas=[0.0, 1.0, 2.0, 4.0, 8.0], n_pulses=n_pulses,
              path=os.path.join(FIG_DIR, "fig2_beta_family.png"))
    print("图已保存: figures/fig1_conductance_curves.png, figures/fig2_beta_family.png")


if __name__ == "__main__":
    main()
