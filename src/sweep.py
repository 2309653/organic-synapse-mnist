# -*- coding: utf-8 -*-
"""
附加实验：器件非线性因子 β 对 MNIST 精度的影响
==============================================

固定其他条件，扫描 β（增强/抑制取相反符号的相同幅度），对每个 β：
  1. 计算该器件的 NL / AR（调用 synapse.py 的指标函数）；
  2. 用 device 模式短训练一轮，记录测试精度；
最后输出 "器件特性 → 网络精度" 对应表和曲线图。

运行（在项目根目录下）::

    python src/sweep.py --subset 10000 --epochs 1

输出：results/beta_sweep.csv, figures/fig4_beta_sweep.png
"""

import argparse
import csv
import os
import sys

# Windows 上 torch 与 matplotlib 各自带了一份 OpenMP 运行库，需放宽重复检测
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from layers import DeviceParams, SynapseMLP
from synapse import measure_curves, nonlinearity, asymmetry_ratio
from train import get_loaders, evaluate, ROOT, RES_DIR, FIG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--subset", type=int, default=10000)
    ap.add_argument("--test-subset", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    rows = []
    for b in args.betas:
        params = DeviceParams(beta_p=b if b > 0 else 1e-6,
                              beta_d=-b if b > 0 else -1e-6)
        # ①② 该 β 下的器件指标
        G_p, G_d = measure_curves(params.g_min, params.g_max,
                                  params.beta_p, params.beta_d, 100)
        nl_p = nonlinearity(G_p, params.g_min, params.g_max)
        nl_d = nonlinearity(G_d, params.g_min, params.g_max)
        ar = asymmetry_ratio(G_p, G_d, params.g_min, params.g_max)

        # ④ 短训练看精度
        torch.manual_seed(args.seed)
        model = SynapseMLP(params, hidden=100)
        train_loader, test_loader = get_loaders(args)
        for _ in range(args.epochs):
            for x, y in train_loader:
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                model.device_update(args.lr)
                model.zero_grad()
        acc = evaluate(model, test_loader)

        rows.append((b, nl_p, nl_d, ar, acc))
        print(f"β={b:4.1f} | NL_p={nl_p*100:5.1f}%  NL_d={nl_d*100:5.1f}%  "
              f"AR={ar:.2f} | test acc = {acc*100:5.2f}%", flush=True)

    # 保存 CSV
    csv_path = os.path.join(RES_DIR, "beta_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "NL_potentiation", "NL_depression",
                    "AR", "test_accuracy"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
                        f"{r[4]:.4f}"])
    print(f"表已保存: results/beta_sweep.csv")

    # 画图：双 y 轴（精度 + NL/AR 随 β）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    betas = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(betas, [r[4] * 100 for r in rows], "o-", color="tab:red",
            label="test accuracy")
    ax.set_xlabel("Nonlinearity factor $\\beta$")
    ax.set_ylabel("Test accuracy (%)", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(betas, [abs(r[1]) * 100 for r in rows], "s--", color="tab:blue",
             label="NL")
    ax2.plot(betas, [r[3] * 100 for r in rows], "^--", color="tab:green",
             label="AR")
    ax2.set_ylabel("NL / AR (%)", color="tab:blue")
    ax.set_title("Device nonlinearity vs. network accuracy")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig4_beta_sweep.png")
    fig.savefig(fig_path, dpi=150)
    print(f"图已保存: figures/fig4_beta_sweep.png")


if __name__ == "__main__":
    main()
