# -*- coding: utf-8 -*-
"""
步骤④：用突触器件权重训练 MLP 做 MNIST 手写数字分类
====================================================

两种模式对比：
  --mode software : 理想浮点 SGD 基线（上限，代表"完美器件"）
  --mode device   : 差分对突触器件仿真（受 NL/AR/噪声/整数脉冲限制）

运行示例（在项目根目录下）::

    python src/train.py --mode software --epochs 3
    python src/train.py --mode device   --epochs 3
    python src/train.py --mode device   --epochs 1 --subset 10000   # 快速试跑

首次运行会自动下载 MNIST（约 11 MB）到 ./data。
输出：终端打印每个 epoch 的测试精度，results/ 下保存 CSV，
      figures/ 下保存精度曲线 fig3_accuracy.png。
"""

import argparse
import csv
import os
import sys
import time

# Windows 上 torch 与 matplotlib 各自带了一份 OpenMP 运行库，需放宽重复检测
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layers import DeviceParams, SynapseMLP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")


# ----------------------------------------------------------------------------
def get_loaders(args):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = torchvision.datasets.MNIST(args.data_dir, train=True,
                                           download=True, transform=tf)
    test_set = torchvision.datasets.MNIST(args.data_dir, train=False,
                                          download=True, transform=tf)
    if args.subset > 0:
        train_set = Subset(train_set, range(args.subset))
    if args.test_subset > 0:
        test_set = Subset(test_set, range(args.test_subset))
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, generator=g, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


# ----------------------------------------------------------------------------
def train_device(args):
    """器件仿真模式：梯度由 PyTorch 算，权重更新严格走器件电导模型。"""
    params = DeviceParams(
        g_min=args.gmin, g_max=args.gmax,
        beta_p=args.beta_p, beta_d=-args.beta_d,
        n_pulses=args.n_pulses,
        read_noise=args.read_noise, update_noise=args.update_noise,
        discrete=args.discrete,
    )
    model = SynapseMLP(params, hidden=args.hidden)
    train_loader, test_loader = get_loaders(args)

    accs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, seen = time.time(), 0
        for x, y in train_loader:
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            model.device_update(args.lr)     # 关键：脉冲更新代替 optimizer.step()
            model.zero_grad()
            seen += y.numel()
        acc = evaluate(model, test_loader)
        accs.append(acc)
        print(f"[device] epoch {epoch}/{args.epochs}  "
              f"test acc = {acc*100:5.2f}%   ({time.time()-t0:.1f}s)", flush=True)
    return accs


def train_software(args):
    """理想浮点基线：普通 MLP + SGD。"""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, args.hidden), nn.ReLU(),
        nn.Linear(args.hidden, 10),
    )
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    train_loader, test_loader = get_loaders(args)

    accs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate(model, test_loader)
        accs.append(acc)
        print(f"[software] epoch {epoch}/{args.epochs}  "
              f"test acc = {acc*100:5.2f}%   ({time.time()-t0:.1f}s)", flush=True)
    return accs


def save_results(args, accs):
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    tag = args.mode
    csv_path = os.path.join(RES_DIR, f"accuracy_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "test_accuracy"])
        for i, a in enumerate(accs, 1):
            w.writerow([i, f"{a:.4f}"])
    print(f"结果已保存: {os.path.relpath(csv_path, ROOT)}")

    # 精度曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(accs) + 1), [a * 100 for a in accs], "o-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"MNIST 784-100-10 ({tag} mode, "
                 f"final {accs[-1]*100:.2f}%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"fig3_accuracy_{tag}.png")
    fig.savefig(fig_path, dpi=150)
    print(f"图已保存: {os.path.relpath(fig_path, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="有机光电突触 MNIST 分类仿真")
    ap.add_argument("--mode", choices=["software", "device"], default="device")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None,
                    help="software 模式默认 0.1；device 模式默认 0.1（含义不同，见 README）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    # ---- 器件参数（device 模式）----
    ap.add_argument("--gmin", type=float, default=0.1, help="G_min (µS)")
    ap.add_argument("--gmax", type=float, default=9.9, help="G_max (µS)")
    ap.add_argument("--beta-p", type=float, default=2.0, help="增强非线性因子 |β|")
    ap.add_argument("--beta-d", type=float, default=4.0, help="抑制非线性因子 |β|")
    ap.add_argument("--n-pulses", type=float, default=100, help="满窗口脉冲数")
    ap.add_argument("--read-noise", type=float, default=0.0, help="读取噪声 σ（窗口比例）")
    ap.add_argument("--update-noise", type=float, default=0.0, help="更新噪声 σ")
    ap.add_argument("--discrete", action="store_true", help="整数脉冲+随机舍入")
    # ---- 数据规模 ----
    ap.add_argument("--subset", type=int, default=0, help="训练子集大小（0=全部60000）")
    ap.add_argument("--test-subset", type=int, default=0, help="测试子集大小（0=全部10000）")
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 0.1 if args.mode == "software" else 0.1

    torch.manual_seed(args.seed)
    print(f"模式: {args.mode} | lr={args.lr} | epochs={args.epochs} | "
          f"subset={args.subset or 'all'}")
    if args.mode == "device":
        print(f"器件: G∈[{args.gmin},{args.gmax}] µS, β_p={args.beta_p}, "
              f"β_d={args.beta_d}, N={args.n_pulses}, "
              f"读噪声={args.read_noise}, 更新噪声={args.update_noise}, "
              f"整数脉冲={args.discrete}")

    accs = train_device(args) if args.mode == "device" else train_software(args)
    save_results(args, accs)


if __name__ == "__main__":
    main()
