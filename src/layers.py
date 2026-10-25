# -*- coding: utf-8 -*-
"""
步骤③：把突触电导映射为神经网络权重 —— 差分对突触层（PyTorch）
================================================================

一个权重 w（归一化到 [-1, 1]）由两个突触器件的电导差实现::

    G_p = G_min + (w + 1)/2 · (G_max − G_min)     ← 正器件（ POT 时增强）
    G_d = G_min + (1 − w)/2 · (G_max − G_min)     ← 负器件（ POT 时抑制）
    w   = (G_p − G_d) / (G_max − G_min)

两个器件始终互补地更新（一个增强、一个抑制），使总电导 G_p+G_d 保持不变。

训练方案（"主机算梯度、器件落权重"的片上训练仿真）：
  1. 前向传播用当前有效权重 W = (G_p − G_d)/(G_max − G_min)，正常自动求导；
  2. 反向传播得到 dL/dW（这发生在计算机里，不消耗器件寿命）;
  3. device_update() 把梯度换算成"要打的脉冲数"，并严格按照器件的指数
     更新模型 G ← G ± α·exp[−β(G−G_min)/(G_max−G_min)] 去修改 G_p / G_d。
     器件的非线性（β）和不对称（β_p ≠ β_d）因此真实地影响学习过程。

可选非理想因素：
  read_noise   : 读取噪声（每次前向给电导加高斯扰动，模拟读出波动）
  update_noise : 更新噪声（每个脉冲步长乘 (1+σ·N(0,1))，模拟脉冲间涨落）
  discrete     : 脉冲数取整 + 随机舍入（更接近真实只能打整数脉冲的器件）
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse import calibrate_alpha


@dataclass
class DeviceParams:
    """突触器件参数（与 synapse.py 中的模型一致）"""
    g_min: float = 0.1          # 最小电导 (µS)
    g_max: float = 9.9          # 最大电导 (µS)
    beta_p: float = 2.0         # 增强非线性因子 (>0)
    beta_d: float = -4.0        # 抑制非线性因子 (<0)
    n_pulses: float = 100.0     # 标定 α 用的满窗口脉冲数
    read_noise: float = 0.0     # 读取噪声 σ（相对电导窗口）
    update_noise: float = 0.0   # 更新噪声 σ（相对单脉冲步长）
    discrete: bool = False      # True → 整数脉冲 + 随机舍入

    def __post_init__(self):
        R = self.g_max - self.g_min
        # 标定两个方向的步长 α，使 n_pulses 个脉冲恰好扫满窗口
        self.alpha_p = calibrate_alpha(self.beta_p, self.g_min, self.g_max,
                                       int(self.n_pulses), potentiate=True)
        self.alpha_d = calibrate_alpha(self.beta_d, self.g_min, self.g_max,
                                       int(self.n_pulses), potentiate=False)


class SynapseLinear(nn.Module):
    """由一对电导矩阵 (G_p, G_d) 构成的全连接层（无偏置，交叉阵列常规做法）。

    内部保留一个浮点参数 W 仅供自动求导；每次 device_update() 后
    W 会被重新同步为器件当前的真实电导差。
    """

    def __init__(self, in_features, out_features, params: DeviceParams):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = params
        self.R = params.g_max - params.g_min

        # 常规初始化（Kaiming uniform），再换算成一对互补电导
        w0 = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w0, a=5 ** 0.5)
        w0.clamp_(-0.5, 0.5)                       # 初始权重限制在窗口中部

        G_mid = 0.5 * (params.g_min + params.g_max)
        G_p = G_mid + 0.5 * w0 * self.R
        G_d = G_mid - 0.5 * w0 * self.R
        self.G_p = G_p                             # 普通张量，手动更新
        self.G_d = G_d
        self.W = nn.Parameter((G_p - G_d) / self.R)  # 仅供 autograd 使用

    # ------------------------------------------------------------------
    def forward(self, x):
        W = self.W
        if self.p.read_noise > 0 and self.training:
            # 读出噪声：每次前向给有效权重加高斯扰动（只影响数值，不参与求导）
            W = W + self.p.read_noise * torch.randn_like(W)
        return F.linear(x, W)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def device_update(self, lr):
        """把浮点梯度翻译成器件脉冲：这是"片上学习"的软件仿真核心。

        注意：外设控制器只知道"打一对脉冲，权重大约变 2/N"（线性假设），
        并不知道器件当前电导处的真实步长。真实步长由指数模型决定，
        与标称值的偏差正是非线性 NL / 不对称 AR 影响精度的来源。
        """
        if self.W.grad is None:
            return
        dw = -lr * self.W.grad                     # 期望的有效权重变化
        p = self.p

        # 线性假设下的脉冲数：一对脉冲平均改变 w ≈ 2/N
        n_f = dw.abs() * (p.n_pulses / 2.0)

        if p.discrete:
            # 只能打整数脉冲：向下取整 + 按小数部分概率随机舍入（无偏）
            n = torch.floor(n_f) + (torch.rand_like(n_f) < n_f.frac()).float()
        else:
            n = n_f                                # 连续近似：允许分数脉冲

        # 器件实际执行：步长严格按指数模型、由当前电导决定（非线性 + 不对称）
        if p.update_noise > 0:
            noise = 1 + p.update_noise * torch.randn_like(n)
        else:
            noise = 1.0
        up = dw > 0                                # True: 需要 G_p↑ G_d↓
        step_pot = p.alpha_p * torch.exp(-p.beta_p * (self.G_p - p.g_min) / self.R)
        step_dep = p.alpha_d * torch.exp(-p.beta_d * (self.G_p - p.g_min) / self.R)
        step_pot_d = p.alpha_p * torch.exp(-p.beta_p * (self.G_d - p.g_min) / self.R)
        step_dep_d = p.alpha_d * torch.exp(-p.beta_d * (self.G_d - p.g_min) / self.R)

        G_p_new = torch.where(up, self.G_p + n * step_pot * noise,
                              self.G_p - n * step_dep * noise)
        G_d_new = torch.where(up, self.G_d - n * step_dep_d * noise,
                              self.G_d + n * step_pot_d * noise)
        self.G_p = G_p_new.clamp_(p.g_min, p.g_max)
        self.G_d = G_d_new.clamp_(p.g_min, p.g_max)

        # 用器件的真实电导差刷新前向权重
        self.W.data = (self.G_p - self.G_d) / self.R
        self.W.grad = None


class SynapseMLP(nn.Module):
    """784–100–10 的两层感知机，权重全部由差分对突触器件实现。"""

    def __init__(self, params: DeviceParams, hidden: int = 100,
                 in_features: int = 784, out_features: int = 10):
        super().__init__()
        self.fc1 = SynapseLinear(in_features, hidden, params)
        self.fc2 = SynapseLinear(hidden, out_features, params)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))

    @torch.no_grad()
    def device_update(self, lr):
        self.fc1.device_update(lr)
        self.fc2.device_update(lr)
