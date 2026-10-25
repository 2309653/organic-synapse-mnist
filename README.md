# 模拟有机光电突触的 MNIST 分类（神经形态计算入门项目）

用软件仿真"有机光电突触器件"的电导更新行为，把器件的非线性、不对称等真实特性
带进神经网络训练，在 MNIST 手写数字分类（784-100-10 MLP）上量化这些特性对精度
的影响。适合作为神经形态计算 / 存内计算 / 突触器件方向的入门项目。

## 实测结果（本项目默认参数，CPU 即可复现）

| 配置 | 测试精度 |
|---|---|
| 理想浮点 SGD（software 模式，3 epoch，全量 6 万张） | **96.37%** |
| 突触器件仿真（device 模式，3 epoch，全量 6 万张） | **96.26%** |
| 器件 + 读噪声 0.02 + 更新噪声 0.1 + 整数脉冲（2 epoch，2 万张） | 87.54% |

非线性因子 β 扫描（1 epoch，2 万张子集）：β=0 时 86.14%（NL=0%），
β=8 时降到 79.78%（NL≈95%）——**器件非线性越强，网络学得越差**。

## 项目结构（对应实验流程四步）

```
organic-synapse-mnist/
├── src/
│   ├── synapse.py   # ①电导更新曲线 G(n)  ②提取非线性度 NL、不对称比 AR
│   ├── layers.py    # ③归一化电导 → 差分对 → 网络权重（PyTorch 自定义层）
│   ├── train.py     # ④MNIST 训练主脚本（software 基线 / device 仿真）
│   └── sweep.py     # 附加实验：β（非线性）对精度的影响
├── figures/         # 运行后自动生成的图（已提交，可直接看）
├── results/         # 运行后自动生成的 CSV 数据（已提交）
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖（建议 Python ≥ 3.9）
pip install -r requirements.txt

# 2. 看器件长什么样：打印 NL/AR 指标，生成 figures/fig1、fig2
python src/synapse.py

# 3a. 理想基线训练（约 35 秒/3 epoch，CPU）
python src/train.py --mode software --epochs 3

# 3b. 器件仿真训练（同样速度）
python src/train.py --mode device --epochs 3

# 4. 附加实验：非线性对精度的影响（约 30 秒）
python src/sweep.py --subset 20000 --test-subset 5000

# 急着看结果？用子集快速跑：
python src/train.py --mode device --epochs 1 --subset 10000 --test-subset 2000
```

首次运行会自动下载 MNIST（约 11 MB）到 `./data`（已 gitignore，不入库）。

## 原理：四步流程

### ① 电导更新曲线 G(n)

突触器件在连续相同极性电脉冲下的电导演化用指数模型描述：

```
G(n+1) = G(n) ± α · exp[ −β · (G(n) − G_min) / (G_max − G_min) ]
```

- **增强（potentiation）**：β > 0，电导低时步长大、高时步长小（先快后慢，饱和）；
- **抑制（depression）**：β < 0，电导高时步长大（同样先快后慢）；
- α 为步长幅度，脚本里用二分法自动标定，使 N=100 个脉冲恰好扫满电导窗口。

### ② 提取 NL 与 AR

- **非线性度 NL** = maxₙ|G(n) − G_lin(n)| / (G_max − G_min)，
  其中 G_lin(n) 是从 G(0) 到 G(N) 的理想直线。NL=0 表示完全线性。
- **不对称比 AR**：把抑制曲线镜像到增强方向后比较，
  AR = maxₙ|G_p(n) − (G_max+G_min−G_d(n))| / (G_max − G_min)。
  完美对称的器件 AR=0。
  > 注：如果直接按 max|G_p(n)−G_d(n)| / (G_p(N)−G_d(N)) 计算，由于两条曲线
  > 分别从窗口两端出发，端点处电导差恒等于窗口值，结果恒为 1，没有区分度，
  > 所以本项目采用文献通行的镜像定义（`synapse.py` 里两种都实现了）。

运行 `python src/synapse.py` 的默认输出：

```
NL_p = 23.97 %   NL_d = 42.66 %   AR = 0.230
```

### ③ 归一化电导 → 网络权重（差分对）

一个权重 w（归一化到 [−1, 1]）由两个器件的**电导差**实现，正负权重都cover：

```
G_p = G_min + (w+1)/2 · (G_max − G_min)   ← 正器件
G_d = G_min + (1−w)/2 · (G_max − G_min)   ← 负器件
w   = (G_p − G_d) / (G_max − G_min)
```

更新时两个器件互补地变化（一个增强、一个抑制），总电导 G_p+G_d 保持不变。

### ④ 片上训练仿真（PyTorch）

采用"**主机算梯度、器件落权重**"的标准方案（`src/layers.py`）：

1. 前向传播用当前有效权重 w = (G_p − G_d)/(G_max − G_min)，正常自动求导；
2. 反向传播得到梯度（发生在计算机里，不消耗器件）；
3. `device_update()` 把梯度换算成**脉冲数**——外设只按线性假设
   （一对脉冲 ≈ 改变 2/N）发脉冲，器件则严格按指数模型执行。
   真实步长与标称步长的偏差，正是 NL/AR 影响精度的来源；
4. 可选非理想因素：`--read-noise`（读噪声）、`--update-noise`（脉冲涨落）、
   `--discrete`（只能打整数脉冲，用随机舍入保持无偏）。

## 常用参数（train.py）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | device | `software`=理想浮点 SGD；`device`=器件仿真 |
| `--epochs` | 3 | 训练轮数 |
| `--lr` | 0.1 | software 是普通 SGD 步长；device 是每批的期望权重变化尺度 |
| `--beta-p` / `--beta-d` | 2.0 / 4.0 | 增强/抑制非线性因子 \|β\| |
| `--gmin` / `--gmax` | 0.1 / 9.9 µS | 电导窗口 |
| `--n-pulses` | 100 | 扫满窗口所需脉冲数（决定权重分辨率） |
| `--read-noise` | 0 | 读取噪声 σ（相对窗口） |
| `--update-noise` | 0 | 更新噪声 σ（相对单脉冲步长） |
| `--discrete` | 关 | 开启后只能打整数脉冲（随机舍入） |
| `--subset` / `--test-subset` | 0（全量） | 用子集快速实验 |

## 可以做的延伸实验

1. **不对称的影响**：固定 β_p=2，改 β_d ∈ {1, 2, 4, 8}，看精度怎么掉；
2. **分辨率的影响**：`--n-pulses` 从 100 降到 10、5，看权重"档位"变粗的后果；
3. **开启整数脉冲** `--discrete`：对比连续近似与真实整数脉冲的差距；
4. **加噪声**：`--read-noise 0.02 --update-noise 0.1`，量化噪声鲁棒性；
5. **换你自己的器件**：把实测的 G(n) 曲线代替指数模型（改 `synapse.py` 的
   `pulse_sweep`，或直接把实测 α、β 填进 `DeviceParams`）。

## 推送到 GitHub（新手教程）

### 方法 A：用 GitHub CLI（推荐，最省事）

```bash
# 1. 安装 GitHub CLI（Windows）：winget install GitHub.cli 或去 https://cli.github.com 下载
# 2. 登录（会给你一个 8 位代码，浏览器里输入即可）
gh auth login

# 3. 在项目目录里一键建仓库并推送（public 可改 private）
cd organic-synapse-mnist
gh repo create organic-synapse-mnist --public --source . --push
```

### 方法 B：网页建仓库 + git 命令

1. 打开 https://github.com/new ，仓库名填 `organic-synapse-mnist`，
   选 Public/Private，**不要**勾选任何初始化选项（README/.gitignore 都不要），
   点 Create repository；
2. 在项目目录下执行（把 `<你的用户名>` 换掉）：

```bash
git init
git add .
git commit -m "有机光电突触 MNIST 分类仿真：器件建模 + 差分对映射 + PyTorch 训练"
git branch -M main
git remote add origin https://github.com/<你的用户名>/organic-synapse-mnist.git
git push -u origin main
```

3. push 时弹出的登录窗口用浏览器授权即可（Windows 的 Git Credential Manager
   会自动保存，之后不用再登录）。

> 注：本项目代码已完成 `git init` 和首次 commit 的话，从上面第 2 步的
> `git remote add` 开始执行即可。

## 常见问题

- **MNIST 下载失败/很慢**：手动从 https://ossci-datasets.s3.amazonaws.com/mnist/
  下载 4 个 .gz 文件放进 `data/MNIST/raw/`；
- **CPU 训练慢**：加 `--subset 20000`；本默认网络在普通 CPU 上全量一轮约 12 秒；
- **Windows 报 OpenMP 错误**：脚本开头已设置 `KMP_DUPLICATE_LIB_OK=TRUE` 规避；
- **精度为什么比论文低**：论文常用 CNN / 更长训练 / 更大隐藏层，本项目刻意用
  最小的 784-100-10 MLP 突出器件效应本身。

## 参考脉络

指数电导更新模型与 NL/AR 指标的定义见忆阻器/突触阵列仿真文献（如 NeuroSim、
CrossSim 等评估框架的器件建模部分）；差分对权重映射与"主机算梯度、器件落权重"
是片上学习仿真的通行做法。
