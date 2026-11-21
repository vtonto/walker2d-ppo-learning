# Walker2d PPO 学习项目

这是从[倒立摆 PPO 学习仓库](https://github.com/vtonto/ppo-inverted-pendulum)迁移而来的第二阶段项目：不用 Stable-Baselines3 代替核心算法，直接阅读和修改 Actor、Critic、GAE、PPO-Clip、rollout、评估与日志代码，在 `Gymnasium Walker2d-v5` 上学习多关节接触控制。

> 首要目标是实现正确、可检查、可复现的训练闭环。`smoke` 只验证一次 rollout → GAE → PPO update → checkpoint → evaluation；它不代表策略已经学会行走。

## 你会学到什么

- 17 维观测如何对应躯干、六个关节的位置与速度；
- 6 维动作为什么是关节力矩，而不是“目标角度 + PD”；
- `reward_forward + reward_survive + reward_ctrl` 的物理含义（`info["reward_ctrl"]` 已经是负数）；
- 为什么 `terminated` 禁止 TD bootstrap，而 `truncated` 仍需从真实末状态 bootstrap；
- 如何在 rollout 边界保持同一 Episode，并在 reset 边界切断 GAE；
- 如何记录动作饱和、KL、Clip Fraction、回报分项和独立评估指标。

官方环境事实以 [Gymnasium Walker2d-v5 文档](https://gymnasium.farama.org/main/environments/mujoco/walker2d/)和[环境源码](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/walker2d_v5.py)为准。

## Apple Silicon 安装

推荐 Python 3.10–3.12，默认 `device="cpu"`。MuJoCo 仿真在 CPU 上运行，小网络放到 MPS 通常未必更快。

```bash
git clone https://github.com/vtonto/walker2d-ppo-learning.git
cd walker2d-ppo-learning

# 推荐：uv
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev,notebook]"

# 注册 Notebook 内核
python -m ipykernel install --sys-prefix \
  --name walker2d-ppo --display-name "Python (Walker2d PPO)"
```

也可以使用标准 `venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
```

## 三步验收

```bash
# 1. 数学、张量、边界语义、环境和 checkpoint 测试
pytest

# 2. 128 个环境步的一次端到端短跑（不会收敛）
python scripts/smoke_train.py

# 3. 教学入口
python -m jupyterlab
# 打开 walker2d_ppo_learning.ipynb
```

短跑成功后会生成：

```text
runs/smoke/
├── config.json
├── metrics.csv
├── evaluation.json
└── final.pt
```

## 训练档位

```bash
# 只验收接线
walker2d-train --profile smoke

# 初步观察趋势，不承诺学会稳定行走
walker2d-train --profile short --output-dir runs/short-seed1

# 较大预算；请先根据本机吞吐决定是否运行
walker2d-train --profile full --total-timesteps 200000 \
  --output-dir runs/full-seed1
```

确定性评估和 GIF：

```bash
walker2d-evaluate runs/short-seed1/final.pt \
  --episodes 5 --seed 10000 --video outputs/short-seed1.gif
```

评估不会更新参数。输出包含 Return、Length、前进距离和控制能量的均值/标准差。视频编码不可用时可省略 `--video`，训练本身不依赖渲染。

## 环境速查

默认观测 `shape=(17,)`，环境给出 `float64`，送入网络前显式转为 `float32`：

| 索引 | 含义 |
|---:|---|
| 0–1 | 躯干高度、躯干角度 |
| 2–7 | 右侧髋/膝/踝与左侧髋/膝/踝关节角 |
| 8–10 | 躯干 x/z 速度与角速度 |
| 11–16 | 六个关节角速度 |

默认不包含绝对 x 位置，但 `info["x_position"]` 可用于评估前进距离。动作 `shape=(6,)`，顺序为右侧髋/膝/踝、左侧髋/膝/踝的力矩控制，范围 `[-1, 1]`。

默认环境步长 `dt = frame_skip(4) × 物理积分步长(0.002) = 0.008 s`。奖励分解为：

```text
reward = reward_forward + reward_survive + reward_ctrl
```

其中 `reward_ctrl = -ctrl_cost_weight * ||action||²`，记录时不要再次减去。

## 实现选择：raw Gaussian + clip

Actor 建模未限制高斯动作 `u`；环境执行 `clip(u, -1, 1)`。Buffer 保存同一个 `u` 和采样时的联合 `log_prob`，PPO 更新也对这个 `u` 重新计算新策略概率：

```text
sample/store/update: raw action u
environment:          clip(u, -1, 1)
```

这保证 PPO ratio 内部一致，但没有把执行动作的有界分布冒充成原始高斯的精确密度。代价是边界处存在动作饱和与探索效率问题；日志中的 `action_saturation` 专门帮助检查它。`tanh`-squashed Gaussian 和 Jacobian 修正适合作为后续独立实验。

## 目录

```text
walker2d-ppo-learning/
├── walker2d_ppo_learning.ipynb
├── src/walker2d_ppo/       # 模型、GAE、rollout、更新、训练、评估
├── tests/                  # 公式、边界语义、环境、checkpoint 测试
├── scripts/smoke_train.py
├── FORMULAS_AND_NOTES.md
├── EXPERIMENT_GUIDE.md
├── ACCEPTANCE_REPORT.md
├── outputs/                # 可分享图表与 GIF（默认内容不提交）
└── checkpoints/            # 手动保留的权重（默认内容不提交）
```

## 从倒立摆迁移了什么

复用：独立 Actor/Critic、联合高斯 log probability、GAE、PPO-Clip、mini-batch 多 epoch、checkpoint 和诊断思路。

重点改动：动作维从 1 到 6；环境边界变成 `[-1,1]`；Episode 更常因摔倒终止；日志增加奖励分项、距离、动作饱和；rollout collector 明确持有跨 rollout 的 Episode 状态。

## 已知边界

- 首版单环境、CPU 优先，没有并行采样、观测归一化或学习率退火。
- checkpoint 保存模型、优化器、配置和随机数状态，但不保存 MuJoCo 当前 Episode 的完整状态，因此不支持位级一致的中途恢复。
- 训练结果有随机性。应使用固定评估种子，并在算力允许时使用多个训练 seed。
- 物理参数扰动属于纯仿真泛化测试，不等于真正 Sim-to-Real。

