# Walker2d PPO 单 Notebook 学习项目

这是从[倒立摆 PPO 学习仓库](https://github.com/vtonto/ppo-inverted-pendulum)迁移而来的 Walker2d 学习项目。核心原则是：**所有可执行代码集中在一个 Notebook 中**，不需要在 `.ipynb` 和多个 `.py` 文件之间来回跳转。

主入口：[walker2d_ppo_learning.ipynb](walker2d_ppo_learning.ipynb)

Notebook 内直接包含：

- `Walker2d-v5` 的浮动基、9 个广义自由度、6 个执行器、质量/摩擦/时钟参数；
- 17 维观测、6 维归一化执行器输入、gear 与实际广义驱动力映射；
- 随机策略 GIF 与摔倒诊断；
- `PPOConfig` 与 `smoke / short / full` 三档配置；
- 独立 Gaussian Actor 与 Critic；
- 六维联合 `log_prob.sum(-1)`；
- raw action → `clip` → environment 的一致动作语义；
- Rollout Buffer 与跨 rollout 的 Episode 状态；
- 区分 `terminated / truncated` 的 GAE 双 mask；
- PPO-Clip、mini-batch、多 epoch、梯度裁剪；
- 完整训练仪表盘、周期 checkpoint、固定 seed 独立评估和 GIF；
- 躯干、关节、动作、广义驱动力和脚接触的 Policy Behavior Analysis；
- 环境、张量、ratio、GAE、Clip、梯度、checkpoint 等断言测试；
- 奖励权重对照实验区和鲁棒性扩展说明。

> `smoke` 只验证 rollout → GAE → PPO update → checkpoint → evaluation，不代表策略已经学会行走。

## Apple Silicon 安装

推荐 Python 3.10–3.12，默认使用 CPU。MuJoCo 主要在 CPU 上做物理仿真，小型 MLP 放到 MPS 不一定更快。

```bash
git clone https://github.com/vtonto/walker2d-ppo-learning.git
cd walker2d-ppo-learning

# 推荐：uv
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 注册 Notebook 内核
python -m ipykernel install --sys-prefix \
  --name walker2d-ppo --display-name "Python (Walker2d PPO)"

python -m jupyterlab
```

也可以使用标准 `venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyterlab
```

打开 `walker2d_ppo_learning.ipynb`，选择 **Python (Walker2d PPO)**，从上到下执行即可。

## 推荐学习顺序

1. 只运行 00～02：理解浮动基、欠驱动、模型参数、奖励和随机策略跌倒。
2. 运行 03～08：逐单元调试 Actor/Critic、GAE、Rollout 和 PPO 更新。
3. 运行 09～11：执行 `smoke`、查看日志并保存评估 GIF。
4. 将 `RUN_PROFILE` 改为 `short`，观察真实训练趋势。
5. 在 12 中一次只改一个奖励权重，完成控制变量实验。

## 环境速查

默认观测 `shape=(17,)`、环境 dtype 为 `float64`，进入网络前转为 `float32`：

| 索引 | 含义 |
|---:|---|
| 0–1 | 躯干高度、躯干角度 |
| 2–7 | 右髋/膝/踝与左髋/膝/踝关节角 |
| 8–10 | 躯干 x/z 速度与角速度 |
| 11–16 | 六个关节角速度 |

动作 `shape=(6,)`，顺序为右髋/膝/踝、左髋/膝/踝的归一化执行器控制输入，范围 `[-1, 1]`。它不是直接以 N·m 表示的关节力矩；Notebook 会读取 actuator gear、`data.actuator_force` 和 `data.qfrc_actuator` 检查映射。默认不把绝对 x 位置交给策略，但 `info["x_position"]` 可用于独立评估。

默认奖励为：

```text
reward = reward_forward + reward_survive + reward_ctrl
```

`info["reward_ctrl"]` 已经带负号，不能再次减去。默认环境步长为 `frame_skip(4) × 0.002 = 0.008 s`。

官方事实以 [Gymnasium Walker2d-v5 文档](https://gymnasium.farama.org/main/environments/mujoco/walker2d/)和[环境源码](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/walker2d_v5.py)为准。

## 动作概率的一致性

Notebook 首版采用容易检查的教学方案：

```text
Actor sample / Buffer store / PPO update: raw action u
Environment execution:                    clip(u, -1, 1)
```

PPO ratio 始终评价同一个 raw action，因此内部一致；但它不是裁剪后有界动作的精确概率密度。Notebook 记录 `action_saturation`，方便判断边界饱和是否严重。后续可独立学习 `tanh`-squashed Gaussian 与 Jacobian 修正。

## 输出目录

运行 Notebook 后会生成：

```text
runs/smoke-YYYYMMDD-HHMMSS/
├── config.json
├── metrics.csv
├── evaluation.json
├── checkpoint_0000064.pt
├── checkpoint_0000128.pt
└── final.pt

outputs/
├── random_policy.gif
├── smoke-evaluation.gif
├── smoke-training-dashboard.png
├── smoke-behavior.gif
└── smoke-behavior-analysis.png
```

这些运行产物默认不提交到 Git，避免仓库膨胀。

## 项目文件

```text
walker2d-ppo-learning/
├── walker2d_ppo_learning.ipynb  # 全部可执行代码与测试
├── README.md                    # 安装和学习入口
├── requirements.txt            # Python 依赖
├── FORMULAS_AND_NOTES.md        # 公式与关键语义
├── EXPERIMENT_GUIDE.md          # 对照实验模板
├── ACCEPTANCE_REPORT.md         # 实际验证记录
├── outputs/
└── checkpoints/
```

## 已知边界

- 首版单环境、CPU 优先，没有并行采样或观测归一化。
- 周期 checkpoint 保存模型、优化器、配置和随机数状态，但不保存 MuJoCo 当前 Episode/collector 状态，因此不支持位级一致的中途恢复。
- `control_effort = Σ||a_env||²` 是归一化控制输入平方和，不是机械能耗。
- 强化学习结果有随机性；正式结论应使用多个训练 seed 和相同的固定评估 seed。
- 物理参数扰动属于纯仿真泛化测试，不等于真正 Sim-to-Real。

