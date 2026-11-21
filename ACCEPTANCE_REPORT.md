# 验收报告

验收日期：2026-11-21（Asia/Shanghai）

## 本轮评审建议改进

项目继续保持“单 Notebook 即完整工程”，并完成以下增量修改：

1. 第 01 节增加浮动基与欠驱动解释，实际读取 `nq/nv/nu`、body mass、joint type/range/address、actuator transmission/gear/ctrlrange、geom friction、仿真时钟与真实 `qpos/qvel`。
2. 增加固定单关节输入实验，核对 `data.ctrl → data.actuator_force → data.qfrc_actuator`。
3. 第 10 节增加可复用的 `plot_training_metrics()`，覆盖 Episode Return/Length、奖励分项、Actor/Critic Loss、KL、Clip Fraction、Std/Entropy、动作饱和、结束原因、吞吐等真实 CSV 指标并保存 PNG。
4. 第 11 节增加 Policy Behavior Analysis，以真实仿真时间绘制躯干、位移/速度、关节角/速度、raw/执行动作、广义驱动力与实际脚地接触，并保存对应 GIF。
5. 将误导性的 `control_energy` 改为累计/每步 `control_effort`，明确它不是物理能耗；评估同时报告平均速度与动作饱和。
6. 增加周期 checkpoint；不同运行使用带时间戳的独立目录，避免覆盖。
7. 第 12 节增加共同默认环境、固定 seeds 的 checkpoint 统一评估和对照绘图函数，等待真实基线后使用。

## 实际验证环境

- Apple Silicon `arm64`
- macOS 27.2.1
- Python 3.12.15
- Gymnasium 1.4.0
- MuJoCo 4.0.1
- PyTorch 2.15.0

## 实际执行结果

- Notebook 共 42 个单元，其中 22 个代码单元。
- 整本 Notebook 从头顺序执行成功，无错误输出。
- 共 32 条显式断言通过，覆盖原 PPO/GAE/rollout 检查，以及模型维度、执行器映射、周期 checkpoint 加载等新增检查。
- 真实模型读取确认 `nq=9, nv=9, nu=6`；固定第 0 个执行器输入验证了 gear 与对应广义驱动力映射。
- smoke 实际运行 128 环境步、2 次 rollout/update，成功生成：
  - `checkpoint_0000064.pt`
  - `checkpoint_0000128.pt`
  - `final.pt`
  - `metrics.csv`、`evaluation.json`
- 两个周期 checkpoint 均成功加载；第一个周期 checkpoint 完成固定 seed 独立评估。
- 训练仪表盘 PNG、评估 GIF、行为分析 PNG/GIF 均实际生成并完成视觉检查。

## smoke 的真实结果

本次 final checkpoint 的一次自动独立评估持续 16 步，Return 约 4.18，前进距离约 -0.087 m，平均速度约 -0.676 m/s，每步平均 `control_effort` 约 0.379。固定 seed 行为图如实显示策略向后移动并很快跌倒。

这些结果只证明模型分析、日志、绘图、周期保存、加载和评估链路正确；128 步 smoke 不代表学会行走。

## 尚未验证

- 未自动运行 20,000 步 `short` 或 200,000 步 `full`，因此没有稳定行走结论。
- 未执行三组奖励权重训练；共同环境评估函数已准备好，但不能用伪造数据填充结果。
- 鲁棒性、Domain Randomization 和 RL + PD 仍是后续项目。
- checkpoint 不包含完整 MuJoCo/collector 当前状态，不能做位级一致的中途恢复。

## 复验方式

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute walker2d_ppo_learning.ipynb \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=240
```

