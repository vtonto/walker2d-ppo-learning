# 验收报告

验收日期：2026-11-21（Asia/Shanghai）

## 已完成

- 根据需求建立 `Walker2d-v5` 教学项目，而不是仅提供规划；
- 从倒立摆项目迁移独立 Actor/Critic、GAE、PPO-Clip、mini-batch 更新、checkpoint 与评估结构；
- 针对 6 维动作实现联合 `log_prob.sum(-1)`；
- 实现 raw Gaussian action → environment clip 的一致 Buffer/ratio 语义；
- 分开处理 `terminated` 的 bootstrap mask 与 `terminated or truncated` 的 GAE continuation mask；
- rollout 边界保留当前 Episode 的状态、Return 和 Length；
- 记录奖励分项、动作饱和、Actor/Critic loss、KL、Clip Fraction、Entropy、吞吐和结束原因；
- 提供 `smoke`、`short`、`full` 三档运行入口、独立确定性评估与 GIF；
- 提供奖励权重对照实验模板和纯仿真鲁棒性扩展说明。

## 实际验证环境

- Apple Silicon `arm64`
- macOS 27.2.1
- Python 3.12.15
- Gymnasium 1.4.0
- MuJoCo 4.0.1
- PyTorch 2.15.0

依赖范围写在 `pyproject.toml`；上述是本次实际解析并运行的版本，不代表唯一兼容组合。

## 实际执行结果

1. `pytest`：11 项全部通过。
2. 真实 `Walker2d-v5`：成功完成 `reset` / `step`，确认观测 `(17,) float64`、动作 `(6,) float32`、奖励分项及 `dt = timestep × frame_skip`。
3. `python scripts/smoke_train.py`：成功完成 128 环境步、一次 rollout、GAE、PPO update、checkpoint 保存和一次独立评估。
4. `walker2d_ppo_learning.ipynb`：24 个单元从头顺序执行完成，无错误。
5. 渲染：成功生成随机策略与 smoke 策略 GIF。
6. checkpoint 往返：同一观测的确定性动作在保存/加载前后一致。

## smoke 的真实结果与解释

本次 smoke 的训练日志显示 128 个环境步、1 次 PPO update；独立确定性评估只持续 17 步，Return 约为 3.51，前进距离约为 -0.10 m。它说明接线、数值和文件输出正常，也清楚说明 128 步策略尚未学会行走。

没有运行足够预算的 `short` / `full` 训练，因此不声称已经收敛，也没有提交“成功步态”权重。下一步应先运行 `short` 查看吞吐、动作饱和、KL、回报和视频，再决定训练预算；如失败，按 README 中的边界语义和诊断指标逐项排查。

## 可复现命令

```bash
source .venv/bin/activate
pytest
python scripts/smoke_train.py
jupyter nbconvert --to notebook --execute walker2d_ppo_learning.ipynb \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=180
```

