# 对照实验指南

先完成 `pytest` 和 `smoke`，再建立一个未改奖励的基线。每次只改变一个变量；不同奖励定义下的训练 Return 不能直接作为跨实验优劣结论。

## 通用记录模板

| 字段 | 填写内容 |
|---|---|
| 问题 | 我想解释什么行为？ |
| 假设 | 改动前预计发生什么，为什么？ |
| 唯一改动 | 参数名、基线值、实验值 |
| 固定项 | 代码提交、训练步数、网络、优化器、训练 seed |
| 评估 | 固定 seed 列表、Episode 数、共同外部指标 |
| 结果 | 曲线、表格、训练前后视频 |
| 解释 | 数据支持/反驳了什么？还有哪些混杂因素？ |
| 下一步 | 最小的后续验证是什么？ |

共同外部指标建议：Episode Length、实际前进距离、每回合控制能量、动作饱和比例。它们不依赖训练 Return 的奖励权重，可在同一评估环境下比较。

## C1：控制代价权重

假设示例：“更大的控制惩罚可能降低动作幅度和能量，但也可能使策略难以恢复平衡。”

```bash
# 基线
walker2d-train --profile short --seed 1 \
  --ctrl-cost-weight 0.001 --output-dir runs/c1/base-seed1

# 较小控制代价
walker2d-train --profile short --seed 1 \
  --ctrl-cost-weight 0.0001 --output-dir runs/c1/low-seed1

# 较大控制代价
walker2d-train --profile short --seed 1 \
  --ctrl-cost-weight 0.01 --output-dir runs/c1/high-seed1
```

观察：`action_saturation`、控制能量、前进距离、Length、拖脚/抖动/摔倒视频。不要只比较三种定义不同的训练 Return。

## C2：前进奖励权重

```bash
walker2d-train --profile short --seed 1 \
  --forward-reward-weight 0.5 --output-dir runs/c2/forward-05-seed1
walker2d-train --profile short --seed 1 \
  --forward-reward-weight 2.0 --output-dir runs/c2/forward-20-seed1
```

统一用默认奖励配置或纯物理指标复评 checkpoint，避免把更大的 reward scale 错解为更好的策略。

## C3：存活奖励（可选）

比较 `healthy_reward` 时，专门观察“站着不动”是否成为局部解。它是待检验假设，不是预设结论。

## 多 seed 原则

单个 seed 适合学习和排错，不适合强结论。算力允许时至少用 3 个训练 seed，每个 checkpoint 用同一组固定评估 seed，报告均值和标准差，不只展示最好的一次。

## D：纯仿真鲁棒性扩展

在训练完成后复制 MuJoCo 模型参数并做测试时扰动，例如地面摩擦或连杆质量的 0.9×/1.0×/1.1×。先保持策略固定，再比较标称/扰动条件的前进距离、Length 和能量。

这只能说明仿真内泛化。真实机器人还包含传感器噪声、执行器带宽、延迟、未建模柔性与接触误差，不能据此声称完成 Sim-to-Real。

