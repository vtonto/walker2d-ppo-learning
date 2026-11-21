# 验收报告

验收日期：2026-11-21（Asia/Shanghai）

## 本次结构调整

根据学习和调试需求，项目已改成“单 Notebook 即完整工程”：

- Actor、Critic、配置、GAE、PPO-Clip、Rollout、训练、评估、checkpoint 和绘图全部位于 `walker2d_ppo_learning.ipynb`；
- 删除版本库中的 `src/`、`scripts/`、`tests/` 和 Python 打包配置；
- 原独立测试改为 Notebook 相邻章节中的可运行断言，修改算法后可立即重跑对应单元；
- `requirements.txt` 只负责创建环境，不安装本地 Python 包。

## 实际验证环境

- Apple Silicon `arm64`
- macOS 27.2.1
- Python 3.12.15
- Gymnasium 1.4.0
- MuJoCo 4.0.1
- PyTorch 2.15.0

依赖范围写在 `requirements.txt`；上述是本次实际运行版本，不代表唯一兼容组合。

## 实际执行结果

1. Notebook 共 36 个单元，其中 19 个代码单元。
2. 整本 Notebook 从头顺序执行成功，没有错误输出。
3. Notebook 内包含 25 条显式 `assert`，覆盖环境维度/dtype、奖励分解、联合 log probability、更新前 ratio、GAE 的 terminated/truncated/rollout 边界、PPO Clip、跨 rollout Episode、有限数值、梯度和 checkpoint 往返。
4. 真实 `Walker2d-v5` 成功完成环境探索、128 步 rollout、一次 PPO update、checkpoint 保存与独立评估。
5. 随机策略和 smoke 策略 GIF 均成功生成。

## smoke 的真实结果

本次 smoke 独立确定性评估只持续 17 步，Return 约 3.51，前进距离约 -0.10 m。它证明 Notebook 的训练闭环可运行，也说明 128 步策略尚未学会行走。

没有运行足够预算的 `short` / `full` 训练，因此不声称已经收敛。下一步应先把 Notebook 第 10 节的 `RUN_PROFILE` 改成 `short`，再根据吞吐、动作饱和、KL、奖励分项和视频判断是否增加预算。

## 复验方式

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute walker2d_ppo_learning.ipynb \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=180
```

