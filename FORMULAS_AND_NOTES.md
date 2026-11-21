# PPO 公式与代码映射

这份笔记只解释 Walker2d 迁移中最容易出错的对应关系。所有实现都直接位于 `walker2d_ppo_learning.ipynb`。

## 1. 多维高斯动作

Actor 对 6 个动作坐标输出因子化高斯。Notebook 中的 `GaussianActor.joint_log_prob` 只沿最后一个动作维求和：

\[
\log \pi_\theta(\mathbf u_t|s_t)
=\sum_{i=1}^{6}\log \pi_{\theta,i}(u_{t,i}|s_t).
\]

因此 `[batch, 6] -> [batch]`，绝不能把 batch 维一起求和。

首版策略变量是 raw action `u`：

1. Actor 采样 `u`；
2. Buffer 保存 `u` 和 `log π_old(u|s)`；
3. 环境执行 `a_env = clip(u, -1, 1)`；
4. 更新时对同一个 `u` 计算 `log π_new(u|s)`。

这不是执行后有界动作的精确概率密度。它是一个刻意简单且内部一致的教学方案。

## 2. TD Error 与两个 mask

代码位置：Notebook 第 05 节的 `compute_gae` 单元。

\[
\delta_t=r_{t+1}+\gamma(1-\text{terminated}_t)V_{old}(s_{t+1})-V_{old}(s_t)
\]

- `terminated=True`：摔倒等 MDP 终止，不 bootstrap；
- `truncated=True`：外部时间上限，仍从 reset 前的真正 `next_obs` bootstrap；
- 两者都会 reset，因此都必须切断 GAE 递推。

对应实现使用：

```text
bootstrap_mask = not terminated
continue_mask  = not (terminated or truncated)
```

\[
\hat A_t=\delta_t+\gamma\lambda\,m_t^{continue}\hat A_{t+1}.
\]

最后一个 rollout 样本即使 Episode 未结束，也会停止数组内递推，但它的 `next_value` 仍进入当前样本的 TD Error。下个 rollout 继续同一 Episode；collector 不做 reset。

## 3. 固定 Critic 目标

\[
y_t=V_{old}(s_t)+\hat A_t,
\qquad
L_V=\frac12\operatorname{mean}(V_\phi(s_t)-y_t)^2.
\]

Rollout 期间的 `old_values`、`next_values`、`advantages`、`returns` 都被 detach。PPO epoch 中只有当前 Critic 的预测参与梯度。

## 4. PPO Clip

\[
\rho_t=\exp(\log\pi_\theta(\mathbf u_t|s_t)-\log\pi_{old}(\mathbf u_t|s_t))
\]

\[
L^{CLIP}=E\left[\min(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)\right].
\]

Actor 最小化 `-L_CLIP - entropy_coef * entropy`。测试覆盖正/负 advantage 的裁剪方向，以及更新前同一策略、同一 raw action 的 ratio 等于 1。

## 5. 日志坐标

`metrics.csv` 每个 rollout/update 写一行，横轴 `env_steps` 是真实环境交互数。同一 rollout 内的多个 epoch/mini-batch 统计会聚合到这一行，因此不是“每个环境步更新一次”。

`reward_ctrl` 来自环境 `info`，已经带负号：

\[
r_t=r_{forward}+r_{survive}+r_{ctrl},\quad
r_{ctrl}=-w_c\|a_{env}\|_2^2.
\]

## 6. 可观测量边界

17 维默认观测包含仿真精确给出的关节状态、躯干姿态与速度。这些量在真实机器人上通常需要编码器和状态估计，不能简单等同于无噪声的真实传感器读数。绝对 x 位置默认不在策略观测中，但用于独立评估。

速度观测会被环境裁剪到 `[-10, 10]`；分析真实仿真动力学状态时应读取 `env.unwrapped.data.qvel`。

## 7. 欠驱动与动作到驱动力

Walker2d 的 9 个广义自由度包括平面浮动基的 `rootx / rootz / rooty` 和六个腿部关节，只有后六个 DOF 由执行器直接驱动：

\[
M(q)\ddot q+h(q,\dot q)=S^\mathsf T\tau+J_c^\mathsf T\lambda.
\]

`action ∈ [-1,1]^6` 是执行器控制输入，不应直接标成 N·m。当前 motor 模型通过 actuator gear 和传动映射形成 `data.qfrc_actuator`；Notebook 第 01B 节用固定单关节输入从实际模型核对这一关系。

## 8. Control effort 不是能耗

独立评估报告：

\[
C_{effort}=\sum_t\lVert a_t^{env}\rVert_2^2,
\qquad
\bar C_{effort}=C_{effort}/T.
\]

它们分别是累计和每步平均的归一化控制输入平方和，不是机械能或电能。若研究机械功，需要另外计算 `τᵀ q̇` 并明确正功、负功、绝对功及物理子步积分。

