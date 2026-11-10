# DQN 基线与实验协议

参考版本：[CleanRL fe8d8a03c41a7ef5b523e2e354bd01c363e786bb](https://github.com/vwxyzjn/cleanrl/tree/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb)。上游封装、来源与许可位于 `src/breakout_rl/vendor/cleanrl/`，不额外保存整份重复的训练脚本。

| 参数 | 默认值 |
|---|---|
| 游戏 | BreakoutNoFrameskip-v4，黏性概率0 |
| 输入 | 4×84×84，CNN 内除以255 |
| 网络 | 32/64/64 卷积，3136→512→4，ReLU |
| 优化器 | Adam，学习率1e-4，其余 PyTorch 默认值 |
| Loss | MSE |
| 梯度裁剪 | 默认关闭；显式指定属于偏离默认基线 |
| 总预算 | 最多1000万决策步 |
| 早停 | 连续3次评估平均得分未创新高；`--early-stopping-patience 0` 关闭 |
| 回放容量 | 100万条，均匀采样 |
| 预热 | 80000步，按上游零基索引时点更新 |
| Batch / gamma | 32 / 0.99 |
| 更新频率 | 每4步训练，每1000步硬更新目标网络 |
| epsilon | 1→0.01，默认在前100万步内衰减 |

环境顺序为完整游戏原始统计、随机空操作、跳帧4与最大池化、EpisodicLife、FireReset、奖励符号裁剪、缩放、灰度和四帧堆叠。丢命停止训练目标的 bootstrap，剩余生命继续原游戏；真实 game over 和时间限制产生整局得分。时间截断使用终止前状态 bootstrap。默认每局108000底层帧。

单帧回放在采样时重建状态，百万帧图像数组约6.57GiB系统内存。生命边界、初始帧填充和环形覆盖均有回归测试。它保留容量和均匀采样策略，但不承诺随机序列或最终分数与上游完全一致。

## 评价指标

- `train/episodic_return`：完整游戏的原始累计奖励。
- `train/return_mean_100`：最近最多100局平均原始奖励。
- `train/episodic_length`：底层帧长度；`train/completed_games`：完整游戏数量。
- `train/life_boundary`：训练片段终止标记，不能当作整局结束次数。
- `losses/td_loss`、`losses/q_mean`、`losses/grad_norm`、`train/epsilon`。
- `eval/mean_return`、`eval/std_return`：独立完整游戏评估，奖励不裁剪。
- `eval/no_improvement_count`：连续未提升评估次数；`eval/early_stopped`：是否触发早停。
- 决策步数、实际模拟器帧数、训练及评估耗时、吞吐量。

默认每10万步评估5局，epsilon=0.01；可通过 `--eval-frequency 0` 关闭周期评估。不同算法统一环境协议、交互预算和独立评价种子，保留各自优化机制。正式比较建议使用多个训练种子，并分别报告样本效率和实际耗时。

## 模型管理

每个实验保存配置、依赖版本、模型和指标。模型快照不包含完整回放池、目标网络与随机数状态，因此当前不能精确恢复训练。对齐前后的实验必须分开命名，不能拼成同一条训练曲线。历史模型的环境兼容逻辑用于复查既有实验。

## PPO 基线

PPO 参考同一 CleanRL 版本的 `cleanrl/ppo_atari.py`，与 DQN 共用游戏、预处理、奖励裁剪与完整游戏评估。训练命令为 `breakout-train --algorithm ppo`。

| 参数 | 默认值 |
|---|---|
| 网络 | 32/64/64 卷积，512 隐藏层，共享 CNN + actor / critic；正交初始化 |
| 采样 | 8 个 SyncVectorEnv 环境，每个128步，共1024条交互 |
| 优化器 | Adam，学习率2.5e-4，eps=1e-5，线性衰减 |
| GAE | gamma=0.99，lambda=0.95 |
| 更新 | 4个 minibatch，每批4轮；minibatch 内优势标准化 |
| PPO / value clipping | 系数0.1，启用价值裁剪 |
| 熵 / value 系数 | 0.01 / 0.5 |
| 梯度裁剪 | 0.5 |
| 总预算 / 早停 | 最多1000万交互步；连续3次评估未创新高则停止 |

`step` 为所有环境累计交互数，向量环境执行一次增加8步。每轮更新后记录损失；跨过10万步阈值后评估，默认第一次在100352步。保存也在更新后进行。最后一轮缩短采样以用完交互预算，`total-timesteps` 须为环境数的整数倍。

Gymnasium 使用 SAME_STEP 自动重置，终止画面通过 `final_obs` 获取。丢命和 game over 不 bootstrap；时间截断使用终止画面的价值 bootstrap，同时切断 GAE 的跨局递推。这是为统一项目环境协议对上游合并终止/截断处理所做的调整。观测按 uint8 保存，在 CNN 中转换并除以255。

评估默认按策略概率采样，使用独立 NumPy 随机数，不消耗训练用的 PyTorch 随机数。`--eval-deterministic` 切换为最大概率动作。评估和演示自动识别模型算法，PPO 可通过 `--deterministic` 选择动作模式。

SwanLab 记录 `losses/policy_loss`、`value_loss`、`entropy`、`total_loss`、`approx_kl`、`old_approx_kl`、`clipfrac`、`explained_variance`、`grad_norm`，以及学习率、整局得分、评估和早停指标。`--target-kl` 可选参数只结束当前批次的优化，整场训练的早停由 `--early-stopping-patience` 控制。早停保存 `final.pt`，历史最佳保存 `best.pt`，手动中断保存 `interrupted.pt`。

## PQN 基线

PQN 参考同一固定版本的 `cleanrl/pqn_atari_envpool.py`。训练命令为 `breakout-train --algorithm pqn`，游戏、预处理、完整游戏评估与 DQN / PPO 保持一致。

| 参数 | 默认值 |
|---|---|
| 网络 | 32/64/64 卷积与512隐藏层，各层使用 LayerNorm + ReLU，正交初始化 |
| 优化器 | RAdam，学习率2.5e-4，eps=1e-8，线性衰减 |
| 目标 | Q(lambda)，lambda=0.65，gamma=0.99；MSE |
| 采样 / 更新 | 8环境 x 128步，4个 minibatch，更新4轮 |
| epsilon | 1→0.01，前100万交互步内衰减，每个环境独立探索 |
| 梯度裁剪 | 10.0 |
| 预算 / 早停 | 最多1000万交互步；连续3次评估未创新高早停 |

采样结束后固定 Q(lambda) 目标，所有更新轮次复用；不使用长期经验回放池或目标网络。SwanLab 记录 `losses/td_loss`、`losses/q_mean`、`losses/grad_norm`、epsilon、学习率、游戏得分和早停状态。

与上游的适配：EnvPool 替换为共享 Gymnasium SyncVectorEnv，继续使用 `BreakoutNoFrameskip-v4`、黏性概率0和 SAME_STEP 重置。丢命与 game over 截断价值；时间截断用 `final_obs` 的最大 Q 值 bootstrap，并切断跨局 Q(lambda) 递推。观测使用 uint8，最后一轮按剩余预算缩短。评估、保存与日志采用项目公共协议，与 PPO 一样在更新完成后执行周期评估与保存。

默认每约10万步评估5局，评估 epsilon=0.01。`--early-stopping-patience 0` 关闭早停；早停保存 `final.pt`，历史最佳为 `best.pt`，手动中断保存 `interrupted.pt`。训练、评估与演示的模型格式均标记为 `pqn`，由命令入口自动识别。

## Double DQN

训练命令为 `breakout-train --algorithm ddqn`。参考 [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)，在现有 CleanRL DQN 基线上扩展。

在线网络选择下一状态的动作 `a* = argmax_a Q_online(s_next, a)`，目标网络评估该动作 `Q_target(s_next, a*)`。TD 目标为 `r + gamma * (1 - terminated) * Q_target(s_next, a*)`；目标计算不参与梯度更新。

网络、优化器、MSE、回放采样、epsilon、预热与更新频率均沿用 DQN。丢命与 game over 不 bootstrap，时间截断使用终止前画面 bootstrap。训练循环、SwanLab、完整游戏评估和早停共用 DQN 逻辑。模型元数据独立标记为 `ddqn`，评估与演示命令自动识别。

默认保留连续3次评估未创新高早停；完整预算对比可使用 `--early-stopping-patience 0`。`best.pt`、`final.pt` 与 `interrupted.pt` 分别保存最佳、结束与手动中断时的策略。

## Dueling DQN

训练命令为 `breakout-train --algorithm dueling`。参考 [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)，采用均值聚合：`Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)`。

共享32/64/64卷积编码器，value 和 advantage 分支各使用512维隐藏层，分别输出1个状态价值与4个动作优势。每个样本独立沿动作维度去均值；原始像素进入 CNN 前转换为浮点数并除以255。

该版本采用标准 DQN 的目标网络最大值 TD 目标。网络之外的 Adam、MSE、回放、探索与目标同步频率均沿用 DQN，完整游戏统计、SwanLab 和早停复用公共训练流程。模型标记为 `dueling`，评估与演示自动构建对应的双分支网络。

默认保留连续3次评估未创新高早停；完整预算对比加 `--early-stopping-patience 0`。最佳、结束与中断策略分别保存为 `best.pt`、`final.pt`、`interrupted.pt`。
