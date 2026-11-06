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
