# RL for Atari Breakout

基于多种深度强化学习算法的 Atari Breakout 游戏智能体设计与性能比较。

项目使用 PyTorch 实现算法，Gymnasium / ALE 提供游戏环境，SwanLab 记录训练过程，比较各算法的游戏得分、样本效率、训练耗时与稳定性。

## 安装

开发环境：Linux / WSL 2、Python 3.10、NVIDIA GPU。

```bash
conda env create -f configs/environment.yml
conda activate breakout
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[dev]'
```

## 训练

```bash
python -m swanlab login
# DQN
breakout-train --algorithm dqn --device cuda --run-name dqn-baseline --seed 1

# Double DQN
breakout-train --algorithm ddqn --device cuda --run-name ddqn-baseline --seed 1

# Dueling DQN
breakout-train --algorithm dueling --device cuda --run-name dueling-baseline --seed 1

# PPO
breakout-train --algorithm ppo --device cuda --run-name ppo-baseline --seed 1

# PQN
breakout-train --algorithm pqn --device cuda --run-name pqn-baseline --seed 1
```

默认最多训练 1000 万步，每 10 万步评估一次，连续 3 次评估平均得分未创新高则早停。`--early-stopping-patience 0` 可关闭早停。指标上传到 SwanLab 云端。模型、配置和评估结果保存在 `outputs/runs/dqn-baseline/`。

使用本地看板：

```bash
breakout-train --device cuda --run-name dqn-local --swanlab-mode local
python -m swanlab watch outputs/swanlab --host 127.0.0.1 --port 5093
```

打开 [http://127.0.0.1:5093](http://127.0.0.1:5093) 查看训练曲线。

## 评估与演示

```bash
# 自动识别 DQN / Double DQN / Dueling DQN / PPO / PQN 模型，评估 10 局
breakout-evaluate --checkpoint outputs/runs/dqn-baseline/final.pt --episodes 10

# 窗口演示
breakout-demo --checkpoint outputs/runs/dqn-baseline/final.pt

# 录制 MP4
breakout-demo --checkpoint outputs/runs/dqn-baseline/final.pt --video-dir outputs/videos/dqn-baseline
```

## 项目结构

```text
configs/                    环境配置
docs/                       实验协议、代码说明与课程报告
scripts/                    安装脚本
src/breakout_rl/
  algorithms/               各算法的网络、优化与训练逻辑
  envs/                     游戏环境与预处理
  evaluation/               公共评估流程
  infrastructure/           设备配置与模型保存
  cli/                      命令入口
  vendor/                   上游代码与许可
tests/                      测试
outputs/                    模型、日志、评估结果与视频
```

模块职责与算法扩展方式见 [代码结构说明](docs/architecture.md)。

## 测试

```bash
python -m pytest -q
```

## 参考

- [CleanRL](https://github.com/vwxyzjn/cleanrl)：算法实现与 Atari 封装。
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) / [ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment)：游戏环境。
- [SwanLab](https://github.com/SwanHubX/SwanLab)：实验记录与可视化。

上游代码的版本与许可见 [UPSTREAM.json](src/breakout_rl/vendor/cleanrl/UPSTREAM.json) 和 [LICENSE](src/breakout_rl/vendor/cleanrl/LICENSE)。
