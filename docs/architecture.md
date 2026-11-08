# 代码结构与扩展

项目采用 src 布局，通过 editable 安装加载 `src/breakout_rl`。根目录只保留说明、依赖和开发目录，不再放算法模块或安装日志。

## 模块职责

| 目录或模块 | 职责 |
|---|---|
| `algorithms/dqn/network.py` | DQN 的 CNN |
| `algorithms/dqn/replay.py` | 单帧压缩的均匀经验回放 |
| `algorithms/dqn/optimization.py` | TD 目标、MSE、优化步骤与 epsilon 调度 |
| `algorithms/dqn/config.py` | DQN 命令参数和校验 |
| `algorithms/dqn/trainer.py` | DQN 训练循环、SwanLab、周期评估 |
| `algorithms/dqn/policy.py` | epsilon-greedy 动作选择 |
| `algorithms/dqn/checkpoints.py` | 从模型快照构建 DQN 策略 |
| `algorithms/ppo/` | Actor-Critic、rollout、GAE、PPO 更新与训练配置 |
| `algorithms/checkpoints.py` | 根据模型元数据选择 DQN / PPO 加载器 |
| `envs/vector.py` | SAME_STEP 向量环境、终止画面与整局统计提取 |
| `envs/atari.py` | 共享游戏协议、预处理与统计 |
| `envs/legacy.py` | 读取已保留旧模型所需的历史协议 |
| `evaluation/runner.py` | 接受动作选择函数，按完整游戏报告结果 |
| `infrastructure/runtime.py` | 设备、随机种子和线程配置 |
| `infrastructure/artifacts.py` | JSON 和通用模型快照格式 |
| `cli/` | 参数入口与命令组装，不实现算法更新 |
| `vendor/cleanrl/` | 保留上游来源与许可的第三方封装 |

## 增加算法

在 `algorithms/` 下新增实际实现，例如 `pqn/`；按需要放入网络、参数、优化和训练器。没有实现的算法不创建空壳目录。

环境复用 `EnvConfig` 和 `make_env`。PPO、PQN 的并行采样可在这一单环境工厂之上组装向量环境，需明确自动重置与终止处理，不把采样后端差异混成算法收益。

评估器只需要 `select_action(observation, rng) -> int`，不依赖 DQN 网络类型。各训练器在评估前后管理网络的 train/eval 模式；PPO 等算法可以提供自己的动作选择函数。

模型保存使用统一格式，算法名称与策略构建由各算法模块负责。`cli` 负责选择适合的加载器，未来增加算法时扩充对应入口，不将所有网络塞进公共工具。

## 测试与产物

测试按 `tests/algorithms/`、`tests/envs/`、`tests/evaluation/` 分组。测试自行生成小模型和必要数据，不依赖短训练目录。

训练输出统一到 `outputs/runs/`，SwanLab 数据到 `outputs/swanlab/`。输出不进入 Git，代码与 Markdown 说明进入版本管理。历史实验配置中的旧绝对路径作为历史记录保留，不改写实验数据。
