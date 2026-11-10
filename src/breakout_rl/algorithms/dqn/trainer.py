"""DQN training loop, logging, and periodic evaluation."""
import json
import random
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
import numpy as np
import swanlab
import torch
from .config import validate
from .network import QNetwork
from .replay import ReplayBuffer
from .optimization import epsilon_at, update
from .policy import act
from ...infrastructure.artifacts import save_checkpoint, write_json
from ...infrastructure.runtime import setup
from ...envs import EnvConfig, make_env, frame_count
from ...evaluation.runner import evaluate

def train(args, *, algorithm="dqn", update_fn=None, network_factory=None):
    validate(args)
    optimize = update if update_fn is None else update_fn
    build_network = QNetwork if network_factory is None else network_factory
    config = EnvConfig(repeat_action_probability=args.sticky_action_probability,
                       max_episode_frames=args.max_episode_frames)
    device = setup(args.seed, args.device, args.torch_threads)
    name = args.run_name or f"{algorithm}-seed{args.seed}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    train_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    train_config["algorithm"] = algorithm
    train_config["resolved_device"] = str(device)
    train_config["cleanrl_upstream"] = json.loads(files("breakout_rl.vendor.cleanrl").joinpath("UPSTREAM.json").read_text())
    train_config["versions"] = {p: version(p) for p in ("torch", "gymnasium", "ale-py", "swanlab", "numpy")}
    write_json(run_dir / "config.json", {"train": train_config, "env": asdict(config)})
    env = make_env(config, training=True)
    initialized = False
    step, best_score = 0, -float("inf")
    evaluations_without_improvement = 0
    early_stopped = False
    start = time.perf_counter()
    evaluation_seconds = 0.0
    try:
        swanlab.init(project=args.swanlab_project, workspace=args.swanlab_workspace,
                     experiment_name=name, config={"train": train_config, "env": asdict(config)},
                     mode=args.swanlab_mode, logdir=str(args.swanlab_logdir),
                     settings=swanlab.Settings(
                         interactive=False,
                         # Explicit package versions are already in config.json;
                         # avoid scanning unrelated global/conda environments.
                         probe=swanlab.Settings.Probe(requirements=False, conda=False),
                     ))
        initialized = True
        network = build_network(config.stack_size, env.action_space.n).to(device)
        target = build_network(config.stack_size, env.action_space.n).to(device)
        target.load_state_dict(network.state_dict())
        target.eval()
        optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
        replay = ReplayBuffer(args.buffer_size, env.observation_space.shape, args.seed)
        memory_gib = replay.image_nbytes / 1024**3
        print(f"run={run_dir} device={device} replay images={memory_gib:.2f} GiB", flush=True)
        obs, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        recent_returns = deque(maxlen=100)
        completed_games = 0
        latest_losses = {}

        def save(filename):
            save_checkpoint(run_dir / filename, network, optimizer, config,
                            train_config, step, frame_count(env),
                            algorithm=algorithm, action_count=env.action_space.n)

        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log_file:
            for step in range(1, args.total_timesteps + 1):
                epsilon = epsilon_at(step - 1, args.start_e, args.end_e,
                                     args.exploration_fraction * args.total_timesteps)
                if random.random() < epsilon:
                    action = int(env.action_space.sample())
                else:
                    with torch.inference_mode():
                        action = int(network(torch.as_tensor(obs, device=device).unsqueeze(0)).argmax(1).item())
                next_obs, reward, terminated, truncated, info = env.step(action)
                replay.add(obs, next_obs, action, reward, terminated, truncated)
                obs = next_obs
                metrics = {}
                if "episode" in info:
                    episode_return = float(info["episode"]["r"])
                    recent_returns.append(episode_return)
                    completed_games += 1
                    metrics["train/episodic_return"] = episode_return
                    metrics["train/episodic_length"] = int(info["episode"]["l"])
                    metrics["train/return_mean_100"] = float(np.mean(recent_returns))
                    metrics["train/completed_games"] = completed_games
                if terminated or truncated:
                    metrics["train/life_boundary"] = 1
                    obs, _ = env.reset()
                if step - 1 > args.learning_starts and replay.size >= args.batch_size and (step - 1) % args.train_frequency == 0:
                    latest_losses = optimize(network, target, optimizer,
                                           replay.sample(args.batch_size, device),
                                           args.gamma, args.max_grad_norm)
                if step - 1 > args.learning_starts and (step - 1) % args.target_network_frequency == 0:
                    target.load_state_dict(network.state_dict())
                if args.eval_frequency and step % args.eval_frequency == 0:
                    eval_start = time.perf_counter()
                    was_training = network.training
                    network.eval()
                    try:
                        result = evaluate(
                            lambda obs, rng: act(network, obs, args.eval_epsilon, rng, device),
                            config, episodes=args.eval_episodes,
                            seed=args.eval_seed,
                        )
                    finally:
                        network.train(was_training)
                    result["epsilon"] = args.eval_epsilon
                    evaluation_seconds += time.perf_counter() - eval_start
                    write_json(run_dir / "evaluations" / f"{step}.json", result)
                    metrics.update({"eval/mean_return": result["mean_return"],
                                    "eval/std_return": result["std_return"]})
                    if result["mean_return"] > best_score:
                        best_score = result["mean_return"]
                        evaluations_without_improvement = 0
                        save("best.pt")
                    else:
                        evaluations_without_improvement += 1
                    early_stopped = (args.early_stopping_patience > 0 and
                                     evaluations_without_improvement >= args.early_stopping_patience)
                    metrics.update({"eval/no_improvement_count": evaluations_without_improvement,
                                    "eval/early_stopped": int(early_stopped)})
                if step % args.save_frequency == 0:
                    save(f"step-{step}.pt")
                if step % args.log_frequency == 0 or step == args.total_timesteps or early_stopped:
                    metrics.update(latest_losses)
                    metrics["train/epsilon"] = epsilon
                if metrics:
                    elapsed = time.perf_counter() - start
                    metrics.update({"charts/environment_steps": step,
                                    "charts/emulator_frames": frame_count(env),
                                    "charts/wall_seconds": elapsed,
                                    "charts/evaluation_seconds": evaluation_seconds,
                                    "charts/steps_per_second": step / max(elapsed - evaluation_seconds, 1e-9)})
                    swanlab.log(metrics, step=step)
                    log_file.write(json.dumps({"step": step, **metrics}) + "\n")
                    log_file.flush()
                    print(f"step={step} frames={frame_count(env)} epsilon={epsilon:.3f} "
                          f"td_loss={latest_losses.get('losses/td_loss', float('nan')):.4f}", flush=True)
                if early_stopped:
                    print(f"Early stopping at step={step}: no improvement for "
                          f"{evaluations_without_improvement} evaluations; best mean return={best_score:.2f}",
                          flush=True)
                    break
        save("final.pt")
        print(f"Saved {run_dir / 'final.pt'}", flush=True)
    except KeyboardInterrupt:
        if "optimizer" in locals():
            save_checkpoint(run_dir / "interrupted.pt", network, optimizer, config,
                            train_config, step, frame_count(env),
                            algorithm=algorithm, action_count=env.action_space.n)
        print("Training interrupted; policy snapshot saved.", flush=True)
    finally:
        env.close()
        if initialized:
            swanlab.finish()
    return run_dir
