"""PPO rollout collection, optimization, evaluation and early stopping."""
import json
import math
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
from .network import ActorCritic
from .optimization import advantages_and_returns, update
from .policy import act
from ...envs import EnvConfig
from ...envs.vector import make_vector_env, total_frames, episode_records, timeout_observations
from ...evaluation.runner import evaluate
from ...infrastructure.artifacts import save_checkpoint, write_json
from ...infrastructure.runtime import setup


def train(args):
    validate(args)
    device = setup(args.seed, args.device, args.torch_threads)
    config = EnvConfig(repeat_action_probability=args.sticky_action_probability,
                       max_episode_frames=args.max_episode_frames)
    name = args.run_name or f"ppo-seed{args.seed}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    batch_size = args.num_envs * args.num_steps
    iterations = math.ceil(args.total_timesteps / batch_size)
    train_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    train_config.update(algorithm="ppo", resolved_device=str(device), batch_size=batch_size,
                        minibatch_size=batch_size // args.num_minibatches,
                        num_iterations=iterations, optimizer_eps=1e-5)
    train_config["cleanrl_upstream"] = json.loads(files("breakout_rl.vendor.cleanrl").joinpath("UPSTREAM.json").read_text())
    train_config["versions"] = {p: version(p) for p in ("torch", "gymnasium", "ale-py", "swanlab", "numpy")}
    write_json(run_dir / "config.json", {"train": train_config, "env": asdict(config)})
    envs = make_vector_env(config, args.num_envs)
    initialized = False
    step = 0
    best_score = -float("inf")
    no_improvement = 0
    early_stopped = False
    evaluation_seconds = 0.0
    next_eval, next_save = args.eval_frequency, args.save_frequency
    recent_returns = deque(maxlen=100)
    completed_games = 0
    start = time.perf_counter()
    try:
        swanlab.init(project=args.swanlab_project, workspace=args.swanlab_workspace,
                     experiment_name=name, config={"train": train_config, "env": asdict(config)},
                     mode=args.swanlab_mode, logdir=str(args.swanlab_logdir),
                     settings=swanlab.Settings(interactive=False,
                         probe=swanlab.Settings.Probe(requirements=False, conda=False)))
        initialized = True
        agent = ActorCritic(config.stack_size, envs.single_action_space.n).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        def save(filename):
            save_checkpoint(run_dir / filename, agent, optimizer, config, train_config,
                            step, total_frames(envs), algorithm="ppo", action_count=envs.single_action_space.n)

        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.as_tensor(next_obs, device=device)
        print(f"run={run_dir} device={device} environments={args.num_envs} batch={batch_size}", flush=True)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log_file:
            def log(metrics):
                elapsed = time.perf_counter() - start
                metrics.update({"charts/environment_steps": step, "charts/emulator_frames": total_frames(envs),
                                "charts/wall_seconds": elapsed, "charts/evaluation_seconds": evaluation_seconds,
                                "charts/steps_per_second": step / max(elapsed - evaluation_seconds, 1e-9)})
                swanlab.log(metrics, step=step)
                log_file.write(json.dumps({"step": step, **metrics}) + "\n")
                log_file.flush()

            for iteration in range(iterations):
                if args.anneal_lr:
                    optimizer.param_groups[0]["lr"] = args.learning_rate * (1 - iteration / iterations)
                rollout_steps = min(args.num_steps, (args.total_timesteps - step) // args.num_envs)
                shape = (rollout_steps, args.num_envs)
                observations = torch.empty(shape + envs.single_observation_space.shape, dtype=torch.uint8, device=device)
                actions = torch.empty(shape, dtype=torch.long, device=device)
                logprobs = torch.empty(shape, device=device)
                values = torch.empty(shape, device=device)
                rewards = torch.empty(shape, device=device)
                boundaries = torch.empty(shape, dtype=torch.bool, device=device)
                for t in range(rollout_steps):
                    observations[t] = next_obs
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(next_obs)
                    actions[t], logprobs[t], values[t] = action, logprob, value
                    raw_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                    step += args.num_envs
                    rewards[t] = torch.as_tensor(reward, device=device)
                    boundaries[t] = torch.as_tensor(terminated | truncated, device=device)
                    indices, final_obs = timeout_observations(infos, terminated, truncated)
                    if len(indices):
                        with torch.no_grad():
                            terminal_values = agent.get_value(torch.as_tensor(final_obs, device=device))
                        rewards[t, indices] += args.gamma * terminal_values
                    next_obs = torch.as_tensor(raw_obs, device=device)
                    for score, length in episode_records(infos):
                        recent_returns.append(score)
                        completed_games += 1
                        log({"train/episodic_return": score, "train/episodic_length": length,
                             "train/return_mean_100": float(np.mean(recent_returns)),
                             "train/completed_games": completed_games})
                        print(f"step={step} episodic_return={score:g}", flush=True)
                with torch.no_grad():
                    next_value = agent.get_value(next_obs)
                    advantages, returns = advantages_and_returns(
                        rewards, values, boundaries, next_value, args.gamma, args.gae_lambda)
                rollout = (observations.flatten(0, 1), actions.flatten(), logprobs.flatten(),
                           values.flatten(), advantages.flatten(), returns.flatten())
                metrics = update(agent, optimizer, rollout, args)
                metrics["charts/learning_rate"] = optimizer.param_groups[0]["lr"]
                if args.eval_frequency and step >= next_eval:
                    eval_start = time.perf_counter()
                    agent.eval()
                    try:
                        result = evaluate(lambda obs, rng: act(agent, obs, rng, device,
                                              deterministic=args.eval_deterministic),
                                          config, episodes=args.eval_episodes, seed=args.eval_seed)
                    finally:
                        agent.train()
                    evaluation_seconds += time.perf_counter() - eval_start
                    result["deterministic"] = args.eval_deterministic
                    write_json(run_dir / "evaluations" / f"{step}.json", result)
                    if result["mean_return"] > best_score:
                        best_score = result["mean_return"]
                        no_improvement = 0
                        save("best.pt")
                    else:
                        no_improvement += 1
                    early_stopped = args.early_stopping_patience > 0 and no_improvement >= args.early_stopping_patience
                    metrics.update({"eval/mean_return": result["mean_return"], "eval/std_return": result["std_return"],
                                    "eval/no_improvement_count": no_improvement, "eval/early_stopped": int(early_stopped)})
                    next_eval = (step // args.eval_frequency + 1) * args.eval_frequency
                if step >= next_save:
                    save(f"step-{step}.pt")
                    next_save = (step // args.save_frequency + 1) * args.save_frequency
                log(metrics)
                print(f"step={step} policy_loss={metrics['losses/policy_loss']:.4f} "
                      f"value_loss={metrics['losses/value_loss']:.4f}", flush=True)
                if early_stopped:
                    print(f"Early stopping: no improvement for {no_improvement} evaluations; "
                          f"best mean return={best_score:g}", flush=True)
                    break
        save("final.pt")
        write_json(run_dir / "training_summary.json", {"step": step, "early_stopped": early_stopped,
                   "best_mean_return": best_score if best_score != -float("inf") else None,
                   "no_improvement_count": no_improvement})
        print(f"Saved {run_dir / 'final.pt'}", flush=True)
    except KeyboardInterrupt:
        if "optimizer" in locals():
            save("interrupted.pt")
            print("Training interrupted; policy snapshot saved.", flush=True)
    finally:
        envs.close()
        if initialized:
            swanlab.finish()
    return run_dir
