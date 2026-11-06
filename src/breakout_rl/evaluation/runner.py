"""Evaluate complete games using a caller-provided action function."""
import numpy as np
from ..envs import frame_count, make_env

def evaluate(select_action, config, *, episodes=10, seed=10_000,
             render_mode=None, video_dir=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = make_env(config, render_mode=render_mode, video_dir=video_dir)
    rng = np.random.default_rng(seed)
    records = []
    try:
        for episode in range(episodes):
            before = frame_count(env)
            obs, _ = env.reset(seed=seed + episode)
            score, length = 0.0, 0
            while True:
                action = int(select_action(obs, rng))
                obs, reward, terminated, truncated, info = env.step(action)
                score += float(reward)
                length += 1
                if "episode" in info:
                    score = float(info["episode"]["r"])
                    break
                if terminated or truncated:
                    # Artificial life boundary: continue the same full game.
                    obs, _ = env.reset()
            record = {"seed": seed + episode, "return": score, "steps": length,
                      "frames": frame_count(env) - before,
                      "terminated": bool(terminated), "truncated": bool(truncated)}
            records.append(record)
            print(f"episode={episode + 1} score={score:g} steps={length}", flush=True)
    finally:
        env.close()
    returns = [r["return"] for r in records]
    return {"episodes": records, "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            "base_seed": seed}
