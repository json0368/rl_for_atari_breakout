"""Build action selectors and decision inspectors for evaluation and playback."""

from dataclasses import dataclass

import numpy as np
import torch

from ..algorithms.dqn.policy import act as dqn_act
from ..algorithms.ppo.policy import act as ppo_act
from ..algorithms.pqn.policy import act as pqn_act


ACTION_NAMES = ("NOOP", "FIRE", "RIGHT", "LEFT")
ACTION_LABELS = ("不动", "发球", "右移", "左移")


@dataclass(frozen=True)
class ActionDecision:
    action: int
    values: tuple[float, ...] = ()
    value_kind: str = ""
    exploratory: bool = False


def action_selector(network, checkpoint, device, *, epsilon=0.01, deterministic=False):
    algorithm = checkpoint["algorithm"]
    if algorithm == "ppo":
        return lambda obs, rng: ppo_act(network, obs, rng, device, deterministic=deterministic)

    act = pqn_act if algorithm == "pqn" else dqn_act
    action_count = int(checkpoint["action_count"])
    return lambda obs, rng: act(network, obs, epsilon, rng, device, action_count=action_count)


def decision_selector(network, checkpoint, device, *, epsilon=0.01, deterministic=False):
    """Return a selector producing ActionDecision with Q-values or action probabilities.

    RNG consumption matches action_selector() so evaluation seeds remain reproducible.
    """
    algorithm = checkpoint["algorithm"]
    if algorithm == "ppo":
        @torch.inference_mode()
        def select_ppo(obs, rng):
            distribution, _ = network.distribution_and_value(
                torch.as_tensor(obs, device=device).unsqueeze(0)
            )
            probabilities = distribution.probs[0].cpu().numpy().astype(float)
            total = float(probabilities.sum())
            if total > 0:
                probabilities /= total
            if deterministic:
                action = int(distribution.logits.argmax(-1).item())
            else:
                action = int(rng.choice(len(probabilities), p=probabilities))
            return ActionDecision(
                action=action,
                values=tuple(float(v) for v in probabilities),
                value_kind="prob",
                exploratory=False,
            )

        return select_ppo

    action_count = int(checkpoint["action_count"])

    @torch.inference_mode()
    def select_value(obs, rng):
        q_tensor = network(torch.as_tensor(obs, device=device).unsqueeze(0))[0]
        q_values = tuple(float(v) for v in q_tensor.detach().cpu().numpy().astype(np.float64))
        if rng.random() < epsilon:
            action = int(rng.integers(action_count))
            return ActionDecision(action=action, values=q_values, value_kind="q", exploratory=True)
        action = int(q_tensor.argmax(-1).item())
        return ActionDecision(action=action, values=q_values, value_kind="q", exploratory=False)

    return select_value
