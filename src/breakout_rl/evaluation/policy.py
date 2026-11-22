"""Build the action selector used for evaluation and playback."""

from ..algorithms.dqn.policy import act as dqn_act
from ..algorithms.ppo.policy import act as ppo_act
from ..algorithms.pqn.policy import act as pqn_act


def action_selector(network, checkpoint, device, *, epsilon=0.01, deterministic=False):
    algorithm = checkpoint["algorithm"]
    if algorithm == "ppo":
        return lambda obs, rng: ppo_act(network, obs, rng, device, deterministic=deterministic)

    act = pqn_act if algorithm == "pqn" else dqn_act
    action_count = int(checkpoint["action_count"])
    return lambda obs, rng: act(network, obs, epsilon, rng, device, action_count=action_count)
