"""Policy sampling with a separate evaluation RNG."""
import torch


@torch.inference_mode()
def act(agent, obs, rng, device, *, deterministic=False):
    distribution, _ = agent.distribution_and_value(torch.as_tensor(obs, device=device).unsqueeze(0))
    if deterministic:
        return int(distribution.logits.argmax(-1).item())
    probabilities = distribution.probs[0].cpu().numpy().astype(float)
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))
