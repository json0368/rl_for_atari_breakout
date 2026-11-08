"""Independent epsilon-greedy exploration in each environment."""
import torch


def epsilon_at(step, start, end, duration):
    return start + min(step / max(duration, 1), 1.0) * (end - start)


@torch.no_grad()
def sample_actions(q_values, epsilon):
    greedy = q_values.argmax(dim=-1)
    random_actions = torch.randint(q_values.shape[-1], greedy.shape, device=q_values.device)
    explore = torch.rand(greedy.shape, device=q_values.device) < epsilon
    return torch.where(explore, random_actions, greedy)


@torch.inference_mode()
def act(network, obs, epsilon, rng, device, action_count=4):
    if rng.random() < epsilon:
        return int(rng.integers(action_count))
    return int(network(torch.as_tensor(obs, device=device).unsqueeze(0)).argmax(-1).item())
