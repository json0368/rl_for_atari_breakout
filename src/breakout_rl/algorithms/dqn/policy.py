"""DQN epsilon-greedy action selection."""
import torch

def act(network, obs, epsilon, rng, device, action_count=4):
    if rng.random() < epsilon:
        return int(rng.integers(action_count))
    with torch.inference_mode():
        return int(network(torch.as_tensor(obs, device=device).unsqueeze(0)).argmax(1).item())
