"""Generalized advantage estimation and clipped PPO updates."""
import numpy as np
import torch


@torch.no_grad()
def advantages_and_returns(rewards, values, boundaries, next_value, gamma, gae_lambda):
    # Rewards already include time-limit bootstrap from the FINAL observation.
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(next_value)
    for t in reversed(range(len(rewards))):
        following = next_value if t == len(rewards) - 1 else values[t + 1]
        continuation = (~boundaries[t]).float()
        delta = rewards[t] + gamma * following * continuation - values[t]
        carry = delta + gamma * gae_lambda * continuation * carry
        advantages[t] = carry
    return advantages, advantages + values


def objective(new_logprobs, old_logprobs, entropy, new_values, old_values, advantages, returns, args):
    logratio = new_logprobs - old_logprobs
    ratio = logratio.exp()
    if args.norm_adv:
        advantages = advantages - advantages.mean()
        if advantages.numel() > 1:
            advantages = advantages / (advantages.std() + 1e-8)
    policy_loss = torch.maximum(-advantages * ratio,
        -advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
    value_error = (new_values - returns).square()
    if args.clip_vloss:
        clipped = old_values + (new_values - old_values).clamp(-args.clip_coef, args.clip_coef)
        value_error = torch.maximum(value_error, (clipped - returns).square())
    value_loss = 0.5 * value_error.mean()
    entropy_mean = entropy.mean()
    loss = policy_loss - args.ent_coef * entropy_mean + args.vf_coef * value_loss
    with torch.no_grad():
        metrics = {
            "losses/policy_loss": policy_loss.item(), "losses/value_loss": value_loss.item(),
            "losses/entropy": entropy_mean.item(), "losses/total_loss": loss.item(),
            "losses/old_approx_kl": (-logratio).mean().item(),
            "losses/approx_kl": ((ratio - 1) - logratio).mean().item(),
            "losses/clipfrac": ((ratio - 1).abs() > args.clip_coef).float().mean().item(),
        }
    return loss, metrics


def update(agent, optimizer, rollout, args):
    obs, actions, old_logprobs, old_values, advantages, returns = rollout
    batch_size = len(actions)
    indices = np.arange(batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(indices)
        # A shortened final rollout uses balanced minibatches, including all samples.
        chunks = min(args.num_minibatches, max(1, batch_size // 2))
        for ids in np.array_split(indices, chunks):
            _, logprobs, entropy, values = agent.get_action_and_value(obs[ids], actions[ids])
            loss, metrics = objective(logprobs, old_logprobs[ids], entropy, values,
                                     old_values[ids], advantages[ids], returns[ids], args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            metrics["losses/grad_norm"] = float(grad_norm)
            clipfracs.append(metrics["losses/clipfrac"])
        if args.target_kl is not None and metrics["losses/approx_kl"] > args.target_kl:
            break
    metrics["losses/clipfrac"] = float(np.mean(clipfracs))
    variance = returns.var(unbiased=False)
    metrics["losses/explained_variance"] = (
        (1 - (returns - old_values).var(unbiased=False) / variance).item() if variance > 0 else float("nan"))
    metrics["charts/update_epochs"] = epoch + 1
    return metrics
