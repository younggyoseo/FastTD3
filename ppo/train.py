"""Minimal PPO training loop for FastTD3 environments."""
import argparse
import torch
import torch.optim as optim
from torch.nn import functional as F

from fast_td3.environments.mujoco_playground_env import make_env
from fast_td3.fast_td3_utils import EmpiricalNormalization
from .ppo import ActorCritic
from .ppo_utils import RolloutBuffer


def make_parser():
    parser = argparse.ArgumentParser(description="Train PPO")
    parser.add_argument("--env-name", type=str, default="CheetahRun")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    return parser


def main():
    args = make_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs, _, _ = make_env(args.env_name, seed=0, num_envs=args.num_envs, num_eval_envs=1, device_rank=0)
    obs = envs.reset()
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    n_act = envs.num_actions

    policy = ActorCritic(n_obs, n_act, args.hidden_dim, device=device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    buffer = RolloutBuffer(args.rollout_length, n_obs, n_act, device=device)
    global_step = 0
    while global_step < args.total_timesteps:
        for _ in range(args.rollout_length):
            norm_obs = normalizer(obs)
            with torch.no_grad():
                action, logp, value = policy.act(norm_obs)
            next_obs, reward, done, _ = envs.step(action)
            buffer.add(obs.squeeze(), action.squeeze(), logp, reward.squeeze(), done.squeeze(), value.squeeze())
            obs = next_obs
            global_step += 1
        with torch.no_grad():
            last_value = policy.value(normalizer(obs)).squeeze()
        buffer.compute_returns_and_advantage(last_value, args.gamma, args.gae_lambda)

        for _ in range(args.update_epochs):
            for b_obs, b_actions, b_logp, b_returns, b_adv in buffer.get_batches(args.batch_size):
                dist = policy.get_dist(normalizer(b_obs))
                new_logp = dist.log_prob(b_actions).sum(-1)
                ratio = (new_logp - b_logp).exp()
                pg_loss1 = -b_adv * ratio
                pg_loss2 = -b_adv * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value = policy.value(normalizer(b_obs))
                value_loss = F.mse_loss(value, b_returns)
                entropy = dist.entropy().sum(-1).mean()

                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        buffer.clear()
    print("Training complete")


if __name__ == "__main__":
    main()
