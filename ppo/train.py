"""Minimal PPO training loop for FastTD3 environments."""
import argparse
import torch
import torch.optim as optim
from torch.nn import functional as F
import time
from collections import deque
from tqdm import tqdm

from fast_td3.environments.mujoco_playground_env import make_env
from fast_td3.fast_td3_utils import EmpiricalNormalization
from .ppo import ActorCritic
from .ppo_utils import RolloutBuffer


def make_parser():
    parser = argparse.ArgumentParser(description="Train PPO")
    parser.add_argument("--env_name", type=str, default="CheetahRun")
    parser.add_argument("--total-timesteps", type=int, default=20000000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rollout-length", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=1000, help="Log every N timesteps")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Evaluate every N timesteps")
    parser.add_argument("--num-eval-envs", type=int, default=10, help="Number of evaluation environments")
    return parser


def main():
    args = make_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Starting PPO training")
    print(f"Device: {device}")
    print(f"Environment: {args.env_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Rollout length: {args.rollout_length}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Evaluation interval: {args.eval_interval}")
    print(f"Number of eval environments: {args.num_eval_envs}")
    print("-" * 60)
    
    # Create training environments
    envs, _, _ = make_env(args.env_name, seed=0, num_envs=args.num_envs, num_eval_envs=1, device_rank=0)
    
    # Create separate evaluation environments
    eval_envs, _, _ = make_env(args.env_name, seed=42, num_envs=args.num_eval_envs, num_eval_envs=1, device_rank=0)
    
    obs = envs.reset()
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    n_act = envs.num_actions

    policy = ActorCritic(n_obs, n_act, args.hidden_dim, device=device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    buffer = RolloutBuffer(args.rollout_length * args.num_envs, n_obs, n_act, device=device)
    
    # Progress tracking variables
    global_step = 0
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    current_episode_reward = torch.zeros(args.num_envs, device=device)
    current_episode_length = torch.zeros(args.num_envs, device=device)
    num_episodes = 0
    start_time = time.time()
    last_log_time = start_time
    
    # Training metrics
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    num_updates = 0
    
    # Evaluation metrics
    eval_returns = []
    eval_lengths = []
    
    def evaluate():
        """Evaluate the current policy on separate evaluation environments."""
        normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        obs = eval_envs.reset()

        # Run for a fixed number of steps
        for _ in range(eval_envs.max_episode_steps):
            with torch.no_grad():
                norm_obs = normalizer(obs)
                action, _, _ = policy.act(norm_obs)

            next_obs, rewards, dones, _ = eval_envs.step(action)
            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        normalizer.train()
        return episode_returns.mean().item(), episode_lengths.mean().item()
    
    print("Starting training loop...")
    
    # Main training loop with progress bar
    with tqdm(total=args.total_timesteps, desc="Training Progress", unit="steps") as pbar:
        while global_step < args.total_timesteps:
            # Data collection phase
            rollout_start_time = time.time()
            for step in tqdm(range(args.rollout_length), desc="Data Collection", leave=False):
                norm_obs = normalizer(obs)
                with torch.no_grad():
                    action, logp, value = policy.act(norm_obs)
                next_obs, reward, done, _ = envs.step(action)
                
                # Add each environment's transition to buffer
                for env_idx in range(args.num_envs):
                    buffer.add(
                        obs[env_idx], 
                        action[env_idx], 
                        logp[env_idx], 
                        reward[env_idx], 
                        done[env_idx], 
                        value[env_idx]
                    )
                
                obs = next_obs
                global_step += args.num_envs  # Update by number of environments
                pbar.update(args.num_envs)
                
                # Track episode statistics
                current_episode_reward += reward
                current_episode_length += 1
                
                # Check for episode completions
                if done.any():
                    for env_idx in range(args.num_envs):
                        if done[env_idx]:
                            episode_rewards.append(current_episode_reward[env_idx].item())
                            episode_lengths.append(current_episode_length[env_idx].item())
                            num_episodes += 1
                            current_episode_reward[env_idx] = 0
                            current_episode_length[env_idx] = 0
            
            rollout_time = time.time() - rollout_start_time
            
            # Compute advantages - handle each environment separately
            with torch.no_grad():
                last_values = policy.value(normalizer(obs))  # Shape: [num_envs]
            
            # Compute advantages for each environment's data separately
            buffer.compute_returns_and_advantage(last_values.mean().item(), args.gamma, args.gae_lambda)

            # Policy update phase
            update_start_time = time.time()
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_updates = 0
            
            for epoch in tqdm(range(args.update_epochs), desc="Policy Updates", leave=False):
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
                    
                    # Accumulate metrics
                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    epoch_entropy += entropy.item()
                    epoch_updates += 1
            
            update_time = time.time() - update_start_time
            
            # Update global metrics
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss
            total_entropy += epoch_entropy
            num_updates += 1
            
            buffer.clear()
            
            # Evaluation phase
            eval_avg_return = None
            eval_avg_length = None
            # if args.eval_interval > 0 and global_step % args.eval_interval == 0:
            print(f"\nEvaluating at global step {global_step}")
            eval_avg_return, eval_avg_length = evaluate()
            eval_returns.append(eval_avg_return)
            eval_lengths.append(eval_avg_length)
            print(f"Evaluation - Avg Return: {eval_avg_return:.3f}, Avg Length: {eval_avg_length:.1f}")
            
            # Update main progress bar description with current metrics
            avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
            avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
            progress = (global_step / args.total_timesteps) * 100
            
            # Add evaluation info to progress bar if available
            eval_info = f" | Eval: {eval_avg_return:.3f}" if eval_avg_return is not None else ""
            
            pbar.set_description(
                f"Training ({progress:.1f}%) | Reward: {avg_reward:.3f}{eval_info} | Policy Loss: {avg_policy_loss:.6f} | Episodes: {num_episodes}"
            )
            
            # Logging
            if global_step % args.log_interval == 0 or global_step >= args.total_timesteps:
                current_time = time.time()
                elapsed_time = current_time - start_time
                time_since_last_log = current_time - last_log_time
                
                # Calculate metrics
                avg_length = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0
                avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
                avg_entropy = total_entropy / num_updates if num_updates > 0 else 0
                
                # Calculate FPS
                fps = args.log_interval / time_since_last_log if time_since_last_log > 0 else 0
                
                print(f"\nTraining Progress - Step {global_step:,}/{args.total_timesteps:,} ({progress:.1f}%)")
                print(f"Elapsed: {elapsed_time:.1f}s | FPS: {fps:.1f} | Time since last log: {time_since_last_log:.1f}s")
                print(f"Episode Stats: Avg Reward: {avg_reward:.3f} | Avg Length: {avg_length:.1f} | Episodes: {num_episodes}")
                print(f"Loss Stats: Policy: {avg_policy_loss:.6f} | Value: {avg_value_loss:.6f} | Entropy: {avg_entropy:.6f}")
                print(f"Timing: Rollout: {rollout_time:.3f}s | Update: {update_time:.3f}s")
                print(f"Epoch {epoch+1}/{args.update_epochs} | Updates: {epoch_updates}")
                
                # Add evaluation results to logging if available
                if eval_avg_return is not None:
                    print(f"Evaluation: Avg Return: {eval_avg_return:.3f} | Avg Length: {eval_avg_length:.1f}")
                
                print("-" * 60)
                
                last_log_time = current_time
    
    total_time = time.time() - start_time
    print(f"\nTraining completed!")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Final stats: {num_episodes} episodes, {global_step:,} timesteps")
    print(f"Average FPS: {global_step/total_time:.1f}")
    
    # Print final evaluation results
    if eval_returns:
        print(f"Final evaluation return: {eval_returns[-1]:.3f}")
        print(f"Best evaluation return: {max(eval_returns):.3f}")


if __name__ == "__main__":
    main()
