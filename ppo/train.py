"""Minimal PPO training loop for FastTD3 environments."""
from .hyperparams import get_args
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
import time
from collections import deque
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp

from fast_td3.environments.mujoco_playground_env import make_env
from fast_td3.fast_td3_utils import EmpiricalNormalization
from .ppo import ActorCritic, calculate_network_norms
from .ppo_utils import RolloutBuffer, save_ppo_params
from tensordict import TensorDict

import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ['XLA_FLAGS'] = "--xla_gpu_triton_gemm_any=True" # Enable triton gemm

import torch._dynamo
torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_enabled = args.amp and torch.cuda.is_available()
    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb if requested
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )
    
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
    print(f"Using wandb: {args.use_wandb}")
    print("-" * 60)
    
    # Create training environments
    envs, _, _ = make_env(args.env_name, seed=args.seed, num_envs=args.num_envs, num_eval_envs=1, device_rank=0)
    
    # Create separate evaluation environments
    eval_envs, _, render_env = make_env(args.env_name, seed=args.seed+42, num_envs=args.num_eval_envs, num_eval_envs=1, device_rank=0)
    
    obs = envs.reset()
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    n_act = envs.num_actions

    policy = ActorCritic(n_obs, n_act, args.hidden_dim, device=device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    if args.compile:
        policy_act = torch.compile(policy.act)
        policy_value = torch.compile(policy.value)
        policy_get_dist = torch.compile(policy.get_dist)
        normalize_obs = torch.compile(normalizer.forward)
    else:
        policy_act = policy.act
        policy_value = policy.value
        policy_get_dist = policy.get_dist
        normalize_obs = normalizer.forward

    buffer = RolloutBuffer(args.rollout_length, args.num_envs, n_obs, n_act, device=device)
    
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
        max_steps = getattr(eval_envs, 'max_episode_length', getattr(eval_envs, 'max_episode_steps', 1000))
        for _ in range(max_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                norm_obs = normalize_obs(obs)
                action, _, _ = policy_act(norm_obs)

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
    
    def render_with_rollout():
        normalizer.eval()
        env_type = args.env_type

        # Quick rollout for rendering
        if env_type == "humanoid_bench":
            obs = render_env.reset()
            renders = [render_env.render()]
        elif env_type == "isaaclab":
            raise NotImplementedError(
                "We don't support rendering for IsaacLab environments"
            )
        else:
            obs = render_env.reset()
            render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            renders = [render_env.state]
        for i in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                norm_obs = normalize_obs(obs)
                action, _, _ = policy_act(norm_obs)
            next_obs, _, done, _ = render_env.step(action)
            if env_type == "mujoco_playground":
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            if i % 2 == 0:
                if env_type == "humanoid_bench":
                    renders.append(render_env.render())
                else:
                    renders.append(render_env.state)
            if done.any():
                break
            obs = next_obs

        if env_type == "mujoco_playground":
            renders = render_env.render_trajectory(renders)

        normalizer.train()
        return renders

    print("Starting training loop...")
    
    # Main training loop with progress bar
    with tqdm(total=args.total_timesteps, desc="Training Progress", unit="steps") as pbar:
        while global_step < args.total_timesteps:
            logs_dict = TensorDict()
            # Evaluation phase - run before data collection
            eval_avg_return = None
            eval_avg_length = None
            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                print(f"\nEvaluating at global step {global_step}")
                eval_avg_return, eval_avg_length = evaluate()
                eval_returns.append(eval_avg_return)
                eval_lengths.append(eval_avg_length)
                print(f"*** Evaluation - Avg Return: {eval_avg_return:.3f}, Avg Length: {eval_avg_length:.1f}****")
                
                # Render video if requested
                # print(f"Rendering video at global step {global_step}")
                # renders = render_with_rollout()
                if args.use_wandb:
                #     wandb.log(
                #         {
                #             "render_video": wandb.Video(
                #                 np.array(renders).transpose(0, 3, 1, 2),  # Convert to (T, C, H, W) format
                #                 fps=30,
                #                 format="gif",
                #             )
                #         },
                #         step=global_step,
                #     )
                    # log the evaluation results
                    wandb.log(
                        {
                            "eval_avg_return": eval_avg_return,
                            "eval_avg_length": eval_avg_length,
                        },
                        step=global_step,
                    )
            
            # Data collection phase
            rollout_start_time = time.time()
            for step in tqdm(range(args.rollout_length), desc="Data Collection", leave=False):
                with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                    norm_obs = normalize_obs(obs)
                    action, logp, value = policy_act(norm_obs)
                next_obs, reward, done, _ = envs.step(action)
                # Add each environment's transition to buffer
                buffer.add(
                    obs, 
                    action, 
                    logp, 
                    reward, 
                    done, 
                    value
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
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                last_values = policy_value(normalize_obs(obs))  # Shape: [num_envs]
            # Compute advantages for each environment's data separately
            buffer.compute_returns_and_advantage(last_values, args.gamma, args.gae_lambda, args.num_envs)

            # Policy update phase
            update_start_time = time.time()
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_updates = 0
            
            for epoch in tqdm(range(args.update_epochs), desc="Policy Updates", leave=False):
                for b_obs, b_actions, b_logp, b_returns, b_adv in buffer.get_batches(args.batch_size):
                    with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                        dist = policy_get_dist(normalize_obs(b_obs))

                        # Since b_actions are tanh-transformed, we need to inverse them
                        raw_actions = torch.atanh(torch.clamp(b_actions, -0.999, 0.999))
                        new_logp = dist.log_prob(raw_actions).sum(-1)
                        # Apply tanh Jacobian correction
                        new_logp = new_logp - (2 * (torch.log(torch.tensor(2.0)) - raw_actions - torch.nn.functional.softplus(-2 * raw_actions))).sum(-1)

                        ratio = (new_logp - b_logp).exp()
                        pg_loss1 = -b_adv * ratio
                        pg_loss2 = -b_adv * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                        value = policy_value(normalize_obs(b_obs))
                        value_loss = F.mse_loss(value, b_returns)
                        entropy = dist.entropy().sum(-1).mean()

                        loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    
                    # Apply gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(),
                        max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
                    )

                    scaler.step(optimizer)
                    scaler.update()
                    
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
            # if global_step % args.log_interval == 0 or global_step >= args.total_timesteps:
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
            
            # Log to wandb if enabled
            if args.use_wandb:
                logs_dict["avg_reward"] = avg_reward
                logs_dict["avg_length"] = avg_length
                logs_dict["policy_loss"] = avg_policy_loss
                logs_dict["value_loss"] = avg_value_loss
                logs_dict["entropy"] = avg_entropy
                logs_dict["num_episodes"] = num_episodes
                logs_dict["rollout_time"] = rollout_time
                logs_dict["update_time"] = update_time
                logs_dict["grad_norm"] = grad_norm.item() if 'grad_norm' in locals() else 0.0

                policy_norms = calculate_network_norms(policy, "policy")
                logs_dict.update(policy_norms)

                if eval_avg_return is not None:
                    logs_dict["eval_avg_return"] = eval_avg_return
                    logs_dict["eval_avg_length"] = eval_avg_length

                wandb.log({"speed": fps, "frame": global_step, **logs_dict}, step=global_step)

            if (
                args.save_interval > 0
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving model at global step {global_step}")
                save_ppo_params(
                    global_step,
                    policy,
                    normalizer,
                    args,
                    f"{args.output_dir}/{run_name}_{global_step}.pt",
                )

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
    
    # Log final results to wandb
    if args.use_wandb:
        wandb.log({
            "final_eval_return": eval_returns[-1] if eval_returns else 0,
            "best_eval_return": max(eval_returns) if eval_returns else 0,
            "total_training_time": total_time,
            "final_fps": global_step/total_time,
        }, step=global_step)
        wandb.finish()

    save_ppo_params(
        global_step,
        policy,
        normalizer,
        args,
        f"{args.output_dir}/{run_name}_final.pt",
    )


if __name__ == "__main__":
    main()
