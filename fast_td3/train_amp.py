import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import random
import time

import tqdm
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from tensordict import TensorDict, from_module

from fast_td3 import Actor, Critic
from fast_td3_utils import (
    SimpleReplayBuffer,
    EmpiricalNormalization,
    RewardNormalizer,
    save_params,
)
from amp import Discriminator
from hyperparams import get_args


torch.set_float32_matmul_precision("high")


def main():
    args = get_args()
    print(args)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    from environments.isaaclab_amp_env import IsaacLabAMPEnv

    envs = IsaacLabAMPEnv(
        args.env_name,
        device.type,
        args.num_envs,
        args.seed,
        action_bounds=args.action_bounds,
    )
    eval_envs = envs

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    n_critic_obs = n_obs
    n_amp_obs = (
        envs.num_amp_obs if type(envs.num_amp_obs) == int else envs.num_amp_obs[0]
    )
    action_low, action_high = -1.0, 1.0

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        amp_obs_normalizer = EmpiricalNormalization(shape=n_amp_obs, device=device)
    else:
        obs_normalizer = nn.Identity()
        amp_obs_normalizer = nn.Identity()

    if args.reward_normalization:
        reward_normalizer = RewardNormalizer(
            gamma=args.gamma,
            device=device,
            g_max=min(abs(args.v_min), abs(args.v_max)),
        )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    actor = Actor(**actor_kwargs)
    actor_detach = Actor(**actor_kwargs)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    qnet = Critic(**critic_kwargs)
    qnet_target = Critic(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

    discriminator = Discriminator(
        n_amp_obs,
        args.discriminator_hidden_dim,
        args.lsgan_reward_scale,
        args.discriminator_gradient_penalty,
        device=device,
    )

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    discriminator_optimizer = optim.AdamW(
        list(discriminator.parameters()),
        lr=torch.tensor(args.discriminator_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_amp_obs=n_amp_obs,
        n_act=n_act,
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def evaluate():
        obs_normalizer.eval()
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        obs = eval_envs.reset()

        # Run for a fixed number of steps
        for i in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs)
                actions = actor(obs)

            next_obs, rewards, dones, infos = eval_envs.step(actions.float())

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

        obs_normalizer.train()
        return episode_returns.mean().item(), episode_lengths.mean().item()

    def update_main(data):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = observations
            next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]

            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            qnet.parameters(),
            max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
        )
        scaler.step(q_optimizer)
        scaler.update()

        return (
            critic_grad_norm.detach(),
            qf_loss.detach(),
            qf1_next_target_value.max().detach(),
            qf1_next_target_value.min().detach(),
        )

    def update_pol(data):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            qf1, qf2 = qnet(data["observations"], actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
        )
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
        )

    def update_discriminator(data):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            amp_replay_obs = data["amp_observations"]
            amp_motion_obs = data["amp_motion_observations"]

            if args.discriminator_gradient_penalty > 0:
                amp_motion_obs.requires_grad_(True)

            amp_replay_logits = discriminator(amp_replay_obs)
            amp_motion_logits = discriminator(amp_motion_obs)

            if args.gan_type == "lsgan":
                # NEW: LSGAN discriminator loss
                discriminator_prediction_loss = 0.5 * (
                    F.mse_loss(amp_replay_logits, -torch.ones_like(amp_replay_logits))
                    + F.mse_loss(amp_motion_logits, torch.ones_like(amp_motion_logits))
                )
            elif args.gan_type == "standard":
                discriminator_prediction_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        amp_replay_logits, torch.zeros_like(amp_replay_logits)
                    )
                    + F.binary_cross_entropy_with_logits(
                        amp_motion_logits, torch.ones_like(amp_motion_logits)
                    )
                )
            else:
                raise ValueError(f"GAN type {args.gan_type} not supported")
            discriminator_loss = discriminator_prediction_loss
    
            if args.discriminator_logit_regularization > 0:
                logit_weights = torch.flatten(discriminator.fc_head.weight)
                logit_weights_squared_sum = torch.sum(torch.square(logit_weights))
                discriminator_loss += args.discriminator_logit_regularization * logit_weights_squared_sum

            if args.discriminator_gradient_penalty > 0:
                amp_motion_gradient = torch.autograd.grad(
                    amp_motion_logits,
                    amp_motion_obs,
                    grad_outputs=torch.ones_like(amp_motion_logits),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                gradient_penalty = torch.sum(torch.square(amp_motion_gradient), dim=-1).mean()
                discriminator_loss += args.discriminator_gradient_penalty * gradient_penalty

        discriminator_optimizer.zero_grad(set_to_none=True)
        scaler.scale(discriminator_loss).backward()
        scaler.unscale_(discriminator_optimizer)

        discriminator_grad_norm = torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(),
            max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
        )
        scaler.step(discriminator_optimizer)
        scaler.update()

        return (
            discriminator_grad_norm.detach(),
            discriminator_prediction_loss.detach(),
            gradient_penalty.detach() if args.discriminator_gradient_penalty > 0 else torch.tensor(0.0, device=device),
            logit_weights_squared_sum.detach() if args.discriminator_logit_regularization > 0 else torch.tensor(0.0, device=device),
            discriminator_loss.detach(),
        )

    normalize_obs = obs_normalizer.forward
    normalize_amp_obs = amp_obs_normalizer.forward
    get_amp_rewards = (
        discriminator.get_lsgan_rewards
        if args.gan_type == "lsgan"
        else discriminator.get_standard_gan_rewards
    )
    if args.compile:
        mode = None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        if args.discriminator_gradient_penalty <= 0:
            update_discriminator = torch.compile(update_discriminator, mode=mode)
        policy = torch.compile(policy, mode=mode)
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward
        get_amp_rewards = torch.compile(get_amp_rewards, mode=mode)
    else:
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    obs, amp_obs = envs.reset_with_amp_obs()
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        amp_obs_normalizer.load_state_dict(torch_checkpoint["amp_obs_normalizer_state"])
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    desc = ""

    with torch.no_grad():
        amp_motion_obs = envs.sample_reference_amp_observations(
            args.num_envs * args.buffer_size // 10
        )

    while global_step < args.total_timesteps:
        if (
            start_time is None
            and global_step >= args.measure_burnin + args.learning_starts
        ):
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        if torch.isnan(actions).any():
            print("NaN detected in actions")
            actions = torch.zeros_like(actions)

        next_obs, online_task_rewards, dones, infos = envs.step(actions.float())

        # Check if any of tensors from environment step are nan, and terminate the current episode
        if torch.isnan(next_obs).any() or torch.isnan(online_task_rewards).any() or torch.isnan(dones).any():
            print("NaN detected in environment step, terminating episode")
            # TODO: Maybe we have to reset each environment individually?
            obs, amp_obs = envs.reset_with_amp_obs()
            continue

        truncations = infos["time_outs"]
        next_amp_obs = infos["amp_obs"]

        if args.reward_normalization:
            online_style_rewards = (
                get_amp_rewards(next_amp_obs) * args.discriminator_reward_scale
            )  # TODO: Why do we need to multiply by args.discriminator_reward_scale?
            online_rewards = (
                args.task_reward_scale * online_task_rewards
                + args.style_reward_scale * online_style_rewards
            )
            update_stats(online_rewards, dones.float())

        # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )

        transition = TensorDict(
            {
                "observations": obs,
                "amp_observations": amp_obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        online_task_rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )

        obs = next_obs
        amp_obs = next_amp_obs

        rb.extend(transition)

        batch_size = args.batch_size // args.num_envs
        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(batch_size)
                data["observations"] = normalize_obs(data["observations"])
                data["amp_observations"] = normalize_amp_obs(data["amp_observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                task_rewards = data["next"]["rewards"]
                # Sample subset of amp_motion_obs to match batch size
                motion_batch_size = data["amp_observations"].shape[0]
                motion_indices = torch.randint(
                    0, amp_motion_obs.shape[0], (motion_batch_size,), device=device
                )
                data["amp_motion_observations"] = normalize_amp_obs(
                    amp_motion_obs[motion_indices]
                )
                # Update discriminator and get style rewards
                (
                    discriminator_grad_norm,
                    discriminator_prediction_loss,
                    gradient_penalty,
                    logit_weights_squared_sum,
                    discriminator_loss,
                ) = update_discriminator(data)
                style_rewards = (
                    get_amp_rewards(data["amp_observations"])
                    * args.discriminator_reward_scale
                )
                if torch.isnan(style_rewards).any():
                    print("NaN detected in style rewards")
                    style_rewards = torch.zeros_like(style_rewards)

                # Weighted sum of task and style rewards
                data["next"]["rewards"] = normalize_reward(
                    args.task_reward_scale * task_rewards
                    + args.style_reward_scale * style_rewards
                )

                critic_grad_norm, qf_loss, qf_max, qf_min = update_main(data)
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        actor_grad_norm, actor_loss = update_pol(data)
                else:
                    if global_step % args.policy_frequency == 0:
                        actor_grad_norm, actor_loss = update_pol(data)

                with torch.no_grad():
                    for param, target_param in zip(
                        qnet.parameters(), qnet_target.parameters()
                    ):
                        target_param.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "actor_loss": actor_loss.mean(),
                        "qf_loss": qf_loss.mean(),
                        "qf_max": qf_max.mean(),
                        "qf_min": qf_min.mean(),
                        "actor_grad_norm": actor_grad_norm.mean(),
                        "critic_grad_norm": critic_grad_norm.mean(),
                        "online_task_rewards": online_task_rewards.mean(),
                        "buffer_task_rewards": task_rewards.mean(),
                        "buffer_style_rewards": style_rewards.mean(),
                        "discriminator_loss": discriminator_loss.mean(),
                        "discriminator_prediction_loss": discriminator_prediction_loss.mean(),
                        "discriminator_gradient_penalty": gradient_penalty.mean(),
                        "discriminator_logit_regularization": logit_weights_squared_sum.mean(),
                        "discriminator_grad_norm": discriminator_grad_norm.mean(),
                    }

                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        print(f"Evaluating at global step {global_step}")
                        eval_avg_return, eval_avg_length = evaluate()
                        # NOTE: Hacky way of evaluating performance, but just works
                        obs = envs.reset()
                        logs["eval_avg_return"] = eval_avg_return
                        logs["eval_avg_length"] = eval_avg_length

                if args.use_wandb:
                    wandb.log(
                        {
                            "speed": speed,
                            "frame": global_step * args.num_envs,
                            **logs,
                        },
                        step=global_step,
                    )

            if (
                args.save_interval > 0
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving model at global step {global_step}")
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    amp_obs_normalizer,
                    args,
                    f"models/{run_name}_{global_step}.pt",
                )

        global_step += 1
        pbar.update(1)

    save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        amp_obs_normalizer,
        args,
        f"models/{run_name}_final.pt",
    )


if __name__ == "__main__":
    main()
