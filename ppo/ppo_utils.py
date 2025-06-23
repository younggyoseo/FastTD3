import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os

class RolloutBuffer:
    """
    Stores one rollout of length `T` collected from `N` parallel envs.

    Data layout is   (T, N, ·)   which makes the GAE recursion easy.
    """

    def __init__(self,
                 rollout_length: int,
                 num_envs: int,
                 obs_dim: int,
                 act_dim: int,
                 device=None):
        self.T, self.N = rollout_length, num_envs
        shape = (rollout_length, num_envs)

        self.obs        = torch.zeros(*shape,  obs_dim,  device=device)
        self.actions    = torch.zeros(*shape,  act_dim,  device=device)
        self.logprobs   = torch.zeros(*shape,            device=device)
        self.rewards    = torch.zeros(*shape,            device=device)
        self.dones      = torch.zeros(*shape,            device=device)
        self.values     = torch.zeros(*shape,            device=device)
        self.advantages = torch.zeros(*shape,            device=device)
        self.returns    = torch.zeros(*shape,            device=device)

        self.ptr_step = 0          # current time index

    # ------------------------------------------------------------------
    # storing one transition for *every* env at the current time step
    # ------------------------------------------------------------------
    def add(self, obs, action, logp, reward, done, value):
        """
        Args are tensors of shape (N, ·) coming straight from the vector env.
        """
        if self.ptr_step >= self.T:
            return
        self.obs[self.ptr_step]      = obs
        self.actions[self.ptr_step]  = action
        self.logprobs[self.ptr_step] = logp
        self.rewards[self.ptr_step]  = reward
        self.dones[self.ptr_step]    = done
        self.values[self.ptr_step]   = value
        self.ptr_step += 1

    # ------------------------------------------------------------------
    # GAE with a *vector* bootstrap `last_values` (shape N)
    # ------------------------------------------------------------------
    def compute_returns_and_advantage(self,
                                      last_values,      # tensor (N,)
                                      gamma: float,
                                      gae_lambda: float,
                                      num_envs: int):
        next_values = last_values              # V_{T}
        next_adv    = torch.zeros(num_envs, device=last_values.device)

        for t in reversed(range(self.ptr_step)):
            # mask: 0 if episode ended at t+1, else 1
            next_non_terminal = 1.0 - self.dones[t]

            delta = (self.rewards[t] +
                     gamma * next_values * next_non_terminal -
                     self.values[t])

            next_adv = (delta +
                        gamma * gae_lambda * next_non_terminal * next_adv)

            self.advantages[t] = next_adv
            next_values        = self.values[t]

        self.returns[:self.ptr_step] = (
            self.advantages[:self.ptr_step] + self.values[:self.ptr_step]
        )

        # Advantage normalisation (time × env)
        flat_adv = self.advantages[:self.ptr_step].reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        self.advantages[:self.ptr_step] = flat_adv.view_as(
            self.advantages[:self.ptr_step]
        )

    # ------------------------------------------------------------------
    # batching — simply flatten (T, N) → (T·N)
    # ------------------------------------------------------------------
    def get_batches(self, batch_size):
        total = self.ptr_step * self.N
        sampler = BatchSampler(
            SubsetRandomSampler(range(total)), batch_size, drop_last=True
        )
        # flatten the first two axes for training
        obs        = self.obs[:self.ptr_step].reshape(total, -1)
        actions    = self.actions[:self.ptr_step].reshape(total, -1)
        logprobs   = self.logprobs[:self.ptr_step].reshape(total)
        returns    = self.returns[:self.ptr_step].reshape(total)
        advantages = self.advantages[:self.ptr_step].reshape(total)
        for idx in sampler:
            yield (obs[idx], actions[idx], logprobs[idx],
                   returns[idx], advantages[idx])

    def clear(self):
        self.ptr_step = 0


def cpu_state(sd):
    """Detach and move tensors to CPU for serialization."""
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def save_ppo_params(global_step, policy, obs_normalizer, args, save_path):
    """Save PPO policy parameters and configuration."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "policy_state_dict": cpu_state(policy.state_dict()),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "args": vars(args),
        "global_step": global_step,
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved parameters and configuration to {save_path}")

