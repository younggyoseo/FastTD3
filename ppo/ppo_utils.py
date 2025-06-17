import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutBuffer:
    """Buffer for storing environment rollouts used in PPO."""

    def __init__(self, buffer_size: int, observation_dim: int, action_dim: int, device=None):
        self.obs = torch.zeros(buffer_size, observation_dim, device=device)
        self.actions = torch.zeros(buffer_size, action_dim, device=device)
        self.logprobs = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)
        self.ptr = 0
        self.max_size = buffer_size
        self.device = device

    def add(self, obs, action, logprob, reward, done, value):
        if self.ptr >= self.max_size:
            return
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantage(self, last_value, gamma: float, gae_lambda: float):
        prev_adv = 0
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step + 1]
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            prev_adv = delta + gamma * gae_lambda * next_non_terminal * prev_adv
            self.advantages[step] = prev_adv
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size):
        sampler = BatchSampler(SubsetRandomSampler(range(self.ptr)), batch_size, drop_last=True)
        for indices in sampler:
            yield (
                self.obs[indices],
                self.actions[indices],
                self.logprobs[indices],
                self.returns[indices],
                self.advantages[indices],
            )

    def clear(self):
        self.ptr = 0
