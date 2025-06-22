import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def calculate_network_norms(network: nn.Module, prefix: str = ""):
    """Return norm metrics of network parameters."""
    metrics = {}
    total_norm = 0.0
    param_count = 0
    for name, param in network.named_parameters():
        if param.requires_grad:
            param_norm = param.data.norm(2).item()
            metrics[f"{prefix}_{name}_norm"] = param_norm
            total_norm += param_norm ** 2
            param_count += param.numel()
    total_norm = total_norm ** 0.5
    metrics[f"{prefix}_total_param_norm"] = total_norm
    metrics[f"{prefix}_param_count"] = param_count
    return metrics


class ActorCritic(nn.Module):
    """Simple actor-critic network used for PPO."""

    def __init__(self, n_obs: int, n_act: int, hidden_dim: int, device=None):
        super().__init__()
        self.device = device

        self.actor_net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
        )
        self.mu = nn.Linear(hidden_dim//4, n_act, device=device)
        self.log_std = nn.Parameter(torch.zeros(n_act, device=device))

        self.critic_net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1, device=device),
        )

    def get_dist(self, obs: torch.Tensor) -> Normal:
        x = self.actor_net(obs)
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.get_dist(obs)
        if deterministic:
            raw_action = dist.mean
        else:
            raw_action = dist.rsample()
        
        # Apply tanh transformation
        action = torch.tanh(raw_action)
        
        # Compute log probability of the raw action (before tanh)
        # and adjust for the tanh transformation
        log_prob = dist.log_prob(raw_action).sum(-1)
        # Tanh Jacobian correction: log(1 - tanh^2(x))
        log_prob = log_prob - (2 * (torch.log(torch.tensor(2.0)) - raw_action - torch.nn.functional.softplus(-2 * raw_action))).sum(-1)
        
        return action, log_prob, self.value(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_net(obs).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.get_dist(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        values = self.value(obs)
        return log_prob, entropy, values
