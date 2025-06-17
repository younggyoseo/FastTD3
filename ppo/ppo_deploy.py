import torch
import torch.nn as nn
from .ppo import ActorCritic
from fast_td3.fast_td3_utils import EmpiricalNormalization


class Policy(nn.Module):
    """Wrapper around ActorCritic for inference."""

    def __init__(self, n_obs: int, n_act: int, hidden_dim: int):
        super().__init__()
        self.ac = ActorCritic(n_obs, n_act, hidden_dim, device="cpu")
        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device="cpu")
        self.ac.eval()
        self.obs_normalizer.eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.obs_normalizer(obs)
        action, _, _ = self.ac.act(obs, deterministic=True)
        return action


def load_policy(checkpoint_path: str) -> Policy:
    chk = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = chk["args"]
    n_obs = chk["policy_state_dict"]["actor_net.0.weight"].shape[1]
    n_act = chk["policy_state_dict"]["mu.weight"].shape[0]
    policy = Policy(n_obs, n_act, args["hidden_dim"])
    policy.ac.load_state_dict(chk["policy_state_dict"])
    if len(chk.get("obs_normalizer_state", {})) == 0:
        policy.obs_normalizer = nn.Identity()
    else:
        policy.obs_normalizer.load_state_dict(chk["obs_normalizer_state"])
    return policy
