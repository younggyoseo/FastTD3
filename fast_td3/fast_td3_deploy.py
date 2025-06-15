import math

import torch
import torch.nn as nn
from .fast_td3_utils import EmpiricalNormalization
from .fast_td3 import Actor
from .fast_td3_simbav2 import Actor as ActorSimbaV2


class Policy(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        args: dict,
        agent: str = "fasttd3",
    ):
        super().__init__()

        self.args = args

        num_envs = args["num_envs"]
        init_scale = args["init_scale"]
        actor_hidden_dim = args["actor_hidden_dim"]

        actor_kwargs = dict(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            device="cpu",
            init_scale=init_scale,
            hidden_dim=actor_hidden_dim,
        )

        if agent == "fasttd3":
            actor_cls = Actor
        elif agent == "fasttd3_simbav2":
            actor_cls = ActorSimbaV2

            actor_num_blocks = args["actor_num_blocks"]
            actor_kwargs.pop("init_scale")
            actor_kwargs.update(
                {
                    "scaler_init": math.sqrt(2.0 / actor_hidden_dim),
                    "scaler_scale": math.sqrt(2.0 / actor_hidden_dim),
                    "alpha_init": 1.0 / (actor_num_blocks + 1),
                    "alpha_scale": 1.0 / math.sqrt(actor_hidden_dim),
                    "expansion": 4,
                    "c_shift": 3.0,
                    "num_blocks": actor_num_blocks,
                }
            )
        else:
            raise ValueError(f"Agent {agent} not supported")

        self.actor = actor_cls(
            **actor_kwargs,
        )
        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device="cpu")

        self.actor.eval()
        self.obs_normalizer.eval()

    @torch.no_grad
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.obs_normalizer(obs)
        actions = self.actor(norm_obs)
        return actions

    @torch.no_grad
    def act(self, obs: torch.Tensor) -> torch.distributions.Normal:
        actions = self.forward(obs)
        return torch.distributions.Normal(actions, torch.ones_like(actions) * 1e-8)


def load_policy(checkpoint_path):
    torch_checkpoint = torch.load(
        f"{checkpoint_path}", map_location="cpu", weights_only=False
    )
    args = torch_checkpoint["args"]

    agent = args.get("agent", "fasttd3")
    if agent == "fasttd3":
        n_obs = torch_checkpoint["actor_state_dict"]["net.0.weight"].shape[-1]
        n_act = torch_checkpoint["actor_state_dict"]["fc_mu.0.weight"].shape[0]
    elif agent == "fasttd3_simbav2":
        # TODO: Too hard-coded, maybe save n_obs and n_act in the checkpoint?
        n_obs = (
            torch_checkpoint["actor_state_dict"]["embedder.w.w.weight"].shape[-1] - 1
        )
        n_act = torch_checkpoint["actor_state_dict"]["predictor.mean_bias"].shape[0]
    else:
        raise ValueError(f"Agent {agent} not supported")

    policy = Policy(
        n_obs=n_obs,
        n_act=n_act,
        args=args,
        agent=agent,
    )
    policy.actor.load_state_dict(torch_checkpoint["actor_state_dict"])

    if len(torch_checkpoint["obs_normalizer_state"]) == 0:
        policy.obs_normalizer = nn.Identity()
    else:
        policy.obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])

    return policy
