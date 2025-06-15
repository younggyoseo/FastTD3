from __future__ import annotations

import os
import sys
import yaml
from typing import Optional

import torch

import isaacgym
import isaacgymenvs


class IsaacGymEnv:
    def __init__(
        self,
        task_name: str,
        device_id: int,
        num_envs: int,
        seed: int,
        action_bounds: Optional[float] = None,
    ):
        self.num_envs = num_envs
        self.action_bounds = action_bounds

        self.env = isaacgymenvs.make(
            task=task_name,
            num_envs=num_envs,
            sim_device=f"cuda:{device_id}",
            rl_device=f"cuda:{device_id}",
            seed=seed,
            headless=True,
        )

        self.asymmetric_obs = self.env.asymmetric_obs
        self.num_obs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.num_privileged_obs = self.env.state_space.shape[0]
        self.max_episode_steps = self.env.max_episode_length

    def reset(self) -> torch.Tensor:
        """Reset the environment."""
        obs_dict = self.env.reset()
        return obs_dict["obs"]

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict = self.env.reset()
        obs = obs_dict["obs"]
        states = obs_dict["states"]
        critic_obs = torch.cat([obs, states], -1)
        return obs, critic_obs

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment."""
        assert isinstance(actions, torch.Tensor)
        if self.action_bounds is not None:
            actions = torch.clamp(actions, -1.0, 1.0) * self.action_bounds

        obs_dict, rew, dones, infos = self.env.step(actions)
        truncations = infos["time_outs"]
        critic_obs = torch.cat([obs_dict["obs"], obs_dict["states"]], dim=-1)
        info_ret = {"time_outs": truncations, "observations": {"critic": critic_obs}}
        # NOTE: There's really no way to get the raw observations from IsaacGym
        # We just use the 'reset_obs' as next_obs, unfortunately.
        info_ret["observations"]["raw"] = {
            "obs": obs_dict["obs"],
            "critic_obs": critic_obs,
        }
        return obs_dict["obs"], rew, dones, info_ret

    def render(self):
        raise NotImplementedError(
            "We don't support rendering for IsaacLab environments"
        )


ISAAC_GYM_TASK_NAMES = [
    "AllegroHand",
    "AllegroHandDextremeADR",
    "AllegroHandDextremeManualDR",
    "AllegroKukaLSTM",
    "AllegroKukaTwoArmsLSTM",
    "Ant",
    "Anymal",
    "AnymalTerrain",
    "BallBalance",
    "Cartpole",
    "FrankaCabinet",
    "Humanoid",
    "Ingenuity Quadcopter",
    "ShadowHand",
    "ShadowHandOpenAI_FF",
    "ShadowHandOpenAI_LSTM",
    "Trifinger",
]
