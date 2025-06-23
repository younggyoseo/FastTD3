from typing import Optional

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class IsaacLabAMPEnv:
    """
    Wrapper for IsaacLab AMP tasks
    Currently breaks the API of main train.py logic
    """

    def __init__(
        self,
        task_name: str,
        device: str,
        num_envs: int,
        seed: int,
        action_bounds: Optional[float] = None,
    ):
        env_cfg = parse_env_cfg(
            task_name,
            device=device,
            num_envs=num_envs,
        )
        env_cfg.seed = seed
        self.seed = seed
        self.envs = gym.make(task_name, cfg=env_cfg, render_mode=None)

        self.num_envs = self.envs.unwrapped.num_envs
        self.max_episode_steps = self.envs.unwrapped.max_episode_length
        self.action_bounds = action_bounds
        self.num_obs = self.envs.unwrapped.single_observation_space["policy"].shape[0]
        self.num_amp_obs = self.envs.amp_observation_space.shape[0]
        self.num_actions = self.envs.unwrapped.single_action_space.shape[0]

        # IsaacLab AMP tasks' observation space is not asymmetric
        self.asymmetric_obs = False
        self.num_privileged_obs = 0

    def reset(self) -> torch.Tensor:
        obs_dict, _ = self.envs.reset()
        return obs_dict["policy"]

    def reset_with_amp_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict, info_dict = self.envs.reset()
        return obs_dict["policy"], info_dict["amp_obs"]

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.action_bounds is not None:
            actions = torch.clamp(actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminations, truncations, infos = self.envs.step(actions)
        dones = (terminations | truncations).to(dtype=torch.long)
        obs = obs_dict["policy"]
        amp_obs = infos["amp_obs"]
        info_ret = {
            "time_outs": truncations,
            "amp_obs": amp_obs,
            "observations": {},
        }
        # NOTE: There's really no way to get the raw observations from IsaacLab
        # We just use the 'reset_obs' as next_obs, unfortunately.
        # See https://github.com/isaac-sim/IsaacLab/issues/1362
        info_ret["observations"]["raw"] = {"obs": obs}
        return obs, rew, dones, info_ret

    def sample_reference_amp_observations(self, batch_size: int) -> torch.Tensor:
        return self.envs.collect_reference_motions(batch_size)
