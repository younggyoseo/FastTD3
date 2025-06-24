from __future__ import annotations

import torch
from omegaconf import OmegaConf

import isaacgym
import isaacgymenvs


class MTBenchEnv:
    def __init__(
        self,
        task_name: str,
        device_id: int,
        num_envs: int,
        seed: int,
    ):
        # NOTE: Currently, we only support Meta-World-v2 MT-10/MT-50 in MTBench
        task_config = MTBENCH_MW2_CONFIG.copy()
        if task_name == "meta-world-v2-mt10":
            # MT-10 Setup
            assert num_envs == 4096, "MT-10 only supports 4096 environments (for now)"
            self.num_tasks = 10
            task_config["env"]["tasks"] = [4, 16, 17, 18, 28, 31, 38, 40, 48, 49]
            task_config["env"]["taskEnvCount"] = [410] * 6 + [409] * 4
        elif task_name == "meta-world-v2-mt50":
            # MT-50 Setup
            self.num_tasks = 50
            assert num_envs == 8192, "MT-50 only supports 8192 environments (for now)"
            task_config["env"]["tasks"] = list(range(50))
            task_config["env"]["taskEnvCount"] = [164] * 42 + [163] * 8  # 6888 + 1304
        else:
            raise ValueError(f"Unsupported task name: {task_name}")
        task_config["env"]["numEnvs"] = num_envs
        task_config["env"]["numObservations"] = 39 + self.num_tasks
        task_config["env"]["seed"] = seed

        # Convert dictionary to OmegaConf object
        env_cfg = {"task": task_config}
        env_cfg = OmegaConf.create(env_cfg)

        self.env = isaacgymenvs.make(
            task=env_cfg.task.name,
            num_envs=num_envs,
            sim_device=f"cuda:{device_id}",
            rl_device=f"cuda:{device_id}",
            seed=seed,
            headless=True,
            cfg=env_cfg,
        )

        self.num_envs = num_envs
        self.asymmetric_obs = False
        self.num_obs = self.env.observation_space.shape[0]
        assert (
            self.num_obs == 39 + self.num_tasks
        ), "MTBench observation space is 39 + num_tasks (one-hot vector)"
        self.num_privileged_obs = 0
        self.num_actions = self.env.action_space.shape[0]
        self.max_episode_steps = self.env.max_episode_length

    def reset(self) -> torch.Tensor:
        """Reset the environment."""
        # TODO: Check if we need no_grad and detach here
        with torch.no_grad():  # do we need this?
            self.env.reset_idx(torch.arange(self.num_envs, device=self.env.device))
            self.env.cumulatives["rewards"][:] = 0
            self.env.cumulatives["success"][:] = 0
            obs_dict = self.env.reset()
            return obs_dict["obs"].detach()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment."""
        assert isinstance(actions, torch.Tensor)

        # TODO: Check if we need no_grad and detach here
        with torch.no_grad():
            obs_dict, rew, dones, infos = self.env.step(actions.detach())
            truncations = infos["time_outs"]
            info_ret = {"time_outs": truncations.detach()}
            if "episode" in infos:
                info_ret["episode"] = infos["episode"]
            # NOTE: There's really no way to get the raw observations from IsaacGym
            # We just use the 'reset_obs' as next_obs, unfortunately.
            info_ret["observations"] = {"raw": {"obs": obs_dict["obs"].detach()}}
            return obs_dict["obs"].detach(), rew.detach(), dones.detach(), info_ret

    def render(self):
        raise NotImplementedError(
            "We don't support rendering for IsaacLab environments"
        )


MTBENCH_MW2_CONFIG = {
    "name": "meta-world-v2",
    "physics_engine": "physx",
    "env": {
        "numEnvs": 1,
        "envSpacing": 1.5,
        "episodeLength": 150,
        "enableDebugVis": False,
        "clipObservations": 5.0,
        "clipActions": 1.0,
        "aggregateMode": 3,
        "actionScale": 0.01,
        "resetNoise": 0.15,
        "tasks": [0],
        "taskEnvCount": [4096],
        "init_at_random_progress": True,
        "exemptedInitAtRandomProgressTasks": [],
        "taskEmbedding": True,
        "taskEmbeddingType": "one_hot",
        "seed": 42,
        "cameraRenderingInterval": 5000,
        "cameraWidth": 1024,
        "cameraHeight": 1024,
        "sparse_reward": False,
        "termination_on_success": False,
        "reward_scale": 1.0,
        "fixed": False,
        "numObservations": None,
        "numActions": 4,
    },
    "enableCameraSensors": False,
    "sim": {
        "dt": 0.01667,
        "substeps": 2,
        "up_axis": "z",
        "use_gpu_pipeline": True,
        "gravity": [0.0, 0.0, -9.81],
        "physx": {
            "num_threads": 4,
            "solver_type": 1,
            "use_gpu": True,
            "num_position_iterations": 8,
            "num_velocity_iterations": 1,
            "contact_offset": 0.005,
            "rest_offset": 0.0,
            "bounce_threshold_velocity": 0.2,
            "max_depenetration_velocity": 1000.0,
            "default_buffer_size_multiplier": 10.0,
            "max_gpu_contact_pairs": 1048576,
            "num_subscenes": 4,
            "contact_collection": 0,
        },
    },
    "task": {"randomize": False},
}
