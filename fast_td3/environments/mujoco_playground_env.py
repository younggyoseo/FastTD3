import jax
import mujoco
from mujoco_playground import registry, wrapper_torch

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


class PlaygroundEvalEnvWrapper:
    def __init__(self, eval_env, max_episode_steps, env_name, num_eval_envs, seed):
        """
        用于评估/渲染环境的包装器。
        请注意，这与使用 RSLRLBraxWrapper 包装的训练环境不同。
        #! RSLRLBraxWrapper 不支持在重置每个环境之前保存最终观测。
        """
        self.env = eval_env
        self.env_name = env_name
        self.num_envs = num_eval_envs

        ##NOTE - 函数并行化
        """
        jax.vmap(self.env.reset): 这行代码创建了一个全新的函数。这个新函数可以接收一批输入（在这里是一批随机种子 key_reset），然后在内部并行地为这批输入中的每一个都调用一次 self.env.reset。它等价于一个用 C++ 或 CUDA 实现的、高度优化的 for 循环，但性能要高得多。
        jax.jit(...) - 即时编译优化,将一个 JAX 函数（包括我们刚刚用 vmap 创建的新函数）编译成针对特定硬件（如 GPU 或 TPU）的高度优化的机器码（XLA）。完全绕过了 Python 解释器的开销。
        """
        self.jit_reset = jax.jit(jax.vmap(self.env.reset))
        self.jit_step = jax.jit(jax.vmap(self.env.step))

        """
        unwrapped.env 获取最底层的、未经任何包装的原始环境对象。
        如果是非对称环境，这个属性通常会被实现为一个字典，例如：{'state': 48, 'privileged_info': 12}。
        如果是标准环境，这个属性就是一个整数，例如：48
        """
        if isinstance(
            self.env.unwrapped.observation_size, dict
        ):  # 如果是字典，则为非对称环境
            self.asymmetric_obs = True
        else:
            self.asymmetric_obs = False

        ##NOTE - jax 随机数生成器
        """
        JAX: 采取了纯函数式的方法。没有隐藏状态。所有的随机性都必须来自于一个你明确管理的**“密钥”（key）。要生成随机数，你必须将一个 key 作为参数传给随机函数。为了生成新的、不同的随机数，你必须从旧的 key 中分裂（split）**出新的 key。
        """
        self.key = jax.random.PRNGKey(
            seed
        )  # 创建一个初始的、顶级的伪随机数生成器（PRNG）密钥。
        self.key_reset = jax.random.split(
            self.key, num_eval_envs
        )  # 从顶级的 key 中分裂出多个独立的子密钥，每个并行环境一个。
        self.max_episode_steps = max_episode_steps

    def reset(self):
        self.state = self.jit_reset(self.key_reset)
        if self.asymmetric_obs:
            obs = wrapper_torch._jax_to_torch(self.state.obs["state"])
        else:
            obs = wrapper_torch._jax_to_torch(self.state.obs)
        return obs

    def step(self, actions):
        self.state = self.jit_step(self.state, wrapper_torch._torch_to_jax(actions))
        if self.asymmetric_obs:
            next_obs = wrapper_torch._jax_to_torch(self.state.obs["state"])
        else:
            next_obs = wrapper_torch._jax_to_torch(self.state.obs)
        rewards = wrapper_torch._jax_to_torch(self.state.reward)
        dones = wrapper_torch._jax_to_torch(self.state.done)
        return next_obs, rewards, dones, None

    def render_trajectory(self, trajectory):
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

        frames = self.env.render(
            trajectory,
            camera="track" if "Joystick" in self.env_name else None,
            height=480,
            width=640,
            scene_option=scene_option,
        )
        return frames


def make_env(
    env_name,
    seed,
    num_envs,
    num_eval_envs,
    device_rank,
    use_tuned_reward=False,
    use_domain_randomization=False,
    use_push_randomization=False,
):
    # 制作训练环境
    train_env_cfg = registry.get_default_config(env_name)
    is_humanoid_task = env_name in [
        "G1JoystickRoughTerrain",
        "G1JoystickFlatTerrain",
        "T1JoystickRoughTerrain",
        "T1JoystickFlatTerrain",
    ]

    ##NOTE - 调整人型奖励配置
    if use_tuned_reward and is_humanoid_task:
        # 注意：调整了G1的奖励，用于生成论文中的图7。
        # 它在T1上也能合理地工作。
        # 然而，请参阅`sim2real.md`了解使用Booster T1进行模拟到现实的强化学习。
        train_env_cfg.reward_config.scales.energy = -5e-5
        train_env_cfg.reward_config.scales.action_rate = -1e-1
        train_env_cfg.reward_config.scales.torques = -1e-3
        train_env_cfg.reward_config.scales.pose = -1.0
        train_env_cfg.reward_config.scales.tracking_ang_vel = 1.25
        train_env_cfg.reward_config.scales.tracking_lin_vel = 1.25
        train_env_cfg.reward_config.scales.feet_phase = 1.0
        train_env_cfg.reward_config.scales.ang_vel_xy = -0.3
        train_env_cfg.reward_config.scales.orientation = -5.0

    if is_humanoid_task and not use_push_randomization:
        train_env_cfg.push_config.enable = False
        train_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    randomizer = (
        registry.get_domain_randomizer(env_name) if use_domain_randomization else None
    )
    raw_env = registry.load(env_name, config=train_env_cfg)
    train_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env,
        num_envs,
        seed,
        train_env_cfg.episode_length,
        train_env_cfg.action_repeat,
        randomization_fn=randomizer,
        device_rank=device_rank,
    )

    # 制作评估环境
    eval_env_cfg = registry.get_default_config(env_name)
    if is_humanoid_task and not use_push_randomization:
        eval_env_cfg.push_config.enable = False
        eval_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    eval_env = registry.load(env_name, config=eval_env_cfg)
    eval_env = PlaygroundEvalEnvWrapper(
        eval_env, eval_env_cfg.episode_length, env_name, num_eval_envs, seed
    )

    render_env_cfg = registry.get_default_config(env_name)
    if is_humanoid_task and not use_push_randomization:
        render_env_cfg.push_config.enable = False
        render_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    render_env = registry.load(env_name, config=render_env_cfg)
    render_env = PlaygroundEvalEnvWrapper(
        render_env, render_env_cfg.episode_length, env_name, 1, seed
    )

    return train_env, eval_env, render_env
