import os
from dataclasses import dataclass
import tyro


@dataclass
class BaseArgs:
    # See IsaacLabArgs for default hyperparameters for IsaacLab
    env_name: str = "Isaac-Humanoid-AMP-Dance-Direct-v0"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    device_rank: int = 0
    """the rank of the device"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    project: str = "FastTD3"
    """the project name"""
    use_wandb: bool = True
    """whether to use wandb"""
    checkpoint_path: str = None
    """the path to the checkpoint file"""
    num_envs: int = 128
    """the number of environments to run in parallel"""
    num_eval_envs: int = 128
    """the number of evaluation environments to run in parallel (only valid for MuJoCo Playground)"""
    total_timesteps: int = 150000
    """total timesteps of the experiments"""
    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""
    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""
    buffer_size: int = 1024 * 50
    """the replay memory buffer size"""
    num_steps: int = 1
    """the number of steps to use for the multi-step return"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.1
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 32768
    """the batch size of sample from the replay memory"""
    policy_noise: float = 0.001
    """the scale of policy noise"""
    std_min: float = 0.001
    """the minimum scale of noise"""
    std_max: float = 0.4
    """the maximum scale of noise"""
    learning_starts: int = 10
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    num_updates: int = 2
    """the number of updates to perform per step"""
    init_scale: float = 0.01
    """the scale of the initial parameters"""
    num_atoms: int = 101
    """the number of atoms"""
    v_min: float = -250.0
    """the minimum value of the support"""
    v_max: float = 250.0
    """the maximum value of the support"""
    critic_hidden_dim: int = 1024
    """the hidden dimension of the critic network"""
    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""
    use_cdq: bool = True
    """whether to use Clipped Double Q-learning"""
    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""
    eval_interval: int = 5000
    """the interval to evaluate the model"""
    compile: bool = True
    """whether to use torch.compile."""
    obs_normalization: bool = True
    """whether to enable observation normalization"""
    reward_normalization: bool = False
    """whether to enable reward normalization"""
    max_grad_norm: float = 0.0
    """the maximum gradient norm"""
    amp: bool = True
    """whether to use amp"""
    amp_dtype: str = "bf16"
    """the dtype of the amp"""

    action_bounds: float = 1.0
    """(IsaacLab only) the bounds of the action space (-action_bounds, action_bounds)"""
    task_embedding_dim: int = 32
    """the dimension of the task embedding"""

    discriminator_reward_scale: float = 2.0
    """the scale of the discriminator reward"""
    task_reward_scale: float = 0.0
    """the scale of the task reward"""
    style_reward_scale: float = 1.0
    """the scale of the style reward"""
    discriminator_learning_rate: float = 3e-4
    """the learning rate of the discriminator"""
    discriminator_hidden_dim: int = 1024
    """the hidden dimension of the discriminator network"""
    gan_type: str = "standard"
    """the type of the gan"""
    lsgan_reward_scale: float = 0.25
    """the scale of the lsgan reward"""
    discriminator_gradient_penalty: float = 0.0
    """the scale of the discriminator gradient penalty"""
    discriminator_logit_regularization: float = 0.0
    """the scale of the discriminator logit regularization"""

    weight_decay: float = 0.1
    """the weight decay of the optimizer"""
    save_interval: int = 5000
    """the interval to save the model"""


def get_args():
    """
    Parse command-line arguments and return the appropriate Args instance based on env_name.
    """
    # First, parse all arguments using the base Args class
    base_args = tyro.cli(BaseArgs)

    # Map environment names to their specific Args classes
    # For tasks not here, default hyperparameters are used
    # See below links for available task list
    # - IsaacLab (https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)
    env_to_args_class = {
        # IsaacLab
        # NOTE: These tasks are not full list of IsaacLab tasks
        "Isaac-Humanoid-AMP-Dance-Direct-v0": IsaacHumanoidAMPDanceDirectArgs,
        "Isaac-Humanoid-AMP-Walk-Direct-v0": IsaacHumanoidAMPWalkDirectArgs,
        "Isaac-Humanoid-AMP-Run-Direct-v0": IsaacHumanoidAMPRunDirectArgs,
    }
    # If the provided env_name has a specific Args class, use it
    if base_args.env_name in env_to_args_class:
        specific_args_class = env_to_args_class[base_args.env_name]
        # Re-parse with the specific class, maintaining any user overrides
        specific_args = tyro.cli(specific_args_class)
        return specific_args

    elif base_args.env_name.startswith("Isaac-"):
        # IsaacLab
        specific_args = tyro.cli(IsaacLabArgs)

    return specific_args


@dataclass
class IsaacLabArgs(BaseArgs):
    v_min: float = -10.0
    v_max: float = 10.0
    buffer_size: int = 1024 * 10
    num_envs: int = 4096
    num_eval_envs: int = 4096
    action_bounds: float = 1.0
    std_max: float = 0.4
    num_atoms: int = 251
    total_timesteps: int = 100000


@dataclass
class IsaacHumanoidAMPDanceDirectArgs(IsaacLabArgs):
    env_name: str = "Isaac-Humanoid-AMP-Dance-Direct-v0"
    num_steps: int = 1
    num_updates: int = 2
    total_timesteps: int = 100000

@dataclass
class IsaacHumanoidAMPWalkDirectArgs(IsaacLabArgs):
    env_name: str = "Isaac-Humanoid-AMP-Walk-Direct-v0"
    num_steps: int = 1
    num_updates: int = 2
    total_timesteps: int = 100000

@dataclass
class IsaacHumanoidAMPRunDirectArgs(IsaacLabArgs):
    env_name: str = "Isaac-Humanoid-AMP-Run-Direct-v0"
    num_steps: int = 1
    num_updates: int = 2
    total_timesteps: int = 100000

@dataclass
class IsaacHumanoidAMPMultiTaskDirectArgs(IsaacLabArgs):
    env_name: str = "Isaac-Humanoid-AMP-MultiTask-Direct-v0"
    num_steps: int = 1
    num_updates: int = 2
    total_timesteps: int = 100000
