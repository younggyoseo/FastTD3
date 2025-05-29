# Guide for Sim2Real Training & Deployment

This guide provides guide to run sim-to-real experiments using FastTD3 and BoosterGym.

**⚠️ Warning:** Deploying RL policies to real hardware can be sometimes very dangerous. Please make sure that you understand everything, check your policies work well in simulation, set every robot configuration correct (e.g., damping, stiffness, torque limits, etc), and follow proper safety protocols. **Simply copy-pasting commands in this README is not safe**.

## ⚙️ Prerequisites

Install dependencies for Playground experiments (see `README.md`)

Then, install `fast_td3` package with `pip install -e .` so you can import its classes in BoosterGym (see `fast_td3_deploy.py`).

**⚠️ Note:** Our sim-to-real experiments depend on our customized MuJoCo Playground that supports `T1LowDimJoystick` tasks for 12-DOF T1 control instead of 23-DOF T1 control in `T1Joystick` tasks.

## 🚀 Training in simulation

Users can train deployable policies for Booster T1 with FastTD3 using the below script:

```bash
python fast_td3/train.py --env_name T1LowDimJoystickRoughTerrain --exp_name FastTD3 --use_domain_randomization --use_push_randomization --total_timesteps 1000000 --render_interval 0 --seed 2
```

**⚠️ Note:** There is no 'guaranteed' number of training steps that can ensure safe real-world deployment. Usually, the gait becomes more stable with longer training. Please check the quality of gaits via sim-to-sim transfer, and fine-tune the policy to fix the issues. Use the checkpoints in `models` directory for sim-to-sim or sim-to-real transfer.

**⚠️ Note:** We set `render_interval` to 0 to avoid dumping a lot of videos into wandb. Make sure to set it to non-zero values if you want to render videos during training.



### (Optional) 2-Stage Training

For faster convergence, users can consider introducing curriculum to the training -- so that the robot first learns to walk in a flat terrain without push perturbations. For this, train policies with the below script:

```bash
STAGE1_STEPS = 100000
STAGE2_STEPS = 300000  # Effective steps: 300000 - 200000 = 100000
SEED = 2
CHECKPOINT_PATH = T1LowDimJoystickFlatTerrain__FastTD3-Stage1__${SEED}_final.pt

conda activate fasttd3_playground

# Stage 1 training
python fast_td3/train.py --env_name T1LowDimJoystickFlatTerrain --exp_name FastTD3-Stage1 --use_domain_randomization --no_use_push_randomization --total_timesteps ${STAGE1_STEPS} --render_interval 0 --seed ${SEED}

# Stage 2 training
python fast_td3/train.py --env_name T1LowDimJoystickRoughTerrain --exp_name FastTD3-Stage2 --use_domain_randomization --use_push_randomization --total_timesteps ${STAGE2_STEPS} --render_interval 0 --checkpoint_path ${CHECKPOINT_PATH} --seed ${SEED}
```

Again, 100K and 200K steps do not guarantee safe real-world deployment. Please check the quality of gaits via sim-to-sim transfer, and fine-tune the policy to fix the issues. Use the final checkpoint (`models/T1LowDimJoystickRoughTerrain__FastTD3-Stage2__${SEED}_final.pt`) for sim-to-sim or sim-to-real transfer.

## 🛝 Deployment with BoosterGym

We use the customized version of [BoosterGym](https://github.com/BoosterRobotics/booster_gym) for deployment with FastTD3.

First, clone our fork of BoosterGym.

```bash
git clone https://github.com/carlosferrazza/booster_gym.git
```

Then, follow the [guide](https://github.com/carlosferrazza/booster_gym) to install dependencies for BoosterGym.

### Sim-to-Sim Transfer

You can check whether the trained policy transfers to non-MJX version of MuJoCo.
Use the following commands in a machine that supports rendering to test sim-to-sim transfer:

```bash
cd <YOUR_WORKSPACE>/booster_gym

# Activate your BoosterGym virtual environemnt

# Launch MuJoCo simulation
python play_mujoco.py --task=T1 --checkpoint=<CHECKPOINT_PATH>
mjpython play_mujoco.py --task=T1 --checkpoint=<CHECKPOINT_PATH>  # for Mac
```


 
### Sim-to-Real Transfer

First, prepare a JIT-scripted checkpoint

```python
# Python snippets for JIT-scripting checkpoints
import torch
from fast_td3 import load_policy
policy = load_policy(<CHECKPOINT_PATH>)
scripted_policy = torch.jit.script(policy)
scripted_policy.save(<JIT_CHECKPOINT_PATH>)
```

Then, deploy this JIT-scripted checkpoint by following the guide on [Booster T1 Deployment](https://github.com/carlosferrazza/booster_gym/tree/main/deploy).


**⚠️ Warning:** Please double-check every value in robot configuration (`booster_gym/deploy/configs/T1.yaml`) is correctly set! If values for position control such as `damping` or `stiffness` are set differently, your robot may perform dangerous behaviors. 

**⚠️ Warning:** You may want to use different configuration (e.g., `damping` and `stiffness`, etc) for your own experiments. Just make sure to thoroughly test it in simulation and make sure to set the values correctly.

---

🚀 That's it! Hope everything went smoothly, and be aware of your safety.