# FastTD3 - Simple and Fast RL for Humanoid Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2505.22642-b31b1b.svg)](https://arxiv.org/abs/2505.22642)


FastTD3 is a high-performance variant of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm, optimized for complex humanoid control tasks. FastTD3 can solve various humanoid control tasks with dexterous hands from HumanoidBench in just a few hours of training. Furthermore, FastTD3 achieves similar or better wall-time-efficiency to PPO in high-dimensional control tasks from popular simulations such as IsaacLab and MuJoCo Playground.

For more information, please see our [project webpage](https://younggyo.me/fast_td3)


## ❗ Updates

- **[Jul/07/2025]** Added support for multi-GPU training! See [Multi-GPU Training](#multi-gpu-training) section for details. 

- **[Jul/02/2025]** Optimized codebase to speed up training around 10-30% when using a single RTX 4090 GPU.

- **[Jun/20/2025]** Added support for [MTBench](https://github.com/Viraj-Joshi/MTBench) with the help of [Viraj Joshi](https://viraj-joshi.github.io/).

- **[Jun/15/2025]** Added support for FastTD3 + [SimbaV2](https://dojeon-ai.github.io/SimbaV2/)! It's faster to train, and often achieves better asymptotic performance. We recommend using FastTD3 + SimbaV2 for most cases.

- **[Jun/06/2025]** Thanks to [Antonin Raffin](https://araffin.github.io/) ([@araffin](https://github.com/araffin)), we fixed the issues when using `n_steps` > 1, which stabilizes training with n-step return quite a lot!

- **[Jun/01/2025]** Updated the figures in the technical report to report deterministic evaluation for IsaacLab tasks.


## ✨ Features

FastTD3 offers researchers a significant speedup in training complex humanoid agents.

- Ready-to-go codebase with detailed instructions and pre-configured hyperparameters for each task
- Support popular benchmarks: HumanoidBench, MuJoCo Playground, and IsaacLab
- User-friendly features that can accelerate your research, such as rendering rollouts, torch optimizations (AMP and compile), and saving and loading checkpoints

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:
- Conda (for environment management)
- Git LFS (Large File Storage) -- For IsaacLab
- CMake -- For IsaacLab

And the following system packages:
```bash
sudo apt install libglfw3 libgl1-mesa-glx libosmesa6 git-lfs cmake
```

## 📖 Installation

This project requires different Conda environments for different sets of experiments.

### Common Setup
First, ensure the common dependencies are installed as mentioned in the [Prerequisites](#prerequisites) section.

### Environment for HumanoidBench

```bash
conda create -n fasttd3_hb -y python=3.10
conda activate fasttd3_hb
pip install --editable git+https://github.com/carlosferrazza/humanoid-bench.git#egg=humanoid-bench
pip install -r requirements/requirements.txt
```

### Environment for MuJoCo Playground
```bash
conda create -n fasttd3_playground -y python=3.10
conda activate fasttd3_playground
pip install -r requirements/requirements_playground.txt
```

**⚠️ Note:** Our `requirements_playground.txt` specifies `Jax==0.4.35`, which we found to be stable for latest GPUs in certain tasks such as `LeapCubeReorient` or `LeapCubeRotateZAxis`

**⚠️ Note:** Current FastTD3 codebase uses customized MuJoCo Playground that supports saving last observations into info dictionary. We will work on incorporating this change into official repository hopefully soon.

### Environment for IsaacLab
```bash
conda create -n fasttd3_isaaclab -y python=3.10
conda activate fasttd3_isaaclab

# Install IsaacLab (refer to official documentation for the latest steps)
# Official Quickstart: https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
cd ..

# Install project-specific requirements
pip install -r requirements/requirements.txt
```

### Environment for MTBench
MTBench does not support humanoid experiments, but is a useful multi-task benchmark with massive parallel simulation. This could be useful for users who want to use FastTD3 for their multi-task experiments.

```bash
conda create -n fasttd3_mtbench -y python=3.8  # Note python version
conda activate fasttd3_mtbench

# Install IsaacGym -- recommend to follow instructions in https://github.com/BoosterRobotics/booster_gym
...

# Install MTBench
git clone https://github.com/Viraj-Joshi/MTBench.git
cd MTbench
pip install -e .
pip install skrl
cd ..

# Install project-specific requirements
pip install -r requirements/requirements_isaacgym.txt
```

### (Optional) Accelerate headless GPU rendering in cloud instances

In some cloud VM images the NVIDIA kernel driver is present but the user-space OpenGL/EGL/Vulkan libraries aren't, so MuJoCo falls back to CPU renderer. You can install just the NVIDIA user-space libraries (and skip rebuilding the kernel module) with:

```bash
sudo apt install -y kmod
sudo sh NVIDIA-Linux-x86_64-<your_driver_version>.run -s --no-kernel-module --ui=none --no-questions
```

As a rule-of-thumb, if you're running experiments and rendering is taking longer than 5 seconds, it is very likely that GPU renderer is not used.

## 🚀 Running Experiments

Activate the appropriate Conda environment before running experiments.

Please see `fast_td3/hyperparams.py` for information regarding hyperparameters!

### HumanoidBench Experiments
```bash
conda activate fasttd3_hb
# FastTD3
python fast_td3/train.py \
    --env_name h1hand-hurdle-v0 \
    --exp_name FastTD3 \
    --render_interval 5000 \
    --seed 1
# FastTD3 + SimbaV2
python fast_td3/train.py \
    --env_name h1hand-hurdle-v0 \
    --exp_name FastTD3 \
    --render_interval 5000 \
    --agent fasttd3_simbav2 \
    --batch_size 8192 \
    --critic_learning_rate_end 3e-5 \
    --actor_learning_rate_end 3e-5 \
    --weight_decay 0.0 \
    --critic_hidden_dim 512 \
    --critic_num_blocks 2 \
    --actor_hidden_dim 256 \
    --actor_num_blocks 1 \
    --seed 1
```

### MuJoCo Playground Experiments
```bash
conda activate fasttd3_playground
# FastTD3
python fast_td3/train.py \
    --env_name T1JoystickFlatTerrain \
    --exp_name FastTD3 \
    --render_interval 5000 \
    --seed 1
# FastTD3 + SimbaV2
python fast_td3/train.py \
    --env_name T1JoystickFlatTerrain \
    --exp_name FastTD3 \
    --render_interval 5000 \
    --agent fasttd3_simbav2 \
    --batch_size 8192 \
    --critic_learning_rate_end 3e-5 \
    --actor_learning_rate_end 3e-5 \
    --weight_decay 0.0 \
    --critic_hidden_dim 512 \
    --critic_num_blocks 2 \
    --actor_hidden_dim 256 \
    --actor_num_blocks 1 \
    --seed 1
```

### IsaacLab Experiments
```bash
conda activate fasttd3_isaaclab
# FastTD3
python fast_td3/train.py \
    --env_name Isaac-Velocity-Flat-G1-v0 \
    --exp_name FastTD3 \
    --render_interval 0 \
    --seed 1
# FastTD3 + SimbaV2
python fast_td3/train.py \
    --env_name Isaac-Repose-Cube-Allegro-Direct-v0 \
    --exp_name FastTD3 \
    --render_interval 0 \
    --agent fasttd3_simbav2 \
    --batch_size 8192 \
    --critic_learning_rate_end 3e-5 \
    --actor_learning_rate_end 3e-5 \
    --weight_decay 0.0 \
    --critic_hidden_dim 512 \
    --critic_num_blocks 2 \
    --actor_hidden_dim 256 \
    --actor_num_blocks 1 \
    --seed 1
```

### MTBench Experiments
```bash
conda activate fasttd3_mtbench
# FastTD3
python fast_td3/train.py \
    --env_name MTBench-meta-world-v2-mt10 \
    --exp_name FastTD3 \
    --render_interval 0 \
    --seed 1
# FastTD3 + SimbaV2
python fast_td3/train.py \
    --env_name MTBench-meta-world-v2-mt10 \
    --exp_name FastTD3 \
    --render_interval 0 \
    --agent fasttd3_simbav2 \
    --batch_size 8192 \
    --critic_learning_rate_end 3e-5 \
    --actor_learning_rate_end 3e-5 \
    --weight_decay 0.0 \
    --critic_hidden_dim 1024 \
    --critic_num_blocks 2 \
    --actor_hidden_dim 512 \
    --actor_num_blocks 1 \
    --seed 1
```

**Quick note:** For boolean-based arguments, you can set them to False by adding `no_` in front each argument, for instance, if you want to disable Clipped Q Learning, you can specify `--no_use_cdq` in your command.

## 💡 Performance-Related Tips

We used a single Nvidia A100 80GB GPU for all experiments. Here are some remarks and tips for improving performances in your setup or troubleshooting in your machine configurations.

- *Sample-efficiency* tends to improve with larger `num_envs`, `num_updates`, and `batch_size`. But this comes at the cost of *Time-efficiency*. Our default settings are optimized for wall-time efficiency on a single A100 80GB GPU. If you're using a different setup, consider tuning hyperparameters accordingly.
- When FastTD3 performance is stuck at local minima at the early phase of training in your experiments
  - First consider increasing the `num_updates`. This happens usually when the agent fails to exploit value functions. We also find higher `num_updates` tends to be helpful for relatively easier tasks or tasks with low-dimensional action spaces.
  - If the agent is completely stuck or much worse than your expectation, try using `num_steps=3` or disabling `use_cdq`.
  - For tasks that have penalty reward terms (e.g., torques, energy, action_rate, ..), consider lowering them for initial experiments, and tune the values. In some cases, curriculum learning with lower penalty terms followed by fine-tuning with stronger terms is effective.
- When you encounter out-of-memory error with your GPU, our recommendation for reducing GPU usage is (i) smaller `buffer_size`, (ii) smaller `batch_size`, and then (iii) smaller `num_envs`. Because our codebase is assigning the whole replay buffer in GPU to reduce CPU-GPU transfer bottleneck, it usually has the largest GPU consumption, but usually less harmful to reduce.
- Consider using `--compile_mode max-autotune` if you plan to run for many training steps. This may speed up training by up to 10% at the cost of a few additional minutes of heavy compilation.

## Multi-GPU Training
We support multi-GPU training. If your machine supports multiple GPUs, or specify multiple GPUs using `CUDA_VISIBLE_DEVICES`, and run `train_multigpu.py`, it will automatically use all GPUs to scale up training.

**Important:** Our multi-GPU implementation launches the **same experiment independently on each GPU** rather than distributing parameters across GPUs. This means:
- Effective number of environments: `num_envs × num_gpus`
- Effective batch size: `batch_size × num_gpus` 
- Effective buffer size: `buffer_size × num_gpus`

Each GPU runs a complete copy of the training process, which scales up data collection and training throughput proportionally to the number of GPUs.

For instance, running IsaacLab experiments with 4 GPUs and `num_envs=1024` will end up in similar results as experiments with 1 GPU with `num_envs=4096`.

## 🛝 Playing with the FastTD3 training

A Jupyter notebook (`training_notebook.ipynb`) is available to help you get started with:
- Training FastTD3 agents.
- Loading pre-trained models.
- Visualizing agent behavior.
- Potentially, re-training or fine-tuning models.

## 🤖 Sim-to-Real RL with FastTD3

We provide the [walkthrough](sim2real.md) for training deployable policies with FastTD3.

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details. Note that the repository relies on third-party libraries subject to their respective licenses.

## Acknowledgements

This codebase builds upon [LeanRL](https://github.com/pytorch-labs/LeanRL) framework. 

We would like to thank people who have helped throughout the project:

- We thank [Kevin Zakka](https://kzakka.com/) for the help in setting up MuJoCo Playground.
- We thank [Changyeon Kim](https://changyeon.site/) for testing the early version of this codebase

## Citations

### FastTD3
```bibtex
@article{seo2025fasttd3,
  title     = {FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control},
  author    = {Seo, Younggyo and Sferrazza, Carmelo and Geng, Haoran and Nauman, Michal and Yin, Zhao-Heng and Abbeel, Pieter},
  booktitle = {preprint},
  year      = {2025},
}
```

### TD3
```bibtex
@inproceedings{fujimoto2018addressing,
  title={Addressing function approximation error in actor-critic methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International conference on machine learning},
  pages={1587--1596},
  year={2018},
  organization={PMLR}
}
```

### SimbaV2
```bibtex
@article{lee2025hyperspherical,
  title={Hyperspherical normalization for scalable deep reinforcement learning},
  author={Lee, Hojoon and Lee, Youngdo and Seno, Takuma and Kim, Donghu and Stone, Peter and Choo, Jaegul},
  journal={arXiv preprint arXiv:2502.15280},
  year={2025}
}
```

### LeanRL

Following the [LeanRL](https://github.com/pytorch-labs/LeanRL)'s recommendation, we put CleanRL's bibtex here:

```bibtex
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```

### Parallel Q-Learning (PQL)
```bibtex
@inproceedings{li2023parallel,
  title={Parallel $ Q $-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation},
  author={Li, Zechu and Chen, Tao and Hong, Zhang-Wei and Ajay, Anurag and Agrawal, Pulkit},
  booktitle={International Conference on Machine Learning},
  pages={19440--19459},
  year={2023},
  organization={PMLR}
}
```

### HumanoidBench
```bibtex
@inproceedings{sferrazza2024humanoidbench,
  title={Humanoidbench: Simulated humanoid benchmark for whole-body locomotion and manipulation},
  author={Sferrazza, Carmelo and Huang, Dun-Ming and Lin, Xingyu and Lee, Youngwoon and Abbeel, Pieter},
  booktitle={Robotics: Science and Systems},
  year={2024}
}
```

### MuJoCo Playground
```bibtex
@article{zakka2025mujoco,
  title={MuJoCo Playground},
  author={Zakka, Kevin and Tabanpour, Baruch and Liao, Qiayuan and Haiderbhai, Mustafa and Holt, Samuel and Luo, Jing Yuan and Allshire, Arthur and Frey, Erik and Sreenath, Koushil and Kahrs, Lueder A and others},
  journal={arXiv preprint arXiv:2502.08844},
  year={2025}
}
```

### IsaacLab
```bibtex
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

### MTBench
```bibtex
@inproceedings{
joshi2025benchmarking,
title={Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks},
author={Viraj Joshi and Zifan Xu and Bo Liu and Peter Stone and Amy Zhang},
booktitle={Reinforcement Learning Conference},
year={2025},
url={https://openreview.net/forum?id=z0MM0y20I2}
}
```

### Getting SAC to Work on a Massive Parallel Simulator
```bibtex
@article{raffin2025isaacsim,
  title   = "Getting SAC to Work on a Massive Parallel Simulator: An RL Journey With Off-Policy Algorithms",
  author  = "Raffin, Antonin",
  journal = "araffin.github.io",
  year    = "2025",
  month   = "Feb",
  url     = "https://araffin.github.io/post/sac-massive-sim/"
}
```

### Speeding Up SAC with Massively Parallel Simulation
```bibtex
@article{shukla2025fastsac,
  title   = "Speeding Up SAC with Massively Parallel Simulation",
  author  = "Shukla, Arth",
  journal = "https://arthshukla.substack.com",
  year    = "2025",
  month   = "Mar",
  url     = "https://arthshukla.substack.com/p/speeding-up-sac-with-massively-parallel"
}
```
