# FastTD3 - Simple and Fast RL for Humanoid Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2505.22642-b31b1b.svg)](https://arxiv.org/abs/2505.22642)

# FastTD3 + Motion Imitation with AMP

This branch supports running FastTD3 + AMP experiments. This branch is still work in progress.

Use the below script to launch experiment:

```bash
python fast_td3/train_amp_multitask.py --env_name Isaac-Humanoid-AMP-MultiTask-Direct-v0 --exp_name FastTD3-AMP --use_wandb --v_min -20.0 --v_max 20.0 --num_envs 1024 --num_updates 2 --num_steps 1 --total_timesteps 10000000 --gan_type lsgan --discriminator_reward_scale 1.0 --max_grad_norm 1.0 --buffer_size 20480 --reward_normalization --task_embedding_dim 64 --discriminator_gradient_penalty 5.0 --discriminator_logit_regularization 0.0 --seed 1
```

Known issues:
- Training collapes after a long hours of training
- The network often ignores task embedding in multi-task AMP