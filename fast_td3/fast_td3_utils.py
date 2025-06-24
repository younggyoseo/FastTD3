import os
from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict


class SimpleReplayBuffer(nn.Module):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        """
        一个简单的重放缓冲区，用于以循环缓冲区存储转换。
        支持 n 步回报和非对称观察。

        当 playground_mode=True 时，critic_observations 被视为常规观察和特权观察的拼接，
        仅存储特权部分以节省内存。

        TODO (Younggyo): 重构以将其拆分为SimpleReplayBuffer和NStepReplayBuffer
        """
        super().__init__()

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.asymmetric_obs = asymmetric_obs
        self.playground_mode = playground_mode and asymmetric_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        self.actions = torch.zeros(
            (n_env, buffer_size, n_act), device=device, dtype=torch.float
        )
        self.rewards = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.float
        )
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.truncations = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.next_observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )

        ##NOTE - 特权观测部分
        if asymmetric_obs:
            if self.playground_mode:
                #! 对于playground仅存储观察的特权部分 (n_critic_obs - n_obs)
                self.privileged_obs_size = n_critic_obs - n_obs
                self.privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
                self.next_privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
            else:
                # 存储完整的评论观察
                self.critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
                self.next_critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
        self.ptr = 0

    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size

        ##NOTE - 从tensor_dict存入数据
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next"]["critic_observations"]

            if self.playground_mode:
                # 提取并仅存储特权部分
                privileged_observations = critic_observations[:, self.n_obs :]
                next_privileged_observations = next_critic_observations[:, self.n_obs :]
                self.privileged_observations[:, ptr] = privileged_observations
                self.next_privileged_observations[:, ptr] = next_privileged_observations
            else:
                # 存储完整的评论观察
                self.critic_observations[:, ptr] = critic_observations
                self.next_critic_observations[:, ptr] = next_critic_observations
        self.ptr += 1

    def sample(self, batch_size: int):
        # 我们将采样 n_env * batch_size 的转换
        ##NOTE - n_steps = 1 时: 我们使用最经典的 1-step TD
        if self.n_steps == 1:
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )  # 形状为 (n_env, batch_size)

            #! 采样的目标形状 (n_env, batch_size, n_obs)，expand 的参数中，-1 是一个特殊值，意思是**“保持这个维度的大小不变”**。
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            ##NOTE - 抽取样本，对于每个环境，抽取的样本编号是完全不一样的，是独立随机的。最后得到了干净的二维张量(总样本数, 特征维度)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            next_observations = torch.gather(
                self.next_observations, 1, obs_indices
            ).reshape(self.n_env * batch_size, self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )

            rewards = torch.gather(self.rewards, 1, indices).reshape(
                self.n_env * batch_size
            )
            dones = torch.gather(self.dones, 1, indices).reshape(
                self.n_env * batch_size
            )
            truncations = torch.gather(self.truncations, 1, indices).reshape(
                self.n_env * batch_size
            )
            effective_n_steps = torch.ones_like(dones)
            if self.asymmetric_obs:
                if self.playground_mode:
                    # 收集特权观察
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)
                    next_privileged_observations = torch.gather(
                        self.next_privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # 与常规观察连接以形成完整的评论观察
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                    next_critic_observations = torch.cat(
                        [next_observations, next_privileged_observations], dim=1
                    )
                else:
                    # 收集完整的评论意见
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
                    next_critic_observations = torch.gather(
                        self.next_critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)

        ##NOTE - n_steps > 1 时: 我们使用 N-step TD learning。
        else:
            # 样本基索引
            if self.ptr >= self.buffer_size:
                # 当缓冲区已满时，无法防止跨不同 episodes 的采样
                # 我们通过暂时将 self.pos - 1 设置为 truncated = True 来避免这种情况（如果未完成）
                # https://github.com/DLR-RM/stable-baselines3/blob/b91050ca94f8bce7a0285c91f85da518d5a26223/stable_baselines3/common/buffers.py#L857-L860
                # TODO (Younggyo)：当此 SB3 分支合并后更改引用
                current_pos = self.ptr % self.buffer_size
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(
                    self.dones[:, current_pos - 1]
                )
                indices = torch.randint(
                    0,
                    self.buffer_size,
                    (self.n_env, batch_size),
                    device=self.device,
                )
            else:
                # 缓冲区未满 - 确保 n 步序列不超过有效数据
                max_start_idx = max(1, self.ptr - self.n_steps + 1)
                indices = torch.randint(
                    0,
                    max_start_idx,
                    (self.n_env, batch_size),
                    device=self.device,
                )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            # 获取基本过渡
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )
            if self.asymmetric_obs:
                if self.playground_mode:
                    # 收集特权观察
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # 与常规观察连接以形成完整的评论观察
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                else:
                    # 收集完整的评论意见
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)

            # 为每个样本创建顺序索引
            # 这将创建一个 [n_env, batch_size, n_step] 的索引张量
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (
                indices.unsqueeze(-1) + seq_offsets
            ) % self.buffer_size  # [环境数量, 批量大小, 步数]

            # 收集所有奖励和终端标志
            # 使用高级索引 - 结果形状：[n_env, batch_size, n_step]
            all_rewards = torch.gather(
                self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones = torch.gather(
                self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # 在第一次完成后创建奖励掩码
            # 这会创建一个累积乘积，在第一次完成后将奖励归零
            all_dones_shifted = torch.cat(
                [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2
            )  # 第一个奖励不应被屏蔽
            done_masks = torch.cumprod(
                1.0 - all_dones_shifted, dim=2
            )  # [环境数量, 批量大小, 步数]
            effective_n_steps = done_masks.sum(2)

            # 创建折扣因子
            discounts = torch.pow(
                self.gamma, torch.arange(self.n_steps, device=self.device)
            )  # [步骤 n]

            # 应用掩码和折扣到奖励
            masked_rewards = all_rewards * done_masks  # [环境数量, 批量大小, 步数]
            discounted_rewards = masked_rewards * discounts.view(
                1, 1, -1
            )  # [环境数量, 批量大小, 步数]

            # 沿着 n_step 维度累加奖励
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env，batch_size]

            # 找到每个序列的第一个完成、截断或最后一步的索引
            first_done = torch.argmax(
                (all_dones > 0).float(), dim=2
            )  # [n_env，batch_size]
            first_trunc = torch.argmax(
                (all_truncations > 0).float(), dim=2
            )  # [n_env，batch_size]

            # 处理没有完成或截断的情况
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0

            # 当没有完成或截断时，使用最后一个索引
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

            # 取完成或截断中的最小值（第一个）
            final_indices = torch.minimum(
                first_done, first_trunc
            )  # [n_env，batch_size]

            # 创建索引以收集最终的下一个观察值
            final_next_obs_indices = torch.gather(
                all_indices, 2, final_indices.unsqueeze(-1)
            ).squeeze(-1)  # [n_env，batch_size]

            # 收集最终值
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)

            if self.asymmetric_obs:
                if self.playground_mode:
                    # 收集最终的特权观察
                    final_next_privileged_observations = (
                        self.next_privileged_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.privileged_obs_size
                            ),
                        )
                    )

                    # 为输出调整形状
                    next_privileged_observations = (
                        final_next_privileged_observations.reshape(
                            self.n_env * batch_size, self.privileged_obs_size
                        )
                    )

                    # 与下一个观察值连接以形成完整的下一个评论观察值
                    next_observations_reshaped = final_next_observations.reshape(
                        self.n_env * batch_size, self.n_obs
                    )
                    next_critic_observations = torch.cat(
                        [next_observations_reshaped, next_privileged_observations],
                        dim=1,
                    )
                else:
                    # 直接收集最终下一步评论意见
                    final_next_critic_observations = (
                        self.next_critic_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.n_critic_obs
                            ),
                        )
                    )
                    next_critic_observations = final_next_critic_observations.reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )

            # 将所有内容重塑为批量维度
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(
                self.n_env * batch_size, self.n_obs
            )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations

        if self.n_steps > 1 and self.ptr >= self.buffer_size:
            # 回滚为安全采样引入的截断标志
            self.truncations[:, current_pos - 1] = curr_truncations
        return out


class EmpiricalNormalization(nn.Module):
    """根据经验值对均值和方差进行归一化。"""

    def __init__(self, shape, device, eps=1e-2, until=None):
        """初始化经验归一化模块。

        参数：
        - shape (int 或 int 元组): 除批处理轴外的输入值形状。
        - eps (float): 用于稳定性的微小值。
        - until (int 或 None): 如果指定了此参数，则链接会学习输入值，直到批处理大小的总和超过该值。
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device

        ## register_buffer 用于注册持久性张量(不参与梯度计算），这些张量在模型保存和加载时会被保留。
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor, center: bool = True) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}"
            )

        ##NOTE - 只要网络是训练模式.train()，那么training属性就会为 True
        if self.training:
            self.update(x)

        ##NOTE - 如果 center=True，则减去均值，用于状态归一化
        if center:
            return (x - self._mean) / (self._std + self.eps)

        ##NOTE - 如果 center=False，则仅除以标准差，用于奖励归一化，因为减去均值会改变奖励的正负
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)

        # 更新计数
        new_count = self.count + batch_size

        # 更新均值
        delta = batch_mean - self._mean
        self._mean += (batch_size / new_count) * delta

        # Update variance using Chan's parallel algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        if self.count > 0:  # Ensure we're not dividing by zero
            batch_var = torch.mean((x - batch_mean) ** 2, dim=0, keepdim=True)
            delta2 = batch_mean - self._mean
            m_a = self._var * self.count
            m_b = batch_var * batch_size
            M2 = m_a + m_b + (delta2**2) * (self.count * batch_size / new_count)
            self._var = M2 / new_count
        else:
            # 对于第一批，只需使用批量方差
            self._var = torch.mean((x - self._mean) ** 2, dim=0, keepdim=True)

        self._std = torch.sqrt(self._var)
        self.count = new_count

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class RewardNormalizer(nn.Module):
    def __init__(
        self,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("G", torch.zeros(1, device=device))  # 折扣回报的运行估算
        self.register_buffer("G_r_max", torch.zeros(1, device=device))  # 运行最大值
        self.G_rms = EmpiricalNormalization(shape=1, device=device)
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon

    def _scale_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        var_denominator = self.G_rms.std[0] + self.epsilon
        min_required_denominator = self.G_r_max / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        return rewards / denominator

    def update_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        self.G = self.gamma * (1 - dones) * self.G + rewards
        self.G_rms.update(self.G.view(-1, 1))
        self.G_r_max = max(self.G_r_max, max(abs(self.G)))

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        return self._scale_reward(rewards)


class PerTaskEmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values for each task."""

    def __init__(
        self,
        num_tasks: int,
        shape: tuple,
        device: torch.device,
        eps: float = 1e-2,
        until: int = None,
    ):
        """
        Initialize PerTaskEmpiricalNormalization module.

        Args:
            num_tasks (int): The total number of tasks.
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If specified, learns until the sum of batch sizes
                                 for a specific task exceeds this value.
        """
        super().__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.num_tasks = num_tasks
        self.shape = shape
        self.eps = eps
        self.until = until
        self.device = device

        # 缓冲区现在具有用于任务的前导维度
        self.register_buffer("_mean", torch.zeros(num_tasks, *shape).to(device))
        self.register_buffer("_var", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer("_std", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer(
            "count", torch.zeros(num_tasks, dtype=torch.long).to(device)
        )

    def forward(
        self, x: torch.Tensor, task_ids: torch.Tensor, center: bool = True
    ) -> torch.Tensor:
        """
        Normalize the input tensor `x` using statistics for the given `task_ids`.

        Args:
            x (torch.Tensor): Input tensor of shape [num_envs, *shape].
            task_ids (torch.Tensor): Tensor of task indices, shape [num_envs].
            center (bool): If True, center the data by subtracting the mean.
        """
        if x.shape[1:] != self.shape:
            raise ValueError(f"Expected input shape (*, {self.shape}), got {x.shape}")
        if x.shape[0] != task_ids.shape[0]:
            raise ValueError("Batch size of x and task_ids must match.")

        # 收集当前批次任务的统计数据
        # 调整 task_ids 以进行广播：[num_envs] -> [num_envs, 1, ...]
        view_shape = (task_ids.shape[0],) + (1,) * len(self.shape)
        task_ids_expanded = task_ids.view(view_shape).expand_as(x)

        mean = self._mean.gather(0, task_ids_expanded)
        std = self._std.gather(0, task_ids_expanded)

        if self.training:
            self.update(x, task_ids)

        if center:
            return (x - mean) / (std + self.eps)
        else:
            return x / (std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor, task_ids: torch.Tensor):
        """Update running statistics for the tasks present in the batch."""
        unique_tasks = torch.unique(task_ids)

        for task_id in unique_tasks:
            if self.until is not None and self.count[task_id] >= self.until:
                continue

            # 为当前任务创建一个掩码以选择数据
            mask = task_ids == task_id
            x_task = x[mask]
            batch_size = x_task.shape[0]

            if batch_size == 0:
                continue

            # 更新此任务的计数
            old_count = self.count[task_id].clone()
            new_count = old_count + batch_size

            # 更新均值
            task_mean = self._mean[task_id]
            batch_mean = torch.mean(x_task, dim=0)
            delta = batch_mean - task_mean
            self._mean[task_id] = task_mean + (batch_size / new_count) * delta

            # 使用Chan的并行算法更新方差
            if old_count > 0:
                batch_var = torch.var(x_task, dim=0, unbiased=False)
                m_a = self._var[task_id] * old_count
                m_b = batch_var * batch_size
                M2 = m_a + m_b + (delta**2) * (old_count * batch_size / new_count)
                self._var[task_id] = M2 / new_count
            else:
                # 对于此任务的第一批
                self._var[task_id] = torch.var(x_task, dim=0, unbiased=False)

            self._std[task_id] = torch.sqrt(self._var[task_id])
            self.count[task_id] = new_count


class PerTaskRewardNormalizer(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        """
        Per-task reward normalizer, motivation comes from BRC (https://arxiv.org/abs/2505.23150v1)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.device = device

        # Per-task running estimate of the discounted return
        self.register_buffer("G", torch.zeros(num_tasks, device=device))
        # Per-task running-max of the discounted return
        self.register_buffer("G_r_max", torch.zeros(num_tasks, device=device))
        # Use the new per-task normalizer for the statistics of G
        self.G_rms = PerTaskEmpiricalNormalization(
            num_tasks=num_tasks, shape=(1,), device=device
        )

    def _scale_reward(
        self, rewards: torch.Tensor, task_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Scales rewards using per-task statistics.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        # Gather stats for the tasks in the batch
        std_for_batch = self.G_rms._std.gather(0, task_ids.unsqueeze(-1)).squeeze(-1)
        g_r_max_for_batch = self.G_r_max.gather(0, task_ids)

        var_denominator = std_for_batch + self.epsilon
        min_required_denominator = g_r_max_for_batch / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        # Add a small epsilon to the final denominator to prevent division by zero
        # in case g_r_max is also zero.
        return rewards / (denominator + self.epsilon)

    def update_stats(
        self, rewards: torch.Tensor, dones: torch.Tensor, task_ids: torch.Tensor
    ):
        """
        Updates the running discounted return and its statistics for each task.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            dones (torch.Tensor): Done tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        if not (rewards.shape == dones.shape == task_ids.shape):
            raise ValueError("rewards, dones, and task_ids must have the same shape.")

        # === Update G (running discounted return) ===
        # Gather the previous G values for the tasks in the batch
        prev_G = self.G.gather(0, task_ids)
        # Update G for each environment based on its own reward and done signal
        new_G = self.gamma * (1 - dones.float()) * prev_G + rewards
        # Scatter the updated G values back to the main buffer
        self.G.scatter_(0, task_ids, new_G)

        # === Update G_rms (statistics of G) ===
        # The update function handles the per-task logic internally
        self.G_rms.update(new_G.unsqueeze(-1), task_ids)

        # === Update G_r_max (running max of |G|) ===
        prev_G_r_max = self.G_r_max.gather(0, task_ids)
        # Update the max for each environment
        updated_G_r_max = torch.maximum(prev_G_r_max, torch.abs(new_G))
        # Scatter the new maxes back to the main buffer
        self.G_r_max.scatter_(0, task_ids, updated_G_r_max)

    def forward(self, rewards: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Normalizes rewards. During training, it also updates the running statistics.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        return self._scale_reward(rewards, task_ids)


def cpu_state(sd):
    # detach & move to host without locking the compute stream
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def save_params(
    global_step,
    actor,
    qnet,
    qnet_target,
    obs_normalizer,
    critic_obs_normalizer,
    args,
    save_path,
):
    """Save model parameters and training configuration to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": cpu_state(actor.state_dict()),
        "qnet_state_dict": cpu_state(qnet.state_dict()),
        "qnet_target_state_dict": cpu_state(qnet_target.state_dict()),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "critic_obs_normalizer_state": (
            cpu_state(critic_obs_normalizer.state_dict())
            if hasattr(critic_obs_normalizer, "state_dict")
            else None
        ),
        "args": vars(args),  # Save all arguments
        "global_step": global_step,
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved parameters and configuration to {save_path}")
