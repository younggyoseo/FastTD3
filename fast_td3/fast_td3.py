import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionalQNetwork(nn.Module):
    """
    来自 C51 论文的分布式 Q 网络。
    传统Q学习 (如DQN, TD3): 网络学习一个单一的数值 Q(s, a)，代表在状态 s 执行动作 a 后所有未来回报的期望值（平均值）。
    分布式Q学习 (你的代码): 网络学习一个概率分布。它不预测平均值，而是预测获得不同回报值的概率。例如，它可能会预测“有70%的概率回报是10，有30%的概率回报是-5”，而不是直接预测平均值 10*0.7 + (-5)*0.3 = 5.5。

    如果是输出价值的高斯分布，有固有缺陷，因为高斯分布是单峰的，无法表示多峰分布。
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        return x

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        网络输出的形状是 [batch_size, num_atoms]，表示每个动作的价值分布，num_atoms 代表有num_atoms 个 q_support 值。

        处理方式：
        1. logits 通过 softmax 转换为概率分布 —— 原始分布。
        2. 计算 target Q 值 target_z。
        3. 计算出来的 target_z 可能为小数，因此需要将其映射到离散的 q_support 上。
        4. 将 target_z 映射到 q_support 上的两个最近点 l 和 u。如当 target_z = 2.3 时，q_support = [0, 1, 2, 3, 4]，则 l = 2, u = 3。
        5. 通过线性插值，将 target_z 的概率分布分配到 l 和 u 上。如当 target_z = 2.3 时，则将 0.3 的概率分配到 u 上，0.7 的概率分配到 l 上（target_z 离 l 更近）。
        6. 最终得到的 proj_dist 是一个新的概率分布，表示在当前状态下采取某个动作后，可能获得的不同回报值的概率分布
        7. proj_dist 的形状是 [batch_size, num_atoms]，与网络输出的形状相同。
        """

        ##NOTE - 每个 q-support 点之间的间隔
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        ##NOTE - q_support 是一个一维张量，包含了从 v_min 到 v_max 之间的 num_atoms 个等间距的点。
        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)

        """
        b：归一化的每个 target_z 的位置。
        l：每个 target_z 在网格中的向下取整。
        u：每个 target_z 在网格中的向上取整。
        
        target_z - self.v_min: 这一步计算出每个 target_z 相对于网格起点的距离。
        / delta_z: 这一步将这个距离用网格的单位间距进行归一化。
        """
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()  # low
        u = torch.ceil(b).long()  # up

        ##NOTE - 当 b 为整数时，防止 l 和 u 相等。
        # 这一步确保 l 和 u 不会同时指向同一个网格点
        l_mask = torch.logical_and((u > 0), (l == u))
        u_mask = torch.logical_and((l < (self.num_atoms - 1)), (l == u))

        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)

        ##SECTION - 概率重分配
        """
        这段代码正在执行概率的重新分配。它将下一状态的价值分布 P(s', a')，根据贝尔曼更新后的新位置，通过线性插值的方法，“涂抹”到一个新的、代表当前状态目标分布的直方图上。
        """

        ##NOTE - 把网络的前向传播结果转换为概率分布
        ##NOTE - next_dist 形状为 [batch_size, num_atoms]，是原始Q值的索引概率分布
        ##NOTE - proj_dist 形状为 [batch_size, num_atoms]，是target Q值的概率分布，但是target Q值可能是小数，因此需要将其映射到离散的 q_support 上。
        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)  # 初始化投影分布

        ##NOTE - 计算偏移量
        """
        想象一下：我们有 batch_size 个长度为 num_atoms 的概率分布，如果把它们全部展平成一个一维长向量，那么：
        - 第 0 个样本的数据占据索引 0 到 num_atoms - 1。
        - 第 1 个样本的数据占据索引 num_atoms 到 2*num_atoms - 1。
        - 第 i 个样本的数据占据索引 i*num_atoms 到 (i+1)*num_atoms - 1。
        这个 offset 张量计算出的就是每个样本的起始索引，即 [0, num_atoms, 2*num_atoms, ...]，并扩展成 [batch_size, num_atoms] 的形状。
        """
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        """
        view(-1) 作用是将多维张量展平为一维张量。
        index_add_ 是一个原地操作，它的功能是：tensor.index_add_(dim, index, source)，将 source 张量中的值加到 tensor 中由 index 指定的位置上。
        
        原始 Q 值，可能是 50 个整数值，每个值对应一个概率。
        target Q 值，可能是 50 个小数，因此需要重新分配概率到离散的 50 个整数值上。
        """
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )  # 处理下取整 l
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )  # 处理上取整 u
        return proj_dist
        ##!SECTION


class Critic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            device=device,
        )

        ##NOTE - 创建一个固定的、不可训练的张量，并将其注册为神经网络模块（Critic 类）的一部分。
        # 这部分创建了一个一维张量。它包含了从 v_min 到 v_max 之间的 num_atoms 个等间距的点。
        # 在分布式强化学习的上下文中，这个张量就是价值分布的“支撑” (Support)或 “原子” (Atoms)。它代表了Q值可能取到的一系列固定的离散值。
        self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
        )
        self.device = device

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.qnet1(obs, actions), self.qnet2(obs, actions)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        q1_proj = self.qnet1.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        q2_proj = self.qnet2.projection(
            obs,
            actions,
            rewards,
            bootstrap,
            discount,
            self.q_support,
            self.q_support.device,
        )
        return q1_proj, q2_proj

    ##NOTE - 计算概率分布的期望值（Expected Value）。
    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=1)


class Actor(nn.Module):
    """
    比原版 TD3 多一层，且隐藏层逐层减小。
    原版 TD3 的 Actor 网络结构是：
    输入层: Linear(obs_dim, 256)
    隐藏层: Linear(256, 256)
    输出层: Linear(256, act_dim)
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        std_min: float = 0.05,
        std_max: float = 0.8,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.ReLU(),
        )

        ##NOTE - 特殊初始化这一层的目的是为了实现一个重要的目标：让智能体在训练刚开始时，倾向于输出接近于零的小动作。
        """
        当权重和偏置都非常接近于零时，无论输入 x 是什么，这一层的输出 W*x + b 都会非常接近于零。再经过 Tanh 激活函数，结果仍然是一个接近于零的值。

        好处: 这鼓励智能体在训练初期围绕一个“什么都不做”的中性策略进行温和的探索，这是一个更安全、更稳定的学习起点。
        """
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dim // 4, n_act, device=device),
            nn.Tanh(),
        )

        ##NOTE - 初始化 mu 的权重和偏置，正态分布中采样，偏置为0，标准差为 init_scale。
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        # 创建一个形状为 [num_envs, 1] 的张量，其中的每个元素都是 [0, 1) 之间的一个随机数。
        # 每一行都为对应的并行环境分配了一个随机的噪声标准差。
        noise_scales = (
            torch.rand(num_envs, 1, device=device) * (std_max - std_min) + std_min
        )
        self.register_buffer("noise_scales", noise_scales)

        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs
        self.device = device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        x = self.net(x)
        action = self.fc_mu(x)
        return action

    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        """
        至少有一个并行环境报告回合结束 (done=True) 时，才会执行。
        """
        # 如果提供了dones，则为已完成的环境重新采样噪声
        if dones is not None and dones.sum() > 0:
            # 为完成的环境生成新的噪声尺度（每个环境一个）。
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # 仅更新已完成环境的噪声尺度，dones 是一个一维张量，形状为 [num_envs]，通过view(-1, 1)变为[num_envs, 1]
            # >0 是一个按元素比较操作，返回一个布尔张量，表示每个元素是否大于0。
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales = torch.where(dones_view, new_scales, self.noise_scales)

        act = self(obs)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales

        ##NOTE - 不裁剪的好处：大多数标准的强化学习环境（如 Gymnasium/MuJoCo）在接收到一个动作后，会自动将该动作裁剪到其合法的动作空间范围内。未裁剪的、带有“意图”的动作（即使这个动作在物理上无法被完全执行），我们可以让 Critic 学习到一个更平滑、信息更丰富的 Q 函数。这个更准确的 Q 函数反过来又能为 Actor 的更新提供更有效的梯度，帮助策略更快地收敛到最优。
        return act + noise


class MultiTaskActor(Actor):
    def __init__(self, num_tasks: int, task_embedding_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim
        self.task_embedding = nn.Embedding(
            num_tasks, task_embedding_dim, max_norm=1.0, device=self.device
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        task_ids_one_hot = obs[..., -self.num_tasks :]
        task_indices = torch.argmax(task_ids_one_hot, dim=1)
        task_embeddings = self.task_embedding(task_indices)
        obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        return super().forward(obs)


class MultiTaskCritic(Critic):
    def __init__(self, num_tasks: int, task_embedding_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim
        self.task_embedding = nn.Embedding(
            num_tasks, task_embedding_dim, max_norm=1.0, device=self.device
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        task_ids_one_hot = obs[..., -self.num_tasks :]
        task_indices = torch.argmax(task_ids_one_hot, dim=1)
        task_embeddings = self.task_embedding(task_indices)
        obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        return super().forward(obs, actions)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
    ) -> torch.Tensor:
        task_ids_one_hot = obs[..., -self.num_tasks :]
        task_indices = torch.argmax(task_ids_one_hot, dim=1)
        task_embeddings = self.task_embedding(task_indices)
        obs = torch.cat([obs[..., : -self.num_tasks], task_embeddings], dim=-1)
        return super().projection(obs, actions, rewards, bootstrap, discount)
