import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_SIM_TYPES = {"", "sim_actor", "sim_critic", "sim_both"}


def _validate_sim_config(sim_type: str, sim_dimension: int, seq_len: int) -> None:
    if sim_type not in VALID_SIM_TYPES:
        raise ValueError(
            f"Unsupported sim_type '{sim_type}'. Expected one of {sorted(VALID_SIM_TYPES)}"
        )
    if sim_dimension <= 0:
        raise ValueError("sim_dimension must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, seq_len=8, simnorm_dim=8):
        super().__init__()
        self.L = seq_len
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], self.L, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(seq_len={self.L}, simnorm_dim={self.dim})"


class SimNormLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        seq_len: int,
        simnorm_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        out_features = seq_len * simnorm_dim
        self.linear = nn.Linear(in_features, out_features, device=device)
        self.norm = nn.LayerNorm(out_features, device=device)
        self.simnorm = SimNorm(seq_len, simnorm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.simnorm(self.norm(self.linear(x)))

class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        sim_type: str,
        sim_dimension: int,
        seq_len: int,
        device: torch.device = None,
    ):
        super().__init__()
        _validate_sim_config(sim_type, sim_dimension, seq_len)

        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
        )

        if sim_type in ["sim_both", "sim_critic"]:
            self.fc_head = nn.Sequential(
                SimNormLinear(
                    hidden_dim // 2,
                    seq_len=seq_len,
                    simnorm_dim=sim_dimension,
                    device=device,
                ),
                nn.Linear(seq_len * sim_dimension, num_atoms, device=device),
            )
        else:
            self.fc_head = nn.Sequential(
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
        x = self.fc_head(x)
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
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        is_int = l == u
        l_mask = is_int & (l > 0)
        u_mask = is_int & (l == 0)

        l = torch.where(l_mask, l - 1, l)
        u = torch.where(u_mask, u + 1, u)

        next_dist = F.softmax(self.forward(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )
        return proj_dist


class Critic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        sim_type: str,
        sim_dimension: int,
        seq_len: int,
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
            sim_type=sim_type,
            sim_dimension=sim_dimension,
            seq_len=seq_len,
            device=device,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            sim_type=sim_type,
            sim_dimension=sim_dimension,
            seq_len=seq_len,
            device=device,
        )

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

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=1)


class Actor(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_envs: int,
        init_scale: float,
        hidden_dim: int,
        std_min: float = 0.05,
        std_max: float = 0.8,
        sim_type: str = "",
        sim_dimension: int = 64,
        seq_len: int=8,
        device: torch.device = None,
    ):
        super().__init__()
        _validate_sim_config(sim_type, sim_dimension, seq_len)

        self.n_act = n_act
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
        )

        if sim_type in ["sim_both", "sim_actor"]:
            self.fc_head = SimNormLinear(
                hidden_dim // 2,
                seq_len=seq_len,
                simnorm_dim=sim_dimension,
                device=device,
            )
            self.fc_mu = nn.Sequential(
                nn.Linear(seq_len * sim_dimension, n_act, device=device),
                nn.Tanh(),
            )
        else:
            self.fc_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
                nn.ReLU(),
            )
            self.fc_mu = nn.Sequential(
                nn.Linear(hidden_dim // 4, n_act, device=device),
                nn.Tanh(),
            )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

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
        x_net = self.net(x)
        x_head = self.fc_head(x_net)
        action = self.fc_mu(x_head)
        return action

    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor = None, deterministic: bool = False
    ) -> torch.Tensor:
        # If dones is provided, resample noise for environments that are done
        if dones is not None and dones.sum() > 0:
            # Generate new noise scales for done environments (one per environment)
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )

            # Update only the noise scales for environments that are done
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales)
            )

        act = self(obs)
        if deterministic:
            return act

        noise = torch.randn_like(act) * self.noise_scales
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
        # TODO: Optimize the code to be compatible with cudagraphs
        # Currently in-place creation of task_indices is not compatible with cudagraphs
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
        # TODO: Optimize the code to be compatible with cudagraphs
        # Currently in-place creation of task_indices is not compatible with cudagraphs
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
