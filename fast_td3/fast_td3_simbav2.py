import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def l2normalize(
    tensor: torch.Tensor, axis: int = -1, eps: float = 1e-8
) -> torch.Tensor:
    """Computes L2 normalization of a tensor."""
    return tensor / (torch.linalg.norm(tensor, ord=2, dim=axis, keepdim=True) + eps)


class Scaler(nn.Module):
    """
    A learnable scaling layer.
    """

    def __init__(
        self,
        dim: int,
        init: float = 1.0,
        scale: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), init * scale, device=device))
        self.forward_scaler = init / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler.to(x.dtype) * self.forward_scaler * x


class HyperDense(nn.Module):
    """
    A dense layer without bias and with orthogonal initialization.
    """

    def __init__(self, in_dim: int, hidden_dim: int, device: torch.device = None):
        super().__init__()
        self.w = nn.Linear(in_dim, hidden_dim, bias=False, device=device)
        nn.init.orthogonal_(self.w.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


class HyperMLP(nn.Module):
    """
    A small MLP with a specific architecture using HyperDense and Scaler.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float,
        scaler_scale: float,
        eps: float = 1e-8,
        device: torch.device = None,
    ):
        super().__init__()
        self.w1 = HyperDense(in_dim, hidden_dim, device=device)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.w2 = HyperDense(hidden_dim, out_dim, device=device)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        # `eps` is required to prevent zero vector.
        x = F.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperEmbedder(nn.Module):
    """
    Embeds input by concatenating a constant, normalizing, and applying layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float,
        device: torch.device = None,
    ):
        super().__init__()
        # The input dimension to the dense layer is in_dim + 1
        self.w = HyperDense(in_dim + 1, hidden_dim, device=device)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.c_shift = c_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_axis = torch.full(
            (*x.shape[:-1], 1), self.c_shift, device=x.device, dtype=x.dtype
        )
        x = torch.cat([x, new_axis], dim=-1)
        x = l2normalize(x, axis=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperLERPBlock(nn.Module):
    """
    A residual block using Linear Interpolation (LERP).
    """

    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int = 4,
        device: torch.device = None,
    ):
        super().__init__()
        self.mlp = HyperMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * expansion,
            out_dim=hidden_dim,
            scaler_init=scaler_init / math.sqrt(expansion),
            scaler_scale=scaler_scale / math.sqrt(expansion),
            device=device,
        )
        self.alpha_scaler = Scaler(
            dim=hidden_dim,
            init=alpha_init,
            scale=alpha_scale,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        mlp_out = self.mlp(x)
        # The original paper uses (x - residual) but x is the residual here.
        # This is interpreted as alpha * (mlp_output - residual_input)
        x = residual + self.alpha_scaler(mlp_out - residual)
        x = l2normalize(x, axis=-1)
        return x


class HyperTanhPolicy(nn.Module):
    """
    A policy that outputs a Tanh action.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        scaler_init: float,
        scaler_scale: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.mean_w1 = HyperDense(hidden_dim, hidden_dim, device=device)
        self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.mean_w2 = HyperDense(hidden_dim, action_dim, device=device)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean path
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias.to(mean.dtype)
        mean = torch.tanh(mean)
        return mean


class HyperCategoricalValue(nn.Module):
    """
    A value function that predicts a categorical distribution over a range of values.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_bins: int,
        scaler_init: float,
        scaler_scale: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.w1 = HyperDense(hidden_dim, hidden_dim, device=device)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.w2 = HyperDense(hidden_dim, num_bins, device=device)
        self.bias = nn.Parameter(torch.zeros(num_bins, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.w1(x)
        logits = self.scaler(logits)
        logits = self.w2(logits) + self.bias.to(logits.dtype)
        return logits


class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        num_blocks: int,
        c_shift: float,
        expansion: int,
        device: torch.device = None,
    ):
        super().__init__()

        self.embedder = HyperEmbedder(
            in_dim=n_obs + n_act,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
            device=device,
        )

        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                    expansion=expansion,
                    device=device,
                )
                for _ in range(num_blocks)
            ]
        )

        self.predictor = HyperCategoricalValue(
            hidden_dim=hidden_dim,
            num_bins=num_atoms,
            scaler_init=1.0,
            scaler_scale=1.0,
            device=device,
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.embedder(x)
        x = self.encoder(x)
        x = self.predictor(x)
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

        is_int = (l == u)
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
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        num_blocks: int,
        c_shift: float,
        expansion: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.qnet1 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            num_blocks=num_blocks,
            c_shift=c_shift,
            expansion=expansion,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.qnet2 = DistributionalQNetwork(
            n_obs=n_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            num_blocks=num_blocks,
            c_shift=c_shift,
            expansion=expansion,
            hidden_dim=hidden_dim,
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
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int,
        c_shift: float,
        num_blocks: int,
        std_min: float = 0.05,
        std_max: float = 0.8,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_act = n_act

        self.embedder = HyperEmbedder(
            in_dim=n_obs,
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
            device=device,
        )
        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                    expansion=expansion,
                    device=device,
                )
                for _ in range(num_blocks)
            ]
        )
        self.predictor = HyperTanhPolicy(
            hidden_dim=hidden_dim,
            action_dim=n_act,
            scaler_init=1.0,
            scaler_scale=1.0,
            device=device,
        )

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
        x = self.embedder(x)
        x = self.encoder(x)
        x = self.predictor(x)
        return x

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
