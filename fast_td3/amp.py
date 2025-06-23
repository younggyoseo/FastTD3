import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(
        self,
        n_amp_obs: int,
        hidden_dim: int,
        lsgan_reward_scale: float,
        discriminator_gradient_penalty: float,
        device: torch.device = None,
    ):
        super().__init__()

        if discriminator_gradient_penalty > 0:
            self.net = nn.Sequential(
                nn.Linear(n_amp_obs, hidden_dim, device=device),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2, device=device),
                nn.LeakyReLU(0.2),
            )
            self.fc_head = nn.Linear(hidden_dim // 2, 1, device=device)
        else:
            self.net = nn.Sequential(
                spectral_norm(nn.Linear(n_amp_obs, hidden_dim, device=device)),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2, device=device)),
                nn.LeakyReLU(0.2),
            )
            self.fc_head = spectral_norm(nn.Linear(hidden_dim // 2, 1, device=device))

        self.lsgan_reward_scale = lsgan_reward_scale
        self.device = device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        x = self.net(x)
        return self.fc_head(x)

    @torch.no_grad()
    def get_lsgan_rewards(self, obs: torch.Tensor) -> torch.Tensor:
        amp_logits = self(obs)
        style_reward = 1.0 - self.lsgan_reward_scale * torch.square(1.0 - amp_logits)
        style_reward = torch.clamp(style_reward, min=0.0)
        style_reward = style_reward.view(obs.shape[0])
        return style_reward.detach()

    @torch.no_grad()
    def get_standard_gan_rewards(self, obs: torch.Tensor) -> torch.Tensor:
        amp_logits = self(obs)
        style_reward = -torch.log(
            torch.maximum(
                1 - torch.sigmoid(amp_logits),
                torch.tensor(0.001, device=self.device),
            )
        )
        style_reward = style_reward.view(obs.shape[0])
        return style_reward.detach()


class MultiTaskDiscriminator(Discriminator):
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