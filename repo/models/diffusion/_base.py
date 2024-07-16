import torch
from torch import nn

class BaseDiff(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_diffusion_timesteps = cfg.generator.num_diffusion_timesteps
        self.denoise_structure = cfg.generator.get('denoise_structure', True)
        self.denoise_atom = cfg.generator.get('denoise_atom', True)
        self.time_sampler = cfg.generator.get('time_sampler', 'symmetric')

    def sample_time(self, batch_size, device='cuda', ctn=False):
        if self.time_sampler == 'uniform':
            time = torch.rand(batch_size) 
            if not ctn:
                time = time * self.num_diffusion_timesteps
                time = torch.round(time.clip(0, self.num_diffusion_timesteps - 1)).long()

        elif self.time_sampler == 'symmetric':
            num_graphs = batch_size
            time = torch.randint(
                0, self.num_diffusion_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time = torch.cat(
                [time, self.num_diffusion_timesteps - time - 1], dim=0)[:num_graphs]
            if ctn:
                time = time / self.num_diffusion_timesteps

        elif self.time_sampler == 'random':
            lowest_t = 0
            time = torch.randint(
                lowest_t, self.num_diffusion_timesteps + 1, size=(batch_size,), device=device).float()

        return time.to(device)