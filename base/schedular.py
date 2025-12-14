import torch

class LinearNoiseSchedular:

    def __init__(self, timesteps, beta_start, beta_end):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_alphas_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):

        org_shape = original.shape
        batch_size = org_shape.shape[0]

        sqrt_alphas_cum_prod = self.sqrt_alphas_cum_prod[t].reshape(batch_size)
        sqrt_one_alphas_cum_prod = self.sqrt_one_alphas_cum_prod[t].reshape(batch_size)

        for _ in range(len(org_shape) - 1):
            sqrt_alphas_cum_prod = sqrt_alphas_cum_prod.unsqueeze(-1)
            sqrt_one_alphas_cum_prod = sqrt_one_alphas_cum_prod.unsqueeze(-1)

        return sqrt_alphas_cum_prod * original + sqrt_one_alphas_cum_prod * noise
    
    def sample_prev_timestep(self, xt, noise_pred, t):

        x0 = xt - (self.sqrt_one_alphas_cum_prod[t] * noise_pred) / self.sqrt_alphas_cum_prod[t]
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = xt - (())

        