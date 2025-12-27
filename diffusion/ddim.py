import torch
import numpy as np
from misc.diffusion_utils import extract_tensor_from_value

from .gaussian_diffusion import GaussianDiffusion

class DDIM(GaussianDiffusion):

    def __init__(self):
        super().__init__()

        self.steps = self.config["sample_steps"]
        self.eta = self.config.get("eta", 0.0)

        if self.config["sampling_method"] == "linear":
            self.sampling_timesteps = np.array(list(range(0, self.diffusion_steps, self.diffusion_steps // self.steps)))
        elif self.config["sampling_method"] == "quadratic":
            self.sampling_timesteps = (np.linspace(0, np.sqrt(self.diffusion_steps * 0.8), self.steps) ** 2).astype(np.int64)
        else:
            raise NotImplementedError(f"method {self.config['sampling_method']} not implemented!")

    def p_sample(self, model, x, t, denoised_clip: bool = False, denoise_fun = None, model_kwargs=None):
        
        output = self.p_mean_variance(model, x, t, denoised_clip, denoise_fun, model_kwargs)
        eps = self.calculate_eps_from_xstart(x, t, output["pred_xstart"])

        alpha_bar = extract_tensor_from_value(self.schedular.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract_tensor_from_value(self.schedular.alphas_cumprod_prev, t, x.shape)
        sigma = (self.eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev))
        noise = torch.randn_like(x)
        
        mean_pred = (output["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": output["pred_xstart"]}

    def p_reverse(self, model, shape, noise = None, denoised_clip: bool = False, denoise_fun = None, model_kwargs=None):
        
        assert isinstance(shape, (tuple, list))

        timesteps = list(reversed(range(self.sampling_timesteps)))

        if noise is None:
            img = torch.randn(shape, dtype=torch.float32)
        else:
            img = noise

        for t in timesteps:
            tm = torch.tensor([t] * shape[0])
            output = self.p_sample(model, img, tm, denoised_clip, denoise_fun, model_kwargs)
            img = output["sample"]

        return img
        
