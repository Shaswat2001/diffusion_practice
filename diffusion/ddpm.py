import torch
import numpy as np
from misc.diffusion_utils import extract_tensor_from_value

from .gaussian_diffusion import GaussianDiffusion

class DDPM(GaussianDiffusion):

    def __init__(self):
        super().__init__()

    def p_sample(self, model, x, t, denoised_clip: bool = False, denoise_fun = None, model_kwargs=None):

        output = self.p_mean_variance(model, x, t, denoised_clip, denoise_fun, model_kwargs)
        noise = torch.randn_like(x, dtype=torch.float32)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 0
        sample = output["mean"] + nonzero_mask * torch.exp(0.5 * output["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": output["pred_xstart"]}
    
    def p_reverse(self, model, shape, noise = None, denoised_clip: bool = False, denoise_fun = None, model_kwargs=None):

        assert isinstance(shape, (tuple, list))

        timesteps = list(reversed(range(self.diffusion_steps)))

        if noise is None:
            img = torch.randn(shape, dtype=torch.float32)
        else:
            img = noise

        for t in timesteps:
            tm = torch.tensor([t] * shape[0])
            output = self.p_sample(model, img, tm, denoised_clip, denoise_fun, model_kwargs)
            img = output["sample"]

        return img
