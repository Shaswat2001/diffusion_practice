from .gaussian_diffusion import GaussianDiffusion
from .ddpm import DDPM
from .ddim import DDIM

diffusion_models = {
    "DDPM" : DDPM, 
    "DDIM": DDIM,
}