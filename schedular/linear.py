import numpy as np
from .base import BaseBetaSchedular

class ConstantBetaSchedular(BaseBetaSchedular):

    def __init__(self, beta_start: float = 0.001, beta_end: float = 0.001, diffusion_steps: int = 1000, **kwargs):
        
        betas = beta_end * np.ones((diffusion_steps,))
        super().__init__(betas)


class LinearBetaSchedular(BaseBetaSchedular):

    def __init__(self, beta_start: float = 0.001, beta_end: float = 0.001, diffusion_steps: int = 1000, **kwargs):
        
        betas = np.linspace(start= beta_start, stop= beta_end, num= diffusion_steps, dtype=np.float32)
        super().__init__(betas)

class WarmupBetaSchedular(BaseBetaSchedular):

    def __init__(self, beta_start: float = 0.001, beta_end: float = 0.001, diffusion_steps: int = 1000, warmup_ratio: float = 0.1, **kwargs):
        
        assert warmup_ratio is not None

        betas = beta_end * np.ones((diffusion_steps,))
        warmup_steps = int(warmup_ratio * diffusion_steps)
        betas[:warmup_steps] = np.linspace(start= beta_start, stop= beta_end, num= warmup_steps, dtype=np.float32)
        super().__init__(betas)
