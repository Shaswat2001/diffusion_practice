import numpy as np

class BaseBetaSchedular:

    def __init__(self, betas: np.ndarray):
        
        self.betas = betas
        self.formulate()

    def formulate(self):

        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.coefficient1 = self.betas * np.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.coefficient2 = (1 - self.alphas_cumprod_prev) * np.sqrt(self.alphas) / (1 - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])) if len(self.posterior_variance) > 1 else np.array([])