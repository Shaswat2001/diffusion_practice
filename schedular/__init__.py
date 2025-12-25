from .linear import LinearBetaSchedular, WarmupBetaSchedular, ConstantBetaSchedular

schedulars = {
    "constant": ConstantBetaSchedular,
    "linear": LinearBetaSchedular,
    "warmup": WarmupBetaSchedular
}