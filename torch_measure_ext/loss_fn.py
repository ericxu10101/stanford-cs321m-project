import torch
from torch.distributions import Beta


def beta_nll(predicted_probs: torch.Tensor, observed: torch.Tensor, phi: float = 10.0) -> torch.Tensor:
    """Beta negative log-likelihood for continuous (0, 1) responses.

    Uses a Beta distribution parameterized by mean ``mu`` and precision
    ``phi``: ``a = mu * phi``, ``b = (1 - mu) * phi``.

    Parameters
    ----------
    predicted_probs : torch.Tensor
        Model-predicted mean mu, should be in (0, 1).
    observed : torch.Tensor
        Observed response probabilities in (0, 1).
    phi : float
        Precision parameter of the Beta distribution. Higher values
        concentrate the distribution more tightly around mu.

    Returns
    -------
    torch.Tensor
        Scalar mean NLL.

    References
    ----------
    .. [1] Item Response Scaling Laws (ICML 2026).
    """
    eps = 1e-6
    observed = observed.clamp(eps, 1 - eps)
    predicted_probs = predicted_probs.clamp(eps, 1 - eps)

    a = predicted_probs * phi
    b = (1.0 - predicted_probs) * phi

    return -Beta(a, b).log_prob(observed).mean()
