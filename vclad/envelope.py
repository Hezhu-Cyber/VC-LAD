"""Cosine event envelope extracted from the experiment implementation.
This module alone is not a full fault-injection pipeline.
"""
import numpy as np

def _smooth_fault_envelope(
    length: int,
    rise_frac: float = 0.30,
    fall_frac: float = 0.20,
) -> np.ndarray:
    """Paper Eq. (17), sampled at u in [0, 1]; defaults rho_r=.30, rho_f=.20."""
    if rise_frac < 0 or fall_frac < 0 or rise_frac + fall_frac > 1:
        raise ValueError("rise/fall fractions must be nonnegative and sum to at most 1")
    n = int(length)
    if n <= 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)
    u = np.linspace(0.0, 1.0, n)
    envelope = np.ones(n, dtype=float)
    if rise_frac > 0:
        rising = u < rise_frac
        envelope[rising] = 0.5 * (1.0 - np.cos(np.pi * u[rising] / rise_frac))
    if fall_frac > 0:
        falling = u > 1.0 - fall_frac
        envelope[falling] = 0.5 * (1.0 + np.cos(
            np.pi * (u[falling] - 1.0 + fall_frac) / fall_frac))
    return np.clip(envelope, 0.0, 1.0)
