"""Synthetic benchmarks with additive superposition (Sec. 3.1).

Three tasks of increasing heterogeneity, each a mixture of ``K ~ U{K_min..K_max}``
sources with one dominant source (amplitude ~ U(5, 10) vs. U(0.2, 1) for the
others) plus additive Gaussian noise:

* **A** ``sinusoids`` -- overlapping sinusoids (frequency-domain structure),
* **B** ``gaussian_bumps`` -- Gaussian bumps (time-domain, no periodicity),
* **C** ``multiscale`` -- sinusoids times Gaussian envelopes.

All generators return ``(signals [B, N], params [B, max_K, P], K [B], components
[B, max_K, N])``; unused source rows are zero. With ``seed`` set, the global
``torch`` and ``numpy`` generators are re-seeded, and all randomness comes from
``torch`` in a fixed per-sample order, so a seed determines the data exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

GeneratorOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _seed(seed: int | None) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def generate_sinusoids(
    batch_size: int,
    K_range: tuple[int, int] = (1, 4),
    N: int = 256,
    noise_std: float = 0.1,
    dominant: bool = True,
    seed: int | None = None,
    amp_ratio: float | None = None,
) -> GeneratorOutput:
    """Task A: overlapping sinusoids ``A sin(2 pi f t + phi)`` on ``t = linspace(0, 1, N)``.

    Frequencies ``f ~ U(2, 10)`` Hz, phases ``phi ~ U(0, 2 pi)``, amplitudes
    ``A ~ U(0.2, 1.0)``; with ``dominant`` the first source has ``A ~ U(5, 10)``.

    Args:
        batch_size: number of mixtures.
        K_range: inclusive range of the source count.
        N: signal length.
        noise_std: std of the additive Gaussian noise.
        dominant: make the first source dominant.
        seed: re-seed the global generators first.
        amp_ratio: amplitude-ratio sweep (App. I, Table 11). If set (with
            ``dominant``), the dominant amplitude is ``amp_ratio`` times a draw
            from the secondary sources' ``U(0.2, 1.0)``, so ``amp_ratio = 1``
            makes it statistically identical to the others. The number of
            random draws is unchanged, so for a fixed seed the secondary sources
            are identical across ratios.

    Returns:
        ``signals [B, N]``, ``params [B, max_K, 3]`` = (frequency, amplitude,
        phase), ``K [B]``, ``components [B, max_K, N]``.
    """
    _seed(seed)
    B = batch_size
    max_K = K_range[1]
    t = torch.linspace(0, 1, N)

    params = torch.zeros(B, max_K, 3)
    components = torch.zeros(B, max_K, N)
    signals = torch.zeros(B, N)
    K_vals = torch.randint(K_range[0], K_range[1] + 1, (B,))

    for b in range(B):
        for k in range(K_vals[b].item()):
            freq = torch.rand(1).item() * 8 + 2
            phase = torch.rand(1).item() * 2 * np.pi
            if dominant and k == 0:
                if amp_ratio is not None:
                    amp = (torch.rand(1).item() * 0.8 + 0.2) * amp_ratio
                else:
                    amp = torch.rand(1).item() * 5 + 5
            else:
                amp = torch.rand(1).item() * 0.8 + 0.2

            params[b, k] = torch.tensor([freq, amp, phase])
            comp = amp * torch.sin(2 * np.pi * freq * t + phase)
            components[b, k] = comp
            signals[b] += comp

    signals += torch.randn(B, N) * noise_std
    return signals, params, K_vals, components


def generate_gaussian_bumps(
    batch_size: int,
    K_range: tuple[int, int] = (1, 4),
    N: int = 256,
    noise_std: float = 0.1,
    dominant: bool = True,
    seed: int | None = None,
) -> GeneratorOutput:
    """Task B: Gaussian bumps ``A exp(-(t - mu)^2 / (2 sigma^2))``.

    Centres ``mu ~ U(0.1, 0.9)``, widths ``sigma ~ U(0.02, 0.07)``, amplitudes as in
    :func:`generate_sinusoids`. Arguments as in :func:`generate_sinusoids`.

    Returns:
        ``params [B, max_K, 3]`` = (center, amplitude, width); other outputs as
        in :func:`generate_sinusoids`.
    """
    _seed(seed)
    B = batch_size
    max_K = K_range[1]
    t = torch.linspace(0, 1, N)

    params = torch.zeros(B, max_K, 3)
    components = torch.zeros(B, max_K, N)
    signals = torch.zeros(B, N)
    K_vals = torch.randint(K_range[0], K_range[1] + 1, (B,))

    for b in range(B):
        for k in range(K_vals[b].item()):
            center = torch.rand(1).item() * 0.8 + 0.1
            width = torch.rand(1).item() * 0.05 + 0.02
            if dominant and k == 0:
                amp = torch.rand(1).item() * 5 + 5
            else:
                amp = torch.rand(1).item() * 0.8 + 0.2

            params[b, k] = torch.tensor([center, amp, width])
            comp = amp * torch.exp(-0.5 * ((t - center) / width) ** 2)
            components[b, k] = comp
            signals[b] += comp

    signals += torch.randn(B, N) * noise_std
    return signals, params, K_vals, components


def generate_multiscale(
    batch_size: int,
    K_range: tuple[int, int] = (1, 4),
    N: int = 256,
    noise_std: float = 0.1,
    dominant: bool = True,
    seed: int | None = None,
) -> GeneratorOutput:
    """Task C: multi-scale sources ``A exp(-(t - mu)^2 / (2 sigma^2)) sin(2 pi f t + phi)``.

    ``f ~ U(3, 18)`` Hz, ``phi ~ U(0, 2 pi)``, ``mu ~ U(0.2, 0.8)``,
    ``sigma ~ U(0.05, 0.15)``, amplitudes as in :func:`generate_sinusoids`.

    Returns:
        ``params [B, max_K, 5]`` = (frequency, amplitude, phase, center, width);
        other outputs as in :func:`generate_sinusoids`.
    """
    _seed(seed)
    B = batch_size
    max_K = K_range[1]
    t = torch.linspace(0, 1, N)

    params = torch.zeros(B, max_K, 5)
    components = torch.zeros(B, max_K, N)
    signals = torch.zeros(B, N)
    K_vals = torch.randint(K_range[0], K_range[1] + 1, (B,))

    for b in range(B):
        for k in range(K_vals[b].item()):
            freq = torch.rand(1).item() * 15 + 3
            phase = torch.rand(1).item() * 2 * np.pi
            center = torch.rand(1).item() * 0.6 + 0.2
            width = torch.rand(1).item() * 0.1 + 0.05
            if dominant and k == 0:
                amp = torch.rand(1).item() * 5 + 5
            else:
                amp = torch.rand(1).item() * 0.8 + 0.2

            params[b, k] = torch.tensor([freq, amp, phase, center, width])
            envelope = torch.exp(-0.5 * ((t - center) / width) ** 2)
            comp = amp * envelope * torch.sin(2 * np.pi * freq * t + phase)
            components[b, k] = comp
            signals[b] += comp

    signals += torch.randn(B, N) * noise_std
    return signals, params, K_vals, components


TASKS: dict[str, dict[str, Any]] = {
    "sinusoids": {
        "generate": generate_sinusoids,
        "n_params": 3,
        "param_names": ["frequency", "amplitude", "phase"],
    },
    "gaussian_bumps": {
        "generate": generate_gaussian_bumps,
        "n_params": 3,
        "param_names": ["center", "amplitude", "width"],
    },
    "multiscale": {
        "generate": generate_multiscale,
        "n_params": 5,
        "param_names": ["frequency", "amplitude", "phase", "center", "width"],
    },
}
"""Task registry: generator, number of parameters per source and their names."""
