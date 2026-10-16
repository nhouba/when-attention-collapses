#!/usr/bin/env python3
"""Model-free oracle check of the FUSS attention-to-mask readout (App. J, "Separation diagnostics").

Can the readout family of :func:`evidence_depletion.separation.slot_separation_readout`
separate FUSS mixtures *at all*? For real mixtures we compute, per source
(identity assignment, no PIT needed for oracles), the SI-SDR of:

* ``floor`` -- the raw mixture (the baseline to beat);
* ``IRM`` -- the ideal ratio mask at full STFT resolution, the upper bound of
  any time-frequency masking method (data separability);
* ``rank1_oracle`` -- the best mask within the readout's family
  ``W_s(f, t) = a_s(t) p_s(f)`` given ground truth: ``a_s`` the source energy
  envelope at full frame resolution, ``p_s`` the source's mean magnitude
  spectrum; masks ratio-normalized across sources as in the readout;
* ``rank1_chunk`` -- the same with ``a_s`` quantized to ``--n-chunks`` time chunks
  (the readout's actual time resolution);
* ``rank1_model_faithful`` -- the readout's own construction: ``a_s`` the
  (chunk-quantized) oracle time envelope, ``p_s`` the ``a_s``-weighted mixture
  power, exactly as the model computes it. This is the ceiling of the model's
  readout for *any* attention.

If ``rank1_model_faithful`` is far above ``floor``, the readout is capable and
near-floor SI-SDR of a trained model is a statement about its attention.

``--n-mixtures`` defaults to 300, the setting behind the paper's +7.7 dB.

Usage::

    python experiments/fuss_sisdr_oracle.py --fuss-dir data/fuss/ssdata
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from evidence_depletion.data.fuss import FUSSDataset
from evidence_depletion.metrics import si_sdr_db
from evidence_depletion.separation import SILENCE_THRESHOLD, SISDR_HOP, SISDR_N_FFT

N_FFT, HOP = SISDR_N_FFT, SISDR_HOP


def stft(x: torch.Tensor, dev: str) -> torch.Tensor:
    w = torch.hann_window(N_FFT, device=dev)
    return torch.stft(x, n_fft=N_FFT, hop_length=HOP, window=w, center=True, return_complex=True)


def istft(Y: torch.Tensor, N: int, dev: str) -> torch.Tensor:
    w = torch.hann_window(N_FFT, device=dev)
    return torch.istft(Y, n_fft=N_FFT, hop_length=HOP, window=w, center=True, length=N)


def apply_masks_sisdr(
    masks: torch.Tensor, X: torch.Tensor, refs: torch.Tensor, N: int, dev: str
) -> torch.Tensor:
    """Per-source SI-SDR of ``masks [K, F, T]`` applied to the mixture STFT ``X``."""
    est = istft(masks * X.unsqueeze(0), N, dev)
    return torch.stack([si_sdr_db(est[k], refs[k]) for k in range(refs.shape[0])])


def _time_envelope(mag: torch.Tensor, n_chunks: int | None) -> torch.Tensor:
    """Per-source energy over frames ``[K, T]``, optionally quantized to ``n_chunks``."""
    a = (mag**2).sum(dim=1)
    if n_chunks:
        a_c = F.adaptive_avg_pool1d(a.unsqueeze(1), n_chunks).squeeze(1)
        a = F.interpolate(
            a_c.unsqueeze(1), size=mag.shape[-1], mode="linear", align_corners=False
        ).squeeze(1)
    return a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _partition(a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rank-1 masks ``a(t) p(f)``, ratio-normalized across sources per TF bin."""
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    W = a.unsqueeze(1) * p.unsqueeze(-1)
    return W / W.sum(dim=0, keepdim=True).clamp(min=1e-8)


def rank1_masks(
    refs: torch.Tensor, X: torch.Tensor, dev: str, n_chunks: int | None = None
) -> torch.Tensor:
    """Best rank-1 masks from ground-truth envelopes and per-source mean spectra."""
    mag = torch.stack([stft(refs[k], dev) for k in range(refs.shape[0])]).abs()
    return _partition(_time_envelope(mag, n_chunks), mag.mean(dim=-1))


def rank1_model_faithful(
    refs: torch.Tensor, X: torch.Tensor, dev: str, n_chunks: int | None = None
) -> torch.Tensor:
    """Rank-1 masks built like the model's readout: oracle time envelope, spectrum
    from the envelope-weighted *mixture* power."""
    mag = torch.stack([stft(refs[k], dev) for k in range(refs.shape[0])]).abs()
    a = _time_envelope(mag, n_chunks)
    power = (X.abs() ** 2).unsqueeze(0)
    return _partition(a, (a.unsqueeze(1) * power).sum(dim=-1))


def irm(refs: torch.Tensor, X: torch.Tensor, dev: str) -> torch.Tensor:
    """Ideal ratio mask at full STFT resolution."""
    power = torch.stack([stft(refs[k], dev) for k in range(refs.shape[0])]).abs() ** 2
    return power / power.sum(dim=0, keepdim=True).clamp(min=1e-8)


def main(argv: list[str] | None = None) -> dict[str, float]:
    ap = argparse.ArgumentParser(description="Oracle check of the FUSS mask readout")
    ap.add_argument("--fuss-dir", required=True)
    ap.add_argument("--signal-length", type=int, default=64000)
    ap.add_argument("--n-mixtures", type=int, default=300)
    ap.add_argument("--n-chunks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FUSSDataset(
        args.fuss_dir, split="eval", max_K=4, signal_length=args.signal_length, return_sources=True
    )
    idx = np.random.default_rng(args.seed).permutation(len(ds))

    estimators = ["floor", "IRM", "rank1_oracle", "rank1_chunk", "rank1_model_faithful"]
    acc: dict[str, list[float]] = {k: [] for k in estimators}
    n = 0
    for i in idx:
        mix, _params, K, sources = ds[int(i)]
        mix = mix.to(dev)
        refs = sources[: int(K)].to(dev)
        refs = refs[refs.pow(2).mean(dim=-1) > SILENCE_THRESHOLD]
        if refs.shape[0] < 2:
            continue
        N = mix.shape[0]
        X = stft(mix, dev)
        acc["floor"].append(
            torch.stack([si_sdr_db(mix, refs[k]) for k in range(refs.shape[0])]).mean().item()
        )
        masks = {
            "IRM": irm(refs, X, dev),
            "rank1_oracle": rank1_masks(refs, X, dev),
            "rank1_chunk": rank1_masks(refs, X, dev, args.n_chunks),
            "rank1_model_faithful": rank1_model_faithful(refs, X, dev, args.n_chunks),
        }
        for name, m in masks.items():
            acc[name].append(apply_masks_sisdr(m, X, refs, N, dev).mean().item())
        n += 1
        if n >= args.n_mixtures:
            break

    notes = {
        "floor": "mixture baseline (to beat)",
        "IRM": "full-res upper bound (data separability)",
        "rank1_oracle": "rank-1 with per-source spectra (stronger than model)",
        "rank1_chunk": f"rank-1 per-source spectra @ {args.n_chunks} chunks",
        "rank1_model_faithful": "model's readout ceiling (p_s from time-weighted mixture)",
    }
    print(f"\n=== FUSS SI-SDR readout oracle check (n={n} mixtures, K>=2 non-silent) ===")
    print(f"{'estimator':22}{'mean dB':>10}{'std':>8}   interpretation")
    means = {}
    for k in estimators:
        v = np.array(acc[k])
        means[k] = float(v.mean()) if n else float("nan")
        print(f"{k:22}{means[k]:>10.3f}{(v.std() if n else float('nan')):>8.3f}   {notes[k]}")
    fl, irm_mean, mf = means["floor"], means["IRM"], means["rank1_model_faithful"]
    print(f"\nGAP IRM - floor                  = {irm_mean - fl:+.2f} dB  (data separability)")
    print(
        f"GAP rank1_model_faithful - floor = {mf - fl:+.2f} dB  (best the model's readout can do)"
    )
    return means


if __name__ == "__main__":
    main()
