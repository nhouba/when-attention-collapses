"""Parameter-free attention-to-mask readout and PIT SI-SDR evaluation (App. J).

The :class:`~evidence_depletion.models.ChunkFFTTokenizer` splits the waveform
into ``L`` non-overlapping time chunks, so a slot's attention over tokens is a
per-slot *time* profile at chunk resolution. :func:`slot_separation_readout`
turns it into a rank-1 soft time-frequency mask per slot:

1. ``a_s(t)``: the slot's attention renormalized to a distribution over tokens
   and linearly interpolated from chunk centres to the STFT frame grid.
2. ``p_s(f)``: the ``a_s(t)``-weighted average of the mixture power spectrogram.
3. ``W_s(f, t) = a_s(t) p_s(f)``, zeroed for inactive slots (existence
   ``sigmoid < 0.5``; the top-1 slot is always kept), ratio-normalized across
   slots so that the masks sum to one per TF bin.

``mask_mode="time"`` drops step 2 (pure time gating). Neither mode has
trainable parameters.

SI-SDR policy (:func:`evaluate_sisdr`): references with mean-square power below
:data:`SILENCE_THRESHOLD` are skipped, mixtures with fewer than two non-silent
references are excluded, estimates are the active slots, and PIT matches
``min(K_nonsilent, K_est)`` pairs; metrics are averaged over matched pairs.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evidence_depletion.metrics import pit_si_sdr, si_sdr_db

SILENCE_THRESHOLD = 1e-6
"""Mean-square power below which a reference source counts as silent."""
SISDR_N_FFT = 1024
SISDR_HOP = 256


def slot_separation_readout(
    mixture: torch.Tensor,
    attn: torch.Tensor,
    exist_logits: torch.Tensor,
    n_fft: int = SISDR_N_FFT,
    hop: int = SISDR_HOP,
    mask_mode: str = "rank1",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-slot soft TF masks applied to the mixture.

    Args:
        mixture: ``[B, N]`` waveforms.
        attn: ``[B, S, L]`` slot attention over chunk tokens (any normalization).
        exist_logits: ``[B, S]``.
        mask_mode: ``"rank1"`` (time envelope x attended spectrum) or ``"time"``.

    Returns:
        ``(est [B, S, N], active [B, S])``; inactive slots yield zero estimates.
    """
    B, N = mixture.shape
    S = attn.shape[1]
    dev = mixture.device

    window = torch.hann_window(n_fft, device=dev)
    X = torch.stft(
        mixture, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True
    )
    n_freq, n_frames = X.shape[-2], X.shape[-1]

    a = attn.clamp(min=0)
    a = a / a.sum(dim=-1, keepdim=True).clamp(min=eps)
    a_t = F.interpolate(a, size=n_frames, mode="linear", align_corners=False)

    active = torch.sigmoid(exist_logits) > 0.5
    none_active = ~active.any(dim=1)
    if none_active.any():
        top1 = exist_logits.argmax(dim=1)
        active[none_active, top1[none_active]] = True

    if mask_mode == "rank1":
        power = (X.abs() ** 2).unsqueeze(1)
        p = (a_t.unsqueeze(2) * power).sum(dim=-1)
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=eps)
        W = a_t.unsqueeze(2) * p.unsqueeze(-1)
    elif mask_mode == "time":
        W = a_t.unsqueeze(2).expand(B, S, n_freq, n_frames)
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")

    W = W * active.float().view(B, S, 1, 1)
    masks = W / W.sum(dim=1, keepdim=True).clamp(min=eps)

    Y = masks * X.unsqueeze(1)
    est = torch.istft(
        Y.reshape(B * S, n_freq, n_frames),
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        center=True,
        length=N,
    ).reshape(B, S, N)
    return est, active


@torch.no_grad()
def evaluate_sisdr(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    mask_mode: str = "rank1",
) -> dict[str, float | int | None]:
    """PIT SI-SDR of the mask readout over a loader yielding ``(mix, params, K, sources)``.

    Returns:
        ``si_sdr``, ``si_sdri`` (improvement over the mixture) and
        ``si_sdr_mixture`` in dB (means over matched pairs, ``None`` if nothing
        was evaluated) and ``n_mixtures``.
    """
    model.eval()
    sisdr_vals, sisdri_vals, mix_vals = [], [], []
    n_mixtures = 0

    for signals, _params, K_vals, sources in loader:
        signals = signals.to(device)
        _, attn, exist_logits, _, _ = model(signals)
        est, active = slot_separation_readout(signals, attn, exist_logits, mask_mode=mask_mode)
        est = est.cpu()
        active = active.cpu()
        signals_cpu = signals.cpu()

        for b in range(signals.shape[0]):
            K = int(K_vals[b].item())
            refs = sources[b, :K]
            refs = refs[refs.pow(2).mean(dim=-1) > SILENCE_THRESHOLD]
            if refs.shape[0] < 2:
                continue
            ests_b = est[b][active[b]]
            if ests_b.shape[0] == 0:
                continue

            _, matched = pit_si_sdr(ests_b, refs)
            for ri, _ei, val in matched:
                base = float(si_sdr_db(signals_cpu[b], refs[ri]))
                sisdr_vals.append(val)
                mix_vals.append(base)
                sisdri_vals.append(val - base)
            n_mixtures += 1

    if n_mixtures == 0:
        return {"si_sdr": None, "si_sdri": None, "si_sdr_mixture": None, "n_mixtures": 0}
    return {
        "si_sdr": round(float(np.mean(sisdr_vals)), 3),
        "si_sdri": round(float(np.mean(sisdri_vals)), 3),
        "si_sdr_mixture": round(float(np.mean(mix_vals)), 3),
        "n_mixtures": n_mixtures,
    }
