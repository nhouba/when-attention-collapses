"""FUSS (Free Universal Sound Separation) data loading (Sec. 3.2, App. J).

FUSS (Wisdom et al., ICASSP 2021; Zenodo record 3743844, CC BY 4.0) contains real
mixtures of 1-4 everyday sounds summed additively at 16 kHz. Expected layout
(the ``ssdata`` directory of the release; see ``experiments/download_fuss.sh``)::

    <fuss_dir>/
        train/       example00000.wav  example00000_sources/{background0,foreground0,...}_sound.wav
        validation/  ...
        eval/        ...

Ground truth per source (3 parameters, all normalized to ``[0, 1]``): dominant
frequency, RMS amplitude and spectral centroid (:func:`extract_params`).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_wav(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Load a 16-bit mono WAV file as ``float32`` in ``[-1, 1)``.

    Files at another sampling rate are resampled by nearest-index decimation.
    """
    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2, "Expected 16-bit audio"
        assert wf.getnchannels() == 1, "Expected mono audio"
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != target_sr:
        ratio = sr / target_sr
        indices = np.round(np.arange(0, len(audio), ratio)).astype(int)
        indices = indices[indices < len(audio)]
        audio = audio[indices]
    return audio


def extract_params(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Spectral ground-truth parameters of one source waveform.

    Computed on the first ``min(len, 2048)`` samples.

    Returns:
        ``float32 [3]``: dominant frequency / Nyquist (DC excluded),
        ``min(RMS / 0.5, 1)``, spectral centroid / Nyquist.
    """
    n_fft = min(len(waveform), 2048)
    spectrum = np.abs(np.fft.rfft(waveform[:n_fft]))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    peak_idx = np.argmax(spectrum[1:]) + 1
    dom_freq = freqs[peak_idx] / (sr / 2)

    rms = np.sqrt(np.mean(waveform**2))
    rms_norm = min(rms / 0.5, 1.0)

    power = spectrum**2
    if power.sum() > 0:
        centroid_norm = (np.sum(freqs * power) / np.sum(power)) / (sr / 2)
    else:
        centroid_norm = 0.0

    return np.array([dom_freq, rms_norm, centroid_norm], dtype=np.float32)


class FUSSDataset(Dataset):
    """FUSS mixtures with per-source spectral parameters.

    Each item is ``(mixture [N], params [max_K, 3], K)``, plus the reference
    sources ``[max_K, N]`` (zero-padded beyond ``K``) when ``return_sources``.
    Mixtures with no source directory, no sources or more than ``max_K``
    sources are skipped.

    Args:
        fuss_dir: FUSS ``ssdata`` directory (contains ``train/``, ``validation/``, ``eval/``).
        split: ``"train"``, ``"validation"`` or ``"eval"``.
        max_K: maximum number of sources.
        signal_length: crop / zero-pad every waveform to this many samples.
        max_samples: keep only the first ``max_samples`` mixtures (sorted by name).
        return_sources: also return the reference source waveforms.
    """

    def __init__(
        self,
        fuss_dir: str | Path,
        split: str = "train",
        max_K: int = 4,
        signal_length: int = 16000,
        max_samples: int | None = None,
        return_sources: bool = False,
    ) -> None:
        self.max_K = max_K
        self.signal_length = signal_length
        self.return_sources = return_sources

        split_dir = Path(fuss_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"FUSS split directory not found: {split_dir}")

        mix_files = sorted(split_dir.glob("example*.wav"))
        mix_files = [
            f for f in mix_files if "_sources" not in str(f.parent.name) and "_sound" not in f.name
        ]
        if max_samples:
            mix_files = mix_files[:max_samples]

        self.items: list[tuple[Path, list[Path], int]] = []
        skipped = 0
        for mix_path in mix_files:
            source_dir = mix_path.parent / f"{mix_path.stem}_sources"
            if not source_dir.exists():
                skipped += 1
                continue
            source_paths = sorted(source_dir.glob("*.wav"))
            K = len(source_paths)
            if K < 1 or K > max_K:
                skipped += 1
                continue
            self.items.append((mix_path, source_paths, K))

        print(
            f"  FUSS {split}: {len(self.items)} mixtures loaded "
            f"(skipped {skipped} with K=0 or K>{max_K})"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        mix_path, source_paths, K = self.items[idx]
        mix = self._crop_pad(load_wav(mix_path))

        params = np.zeros((self.max_K, 3), dtype=np.float32)
        sources = (
            np.zeros((self.max_K, self.signal_length), dtype=np.float32)
            if self.return_sources
            else None
        )
        for k, sp in enumerate(source_paths):
            src = self._crop_pad(load_wav(sp))
            params[k] = extract_params(src)
            if sources is not None:
                sources[k] = src

        item: tuple[torch.Tensor, ...] = (
            torch.from_numpy(mix),
            torch.from_numpy(params),
            torch.tensor(K, dtype=torch.long),
        )
        if sources is not None:
            item = item + (torch.from_numpy(sources),)
        return item

    def _crop_pad(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) >= self.signal_length:
            return audio[: self.signal_length]
        padded = np.zeros(self.signal_length, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded
