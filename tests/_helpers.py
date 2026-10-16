"""Test helpers: a tiny FUSS-layout directory and access to the experiment CLIs."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import wave
from pathlib import Path
from types import ModuleType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def write_wav(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    """Write ``audio`` (float in [-1, 1)) as 16-bit mono PCM."""
    pcm = np.clip(np.round(audio * 32767), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def make_fake_fuss(root: Path, n_per_split: int = 6, length: int = 2400, seed: int = 0) -> Path:
    """Create a FUSS-like ``ssdata`` tree with random tonal sources.

    Each split gets ``n_per_split`` valid mixtures with 1-4 sources, plus one
    mixture with 5 sources and one without a source directory (both must be
    skipped). One source per split is stored at 32 kHz to exercise resampling.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length) / 16000
    for split in ("train", "validation", "eval"):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        for i in range(n_per_split + 2):
            stem = f"example{i:05d}"
            K = 5 if i == n_per_split else 1 + i % 4
            if i == n_per_split + 1:
                write_wav(split_dir / f"{stem}.wav", 0.1 * rng.standard_normal(length))
                continue
            src_dir = split_dir / f"{stem}_sources"
            src_dir.mkdir()
            mix = np.zeros(length)
            for k in range(K):
                f0 = rng.uniform(100, 3000)
                env = np.exp(-(((t - rng.uniform(0, t[-1])) / 0.05) ** 2))
                src = rng.uniform(0.05, 0.3) * env * np.sin(2 * np.pi * f0 * t)
                src += 0.002 * rng.standard_normal(length)
                name = ("background" if k == 0 else "foreground") + f"{k}_sound.wav"
                if k == 1 and i == 1:
                    t32 = np.arange(2 * length) / 32000
                    write_wav(src_dir / name, 0.2 * np.sin(2 * np.pi * f0 * t32), sr=32000)
                else:
                    write_wav(src_dir / name, src)
                mix += src
            write_wav(split_dir / f"{stem}.wav", mix)
    return root


def load_experiment(name: str) -> ModuleType:
    """Import ``experiments/<name>.py`` under a unique module name."""
    spec = importlib.util.spec_from_file_location(
        f"ed_experiment_{name}", EXPERIMENTS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run ``experiments/<script>`` with the current interpreter; fail on non-zero exit."""
    proc = subprocess.run(
        [sys.executable, str(EXPERIMENTS_DIR / script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc
