"""
Argon Persistent In-SRAM Forward-Forward & Hebbian Micro-Kernel (Triton).
Eliminates HBM round-trips during continuous physical control loops by keeping
recurrent synaptic weights pinned inside SM registers/SRAM across multi-step horizons.
Includes automatic NumPy/CPU deterministic simulation fallback when CUDA/Triton is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TRITON_AVAILABLE = False


@dataclass(frozen=True)
class TritonPersistentConfig:
    """Configuration for the persistent in-SRAM micro-kernel."""

    tile_dim: int = 64
    horizon_steps: int = 16
    learning_rate: float = 0.025
    goodness_threshold: float = 2.0
    weight_decay: float = 0.0005


if TRITON_AVAILABLE:

    @triton.jit
    def _argon_persistent_sram_ff_kernel(
        stream_in_ptr,
        weights_ptr,
        stream_out_ptr,
        goodness_out_ptr,
        polarity_ptr,
        horizon_steps: int,
        lr: float,
        threshold: float,
        weight_decay: float,
        BLOCK_DIM: tl.constexpr,
    ) -> None:
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

        w_sram = tl.load(weights_ptr + offs)

        for step_idx in range(horizon_steps):
            step_offset = step_idx * BLOCK_DIM + offs
            x_t = tl.load(stream_in_ptr + step_offset)
            polarity = tl.load(polarity_ptr + step_idx)

            z_t = w_sram * x_t
            y_t = tl.maximum(z_t, 0.0)

            g_t = tl.sum(y_t * y_t, axis=0) / BLOCK_DIM
            tl.store(goodness_out_ptr + step_idx, g_t)

            margin = polarity * (threshold - g_t)
            modulation = tl.sigmoid(margin)
            effective_lr = lr * polarity * modulation

            delta_w = effective_lr * y_t * x_t - (weight_decay * w_sram)
            w_sram = w_sram + delta_w

            inv_rms = 1.0 / (tl.sqrt(g_t) + 1e-6)
            tl.store(stream_out_ptr + step_offset, y_t * inv_rms)

        tl.store(weights_ptr + offs, w_sram)


def execute_persistent_horizon_numpy(
    stream_in: np.ndarray,
    weights: np.ndarray,
    polarities: np.ndarray,
    config: TritonPersistentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic CPU/ARM64 fallback matching the Triton persistent kernel semantics."""
    if stream_in.shape != (config.horizon_steps, config.tile_dim):
        raise ValueError(
            f"Expected stream_in shape {(config.horizon_steps, config.tile_dim)}, got {stream_in.shape}"
        )
    if weights.shape != (config.tile_dim,):
        raise ValueError(f"Expected weights shape {(config.tile_dim,)}, got {weights.shape}")
    if polarities.shape != (config.horizon_steps,):
        raise ValueError(
            f"Expected polarities shape {(config.horizon_steps,)}, got {polarities.shape}"
        )

    w_sram = weights.astype(np.float32).copy()
    stream_out = np.zeros_like(stream_in, dtype=np.float32)
    goodness_trace = np.zeros((config.horizon_steps,), dtype=np.float32)

    for step_idx in range(config.horizon_steps):
        x_t = stream_in[step_idx]
        polarity = float(polarities[step_idx])

        y_t = np.maximum(w_sram * x_t, 0.0)
        g_t = float(np.mean(y_t * y_t))
        goodness_trace[step_idx] = g_t

        margin = polarity * (config.goodness_threshold - g_t)
        modulation = 1.0 / (1.0 + math.exp(-margin))
        effective_lr = config.learning_rate * polarity * modulation

        w_sram = w_sram + (effective_lr * y_t * x_t) - (config.weight_decay * w_sram)
        inv_rms = 1.0 / (math.sqrt(g_t) + 1e-6)
        stream_out[step_idx] = y_t * inv_rms

    return w_sram, stream_out, goodness_trace


if __name__ == "__main__":
    cfg = TritonPersistentConfig(tile_dim=64, horizon_steps=8)
    rng = np.random.default_rng(seed=2026)
    sample_stream = rng.uniform(0.1, 1.0, size=(cfg.horizon_steps, cfg.tile_dim)).astype(
        np.float32
    )
    init_w = np.full((cfg.tile_dim,), 0.35, dtype=np.float32)
    phase_pol = np.ones((cfg.horizon_steps,), dtype=np.float32)

    updated_w, out_acts, g_trace = execute_persistent_horizon_numpy(
        sample_stream, init_w, phase_pol, cfg
    )
    print(
        f"[Argon Triton Kernel] Horizon complete | Initial G: {g_trace[0]:.4f} -> Final G: {g_trace[-1]:.4f}"
    )
