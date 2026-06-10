"""Noise-conditioning FiLM adapter for TabICL.

Sits between ``col_embedder`` and ``row_interactor``. Takes the column
embeddings ``(B, T, G+C, E)`` and applies a learnable FiLM modulation
conditioned on a **per-row** noise scalar (shape ``(B, T)``). Each row's
embedding gets its own (scale, shift), so the adapter can carry
fine-grained per-row signal from a teacher model (LGBM uncertainty,
residual magnitude, anomaly score, etc.).

The FiLM is gated to start near identity, so a freshly initialized adapter
inserted into a pretrained TabICL has near-zero impact on its outputs.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class NoiseFilm(nn.Module):
    """Gated, per-row FiLM modulation on ``(B, T, G+C, E)`` embeddings.

    Conditioned on a per-row scalar ``noise`` of shape ``(B, T)``. Each row's
    embedding gets its own (scale, shift) produced by the conditioning MLP.
    The per-table case (one scalar per table) can be passed as ``(B,)`` or
    ``(B, 1)`` and is broadcast across rows.

    A passed ``noise=None`` triggers a runtime error: an unconditioned FiLM
    has no signal to learn from and would silently no-op (its forward output
    exactly equals the input at init *and* during training since the input
    to ``cond`` is constant zero, the only gradient path is through the
    scalar gate which then has zero gradient because ``modulated == input``).
    Callers who want a learnable unconditional adapter should use NAM, not
    NoiseFilm.

    Initialization
    --------------
    The output gate is zero-initialized so the adapter behaves like identity
    at the start of training. The conditioning MLP's output projection is
    small-random so the modulation branch has non-zero magnitude — this
    gives the gate a non-zero gradient and lets it learn away from zero.

    Parameters
    ----------
    embed_dim : int
        ``E`` — embedding dim of the col_embedder output.
    hidden : int, default=32
        Hidden width of the conditioning MLP.
    """

    def __init__(self, embed_dim: int, hidden: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        # Conditioning MLP: maps a per-row scalar -> (scale, shift) of size E
        self.cond = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * embed_dim),
        )
        nn.init.normal_(self.cond[-1].weight, std=0.02)
        nn.init.zeros_(self.cond[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, embeddings: Tensor, noise: Tensor) -> Tensor:
        """Apply per-row FiLM to embeddings.

        Parameters
        ----------
        embeddings : Tensor
            Shape ``(B, T, G+C, E)``.
        noise : Tensor
            Per-row conditioning signal. Accepted shapes:

            * ``(B, T)`` — one scalar per row (the intended use).
            * ``(B,)`` or ``(B, 1)`` — one scalar per table, broadcast over rows.

        Returns
        -------
        Tensor
            Same shape as ``embeddings``.
        """
        if noise is None:
            raise ValueError(
                "NoiseFilm requires a per-row noise tensor (shape (B, T)). "
                "Passing noise=None has no learnable effect — use NAM if you "
                "want an unconditional adapter."
            )

        B, T = embeddings.shape[0], embeddings.shape[1]
        # Keep noise in fp32 for the cond MLP. The MLP weights are fp32 and
        # we cast its outputs back to embeddings.dtype below — this avoids
        # AMP dtype mismatches (fp16 embeddings × fp32 cond weights would
        # raise inside the linear matmul).
        noise = noise.float()

        # Normalize to (B, T) — accept (B,) or (B, 1) as per-table broadcasts.
        if noise.dim() == 1:                               # (B,)
            if noise.shape[0] != B:
                raise ValueError(f"noise (B,) got {noise.shape[0]} != B={B}")
            noise = noise.view(B, 1).expand(B, T)
        elif noise.dim() == 2 and noise.shape[1] == 1:     # (B, 1)
            noise = noise.expand(B, T)
        elif noise.dim() == 2 and noise.shape == (B, T):
            pass
        else:
            raise ValueError(
                f"noise must be (B,), (B, 1), or (B, T); got {tuple(noise.shape)} "
                f"(B={B}, T={T})"
            )

        # Compute per-row (scale, shift) in fp32 to match the cond MLP's
        # parameters. We promote ``embeddings`` to fp32 for the FiLM ops so
        # the autograd graph stays connected to the cond/gate parameters
        # under AMP, and cast back at the end.
        scale_shift = self.cond(noise.unsqueeze(-1))       # (B, T, 2*E) fp32
        scale, shift = scale_shift.chunk(2, dim=-1)        # each (B, T, E)
        scale = scale.unsqueeze(2)                          # (B, T, 1, E)
        shift = shift.unsqueeze(2)                          # (B, T, 1, E)

        in_dtype = embeddings.dtype
        emb_f = embeddings.float()
        modulated = emb_f * (1.0 + scale) + shift
        out = emb_f + self.gate * (modulated - emb_f)
        return out.to(in_dtype)
