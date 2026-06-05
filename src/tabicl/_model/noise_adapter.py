"""Noise-conditioning FiLM adapter for TabICL.

Sits between ``col_embedder`` and ``row_interactor``. Takes the column
embeddings ``(B, T, G+C, E)`` and applies a learnable per-element FiLM
modulation conditioned on a global per-table noise scalar (or, if you want
to use it for unconditional fine-tuning, a fixed zero vector — the adapter
still has trainable parameters through its gates).

The FiLM is gated to start near identity, so a freshly initialized adapter
inserted into a pretrained TabICL has near-zero impact on its outputs.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class NoiseFilm(nn.Module):
    """Gated FiLM modulation on (B, T, G+C, E) embeddings.

    Conditioned on a scalar ``noise`` per table (shape ``(B,)``). If ``noise``
    is ``None``, a zero scalar is broadcast — the adapter still has gradient
    flow through its gate and scale/shift heads.

    Initialization
    --------------
    The output gate is zero-initialized so the adapter behaves like identity
    at the start of training. As ``gate`` learns away from zero the FiLM kicks
    in.

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
        # Conditioning MLP: maps noise scalar -> (scale, shift) of size embed_dim
        self.cond = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * embed_dim),
        )
        # Near-identity start: cond output is small-random so the modulation
        # branch has nonzero magnitude (so gate gradient is nonzero), but the
        # gate is zero-init'd so the forward output equals the input at init.
        nn.init.normal_(self.cond[-1].weight, std=0.02)
        nn.init.zeros_(self.cond[-1].bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, embeddings: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """Apply FiLM to embeddings.

        Parameters
        ----------
        embeddings : Tensor
            Shape ``(B, T, G+C, E)``.
        noise : Tensor or None, default=None
            Shape ``(B,)`` or ``(B, 1)``. If ``None``, treated as zeros.

        Returns
        -------
        Tensor
            Same shape as ``embeddings``.
        """
        B = embeddings.shape[0]
        if noise is None:
            noise_in = embeddings.new_zeros(B, 1)
        else:
            noise_in = noise.view(B, 1).to(embeddings.dtype)

        scale_shift = self.cond(noise_in)               # (B, 2*E)
        scale, shift = scale_shift.chunk(2, dim=-1)     # each (B, E)
        # Broadcast to (B, 1, 1, E) for the (B, T, G+C, E) tensor.
        scale = scale.view(B, 1, 1, self.embed_dim)
        shift = shift.view(B, 1, 1, self.embed_dim)

        modulated = embeddings * (1.0 + scale) + shift
        # Gate between identity and the modulated path.
        return embeddings + self.gate * (modulated - embeddings)
