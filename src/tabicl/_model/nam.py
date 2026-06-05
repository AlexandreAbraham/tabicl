"""NAM input encoder for TabICL.

A learnable preprocessing layer that sits before ``col_embedder``. Takes a
raw batched table ``(B, T, H)`` and projects each feature through a per-
feature MLP (continuous columns) or a small embedding (categorical columns,
when categorical indices are provided), then optionally mixes the per-feature
representations before emitting ``(B, T, H_out)``.

This is a native ``nn.Module`` so plain back-propagation through ``TabICL``
updates its parameters — no differentiable-input path needed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NAMEncoder(nn.Module):
    """Per-feature MLP encoder for TabICL inputs.

    Each of the ``H`` input features is passed through its own MLP that maps
    a scalar to ``n_channels`` channels. The per-feature outputs are then
    either passed through a dense mixer (``mix=True``) to produce ``H_out``
    learned features, or concatenated and returned as a `(B, T, H * n_channels)`
    tensor (``mix=False``).

    Categorical columns are handled by callers that pre-encode them as
    integer ids and pass ``cat_idx`` so the corresponding features bypass
    the numeric MLP and go through an ``nn.Embedding`` instead.

    Parameters
    ----------
    n_features : int
        ``H`` — number of input columns.
    cat_idx : list[int] or None, default=None
        Indices of categorical columns. The remaining columns are treated as
        continuous. If ``None``, all columns are continuous.
    cat_cardinalities : list[int] or None, default=None
        Cardinality of each categorical column (one entry per ``cat_idx``).
        Required if ``cat_idx`` is provided.
    hidden : int, default=32
        Hidden width of each per-feature MLP.
    n_channels : int, default=8
        Output channels per feature.
    n_layers : int, default=2
        Total Linear layers in each per-feature MLP (>=1).
    cat_emb_dim : int, default=16
        Max embedding dim for categorical features. Effective dim is
        ``min(cat_emb_dim, sqrt(card)+1)``, capped at ``n_channels`` via
        projection.
    mix : bool, default=True
        If True, append a dense mixer that maps the concatenated per-feature
        outputs to ``H_out`` learned features. If False, the encoder returns
        the concatenated per-feature outputs (``H_out = H * n_channels``).
    h_out : int or None, default=None
        Output feature count when ``mix=True``. Defaults to ``n_features``,
        which is the lowest-impact choice (col_embedder sees the same H).
    dropout : float, default=0.0
        Dropout applied after the mixer.
    """

    def __init__(
        self,
        n_features: int,
        cat_idx: Optional[list[int]] = None,
        cat_cardinalities: Optional[list[int]] = None,
        hidden: int = 32,
        n_channels: int = 8,
        n_layers: int = 2,
        cat_emb_dim: int = 16,
        mix: bool = True,
        h_out: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if cat_idx is None:
            cat_idx = []
        if cat_cardinalities is None:
            cat_cardinalities = []
        if len(cat_idx) != len(cat_cardinalities):
            raise ValueError("cat_idx and cat_cardinalities must have same length")
        if any(i < 0 or i >= n_features for i in cat_idx):
            raise ValueError(f"cat_idx out of range for n_features={n_features}")

        self.n_features = n_features
        self.n_channels = n_channels
        self.mix = mix
        self.h_out = h_out if h_out is not None else n_features

        cat_set = set(cat_idx)
        num_idx = [i for i in range(n_features) if i not in cat_set]
        # Persistent buffers so .to(device) / .cuda() carries them.
        self.register_buffer("num_idx", torch.tensor(num_idx, dtype=torch.long))
        self.register_buffer("cat_idx", torch.tensor(cat_idx, dtype=torch.long))

        def _make_num_mlp() -> nn.Sequential:
            layers: list[nn.Module] = [nn.Linear(1, hidden), nn.ReLU()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            layers.append(nn.Linear(hidden, n_channels))
            return nn.Sequential(*layers)

        self.num_mlps = nn.ModuleList([_make_num_mlp() for _ in num_idx])

        self.cat_embs = nn.ModuleList()
        self.cat_projs = nn.ModuleList()
        for card in cat_cardinalities:
            actual_emb = max(2, min(cat_emb_dim, int(card ** 0.5) + 1))
            emb = nn.Embedding(card, actual_emb)
            nn.init.normal_(emb.weight, std=0.05)
            self.cat_embs.append(emb)
            self.cat_projs.append(nn.Linear(actual_emb, n_channels))

        if mix:
            if self.h_out != n_features:
                raise ValueError(
                    f"mix=True with bypass requires h_out == n_features; "
                    f"got h_out={self.h_out}, n_features={n_features}"
                )
            self.mixer = nn.Linear(n_features * n_channels, self.h_out)
            # Near-identity start: mixer is small-random (gradients can flow
            # back to per-feature MLPs) but the gate is zero so the forward
            # output exactly equals X at init.
            nn.init.normal_(self.mixer.weight, std=0.02)
            nn.init.zeros_(self.mixer.bias)
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.mixer = None
            self.gate = None
            self.h_out = n_features * n_channels
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, X: Tensor) -> Tensor:
        """Encode the batched table.

        Parameters
        ----------
        X : Tensor
            Shape ``(B, T, H)``. Numeric columns are treated as floats; categorical
            columns are expected to hold integer ids in ``[0, cardinality)``.

        Returns
        -------
        Tensor
            Shape ``(B, T, self.h_out)``.
        """
        if X.dim() != 3:
            raise ValueError(f"Expected (B, T, H), got shape {tuple(X.shape)}")
        B, T, H = X.shape
        if H != self.n_features:
            raise ValueError(f"NAMEncoder built for H={self.n_features}, got H={H}")

        parts: list[Tensor] = [None] * self.n_features  # type: ignore[list-item]

        # Numeric columns
        num_idx = self.num_idx.tolist()
        for k, j in enumerate(num_idx):
            xj = X[..., j : j + 1].reshape(-1, 1).float()      # (B*T, 1)
            hj = self.num_mlps[k](xj)                          # (B*T, n_channels)
            parts[j] = hj.reshape(B, T, self.n_channels)

        # Categorical columns
        cat_idx = self.cat_idx.tolist()
        for k, j in enumerate(cat_idx):
            ids = X[..., j].reshape(-1).long()                 # (B*T,)
            ids = ids.clamp_(min=0, max=self.cat_embs[k].num_embeddings - 1)
            emb = self.cat_embs[k](ids)                        # (B*T, actual_emb)
            hj = self.cat_projs[k](emb)                        # (B*T, n_channels)
            parts[j] = hj.reshape(B, T, self.n_channels)

        # Concat along feature axis: (B, T, H * n_channels)
        h = torch.cat(parts, dim=-1)

        if self.mixer is not None:
            # Gated residual: start as identity (gate=0), learn a non-trivial
            # mixing of per-feature MLP outputs over training.
            delta = self.mixer(h)
            delta = self.dropout(delta)
            return X + self.gate * delta
        # No-mix branch: return concatenated per-feature representations
        # without identity preservation (caller asked for an expanded width).
        return self.dropout(h)
