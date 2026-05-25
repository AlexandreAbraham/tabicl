"""TFM-Retouche: lightweight input-space residual adapter for TabICL.

Implementation of Nguyen et al., "TFM-Retouche: A Lightweight Input-Space Adapter
for Tabular Foundation Models" (arXiv:2605.06047v2, May 2026), instantiated on
TabICLv2 as the frozen backbone.

The wrapper differs from the paper's default preprocessing in one place: by default
it uses learnable per-column entity embeddings for categorical features (jointly
optimised with the adapter) instead of plain ordinal+scaled. The original
ordinal-scaled and one-hot variants remain available via ``cat_encoder``.

When the post-training identity guard fires, the wrapper falls back to a clean
``TabICLClassifier`` on raw inputs — entity embeddings and the adapter are both
discarded, matching the paper's full-passthrough interpretation of the guard.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .classifier import TabICLClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Inner blocks (δ)
# ─────────────────────────────────────────────────────────────────────────────


class CrossBlock(nn.Module):
    """DCNv2 cross block (Eq. 2): x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l.

    Low-rank parameterisation W_l = U_l V_l^T with rank h = ⌊r·d⌋, optional
    activation between V and U. Initialised so x_L = x_0 (W weights near zero).
    """

    def __init__(self, d: int, num_layers: int, low_rank_ratio: Optional[float],
                 activation: Optional[str], use_batch_norm: bool, norm_type: str = "batch"):
        super().__init__()
        self.num_layers = num_layers
        self.low_rank = low_rank_ratio is not None
        h = max(1, int(low_rank_ratio * d)) if self.low_rank else d
        self.h = h
        self.act = _resolve_activation(activation)
        self.use_bn = use_batch_norm
        self.norm_type = norm_type

        self.U = nn.ParameterList()
        self.V = nn.ParameterList()
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        self.bn = nn.ModuleList()

        for _ in range(num_layers):
            if self.low_rank:
                U = nn.Parameter(torch.empty(d, h))
                V = nn.Parameter(torch.empty(d, h))
                # near-identity for the residual stack: zero out U so W = 0
                nn.init.normal_(V, std=0.02)
                nn.init.zeros_(U)
                self.U.append(U)
                self.V.append(V)
            else:
                W = nn.Parameter(torch.zeros(d, d))
                self.W.append(W)
            self.b.append(nn.Parameter(torch.zeros(d)))
            self.bn.append(_make_norm(d, use_batch_norm, norm_type))

    def _Wx(self, x: torch.Tensor, l: int) -> torch.Tensor:
        if self.low_rank:
            z = x @ self.V[l]
            if self.act is not None:
                z = self.act(z)
            return z @ self.U[l].T
        return x @ self.W[l].T

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for l in range(self.num_layers):
            inner = self._Wx(x, l) + self.b[l]
            inner = _safe_bn(self.bn[l], inner)
            x = x0 * inner + x
        return x


class ResMLPBlock(nn.Module):
    """Residual-bottleneck MLP block (Eq. 3):
    x_{l+1} = x_l + W²_l σ(W¹_l x_l + b¹_l) + b²_l.

    Hidden width h = max(h_min, ⌊r·d⌋). Initialised so W² = 0 → near-identity.
    """

    def __init__(self, d: int, num_layers: int, low_rank_ratio: Optional[float],
                 activation: Optional[str], use_batch_norm: bool, h_min: int = 2,
                 norm_type: str = "batch"):
        super().__init__()
        self.num_layers = num_layers
        h = max(h_min, int((low_rank_ratio or 0.25) * d))
        self.act = _resolve_activation(activation if activation is not None else "relu")
        self.W1 = nn.ModuleList()
        self.W2 = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(num_layers):
            l1 = nn.Linear(d, h)
            l2 = nn.Linear(h, d)
            nn.init.normal_(l1.weight, std=0.02)
            nn.init.zeros_(l1.bias)
            nn.init.zeros_(l2.weight)  # near-identity
            nn.init.zeros_(l2.bias)
            self.W1.append(l1)
            self.W2.append(l2)
            self.bn.append(_make_norm(d, use_batch_norm, norm_type))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for l in range(self.num_layers):
            h = self.W2[l](self.act(self.W1[l](x)))
            x = x + _safe_bn(self.bn[l], h)
        return x


def _safe_bn(bn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply BN/LN but skip BN if training mode and batch < 2 (avoids the
    'Expected more than 1 value per channel' crash). LayerNorm has no batch
    coupling so it's always safe. Identity is passthrough.
    """
    if isinstance(bn, nn.Identity):
        return x
    if isinstance(bn, nn.BatchNorm1d) and bn.training and x.shape[0] < 2:
        return x
    return bn(x)


def _make_norm(d: int, use_norm: bool, norm_type: str) -> nn.Module:
    """Build a norm layer per the requested type.

    norm_type='batch' (paper default, BatchNorm1d)
    norm_type='layer' (LayerNorm — no batch coupling, more stable under bagging)
    use_norm=False → Identity (norm disabled entirely)
    """
    if not use_norm:
        return nn.Identity()
    if norm_type == "batch":
        return nn.BatchNorm1d(d)
    if norm_type == "layer":
        return nn.LayerNorm(d)
    raise ValueError(f"norm_type must be 'batch' or 'layer', got {norm_type!r}")


def _resolve_activation(name: Optional[str]):
    if name is None:
        return None
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unknown activation: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Gated adapter g_φ
# ─────────────────────────────────────────────────────────────────────────────


class GatedAdapter(nn.Module):
    """g_φ(x) = (1 − α) ⊙ x + α ⊙ δ(x) (Eq. 1).

    α is per-channel (vector of size d) or scalar. The inner block δ is itself a
    near-identity residual stack, so the whole adapter starts at identity for
    α₀ ≈ 0.
    """

    def __init__(self, d: int, block: nn.Module, alpha_init: float, alpha_shape: str):
        super().__init__()
        self.block = block
        if alpha_shape == "per-channel":
            self.alpha = nn.Parameter(torch.full((d,), float(alpha_init)))
        elif alpha_shape == "global":
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            raise ValueError(f"alpha_shape must be 'per-channel' or 'global', got {alpha_shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * (self.block(x) - x)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: entity embeddings for categoricals
# ─────────────────────────────────────────────────────────────────────────────


class _NAMMixerEncoder(nn.Module):
    """NAM + Mixer numeric encoder (QuantumBind-style, numerics-only).

    Per-feature MLP (NAM): each numeric column gets its own small MLP that
    expands the scalar to ``n_channels`` dims. Then a dense mixer projects
    the concatenated per-feature embeddings down to ``d_out``, learnable.

    Architecture per QB v3 / BEST_CONFIG:
       Linear(1, H) → ReLU → [Linear(H, H) → ReLU]^(L-1) → Linear(H, n_channels)
       per feature, then Linear(d_num × n_channels, d_out) → ReLU → Dropout.

    Input  (N, d_num) → Output (N, d_out).
    """

    def __init__(self, d_num: int, hidden: int = 32, n_channels: int = 32,
                 feat_layers: int = 2, d_out: int = 64, dropout: float = 0.2):
        super().__init__()
        self.d_num = d_num
        self.hidden = hidden
        self.n_channels = n_channels
        self.feat_layers = feat_layers
        self._d_out = d_out
        # Per-feature NAM stack: one Sequential per column, sharing the same
        # depth/width but independent parameters.
        def _make_mlp():
            layers = [nn.Linear(1, hidden), nn.ReLU()]
            for _ in range(feat_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            layers.append(nn.Linear(hidden, n_channels))
            return nn.Sequential(*layers)
        self.num_mlps = nn.ModuleList([_make_mlp() for _ in range(d_num)])
        # Mixer collapses (d_num × n_channels) → d_out.
        self.mix = nn.Linear(d_num * n_channels, d_out)
        nn.init.normal_(self.mix.weight, std=0.02)
        nn.init.zeros_(self.mix.bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def d_out(self) -> int:
        return self._d_out

    def forward(self, X_num: torch.Tensor) -> torch.Tensor:
        if X_num.numel() == 0:
            return torch.empty(X_num.shape[0], 0, device=X_num.device)
        # Per-column independent MLP
        parts = [self.num_mlps[j](X_num[:, j:j+1]) for j in range(self.d_num)]
        h = torch.cat(parts, dim=1)  # (N, d_num × n_channels)
        h = self.mix(h)
        h = F.relu(h)
        return self.dropout(h)


class _UnifiedNAMEncoder(nn.Module):
    """Unified NAM-style preprocessor: per-feature embedding for numerics AND
    categoricals, with optional ``n_slots`` independent embeddings per feature
    (Cabin-style). A single dense mixer collapses everything to ``d_out``.

    Per feature × per slot:
      Numerics:    Linear(1,H) → ReLU → ... → Linear(H, n_channels)
      Categoricals: Embedding(card, emb) → Linear(emb, n_channels)

    With n_slots > 1 each feature contributes ``n_slots`` independent views
    that get concatenated before the mixer. Input dim grows ×n_slots; output
    dim is ``d_out`` (unchanged).

    Input  (N, d_num) numerics tensor + list of (N,) cat ID tensors.
    Output (N, d_out).
    """

    def __init__(self, d_num: int, cat_cardinalities: list[int],
                 hidden: int = 32, n_channels: int = 32, feat_layers: int = 2,
                 cat_emb_dim: int = 16, d_out: int = 64, dropout: float = 0.2,
                 n_slots: int = 1):
        super().__init__()
        self.d_num = d_num
        self.n_cat = len(cat_cardinalities)
        self.hidden = hidden
        self.n_channels = n_channels
        self.n_slots = n_slots
        self._d_out = d_out
        # Numeric per-feature MLPs (n_slots independent ones per feature).
        # Layout: num_mlps[j * n_slots + s] is feature j, slot s.
        def _make_num_mlp():
            layers = [nn.Linear(1, hidden), nn.ReLU()]
            for _ in range(feat_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            layers.append(nn.Linear(hidden, n_channels))
            return nn.Sequential(*layers)
        self.num_mlps = nn.ModuleList(
            [_make_num_mlp() for _ in range(d_num * n_slots)]
        )
        # Categorical: embedding + projector to n_channels (n_slots per cat).
        # Layout: cat_embs[k * n_slots + s] / cat_projs[k * n_slots + s].
        self.cat_embs = nn.ModuleList()
        self.cat_projs = nn.ModuleList()
        for card in cat_cardinalities:
            actual_emb = max(2, min(cat_emb_dim, int(math.sqrt(card)) + 1))
            for _ in range(n_slots):
                emb = nn.Embedding(card, actual_emb)
                nn.init.normal_(emb.weight, std=0.05)
                self.cat_embs.append(emb)
                self.cat_projs.append(nn.Linear(actual_emb, n_channels))
        # Mixer: (d_num + n_cat) × n_slots × n_channels → d_out
        d_feat = (d_num + self.n_cat) * n_slots
        self.mix = nn.Linear(d_feat * n_channels, d_out)
        nn.init.normal_(self.mix.weight, std=0.02)
        nn.init.zeros_(self.mix.bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def d_out(self) -> int:
        return self._d_out

    def forward(self, X_num: torch.Tensor,
                cat_ids: list[torch.Tensor]) -> torch.Tensor:
        parts = []
        if self.d_num and X_num.numel():
            for j in range(self.d_num):
                for s in range(self.n_slots):
                    parts.append(self.num_mlps[j * self.n_slots + s](X_num[:, j:j+1]))
        for k, ids in enumerate(cat_ids):
            base = k * self.n_slots
            ids = ids.to(self.cat_embs[base].weight.device)
            for s in range(self.n_slots):
                emb = self.cat_embs[base + s](ids)
                parts.append(self.cat_projs[base + s](emb))
        if not parts:
            raise ValueError("UnifiedNAMEncoder got no features.")
        h = torch.cat(parts, dim=1)  # (N, d_feat × n_slots × n_channels)
        h = self.mix(h)
        h = F.relu(h)
        return self.dropout(h)


class _UnifiedRTDLEncoder(nn.Module):
    """RTDL Periodic embedding + entity embedding + mixer.

    Numerics:    PeriodicEmbeddings(d_num, d_emb)  → (N, d_num, d_emb)
    Categoricals: Embedding(card_j, emb_j) → Linear(emb_j, d_emb)  → (N, d_emb)

    Stack: (N, d_total, d_emb) where d_total = d_num + n_cat.
    Mixer: Linear(d_total × d_emb, d_out) → ReLU → Dropout.

    Mirrors _UnifiedNAMEncoder but uses RTDL's periodic embedding for the
    per-feature numeric step.

    Input  (N, d_num) numerics tensor + list of (N,) cat ID tensors.
    Output (N, d_out).
    """

    def __init__(self, d_num: int, cat_cardinalities: list[int],
                 d_emb: int = 24, n_freqs: int = 48, sigma: float = 0.01,
                 lite: bool = False, cat_emb_dim: int = 16,
                 d_out: int = 64, dropout: float = 0.2):
        super().__init__()
        try:
            from rtdl_num_embeddings import PeriodicEmbeddings
        except ImportError as e:
            raise ImportError(
                "RTDL preproc requires `rtdl_num_embeddings`: "
                "pip install rtdl_num_embeddings"
            ) from e
        self.d_num = d_num
        self.n_cat = len(cat_cardinalities)
        self.d_emb = d_emb
        self._d_out = d_out
        # Numeric: RTDL periodic embeddings
        if d_num:
            self.num_emb = PeriodicEmbeddings(
                n_features=d_num,
                d_embedding=d_emb,
                n_frequencies=n_freqs,
                frequency_init_scale=sigma,
                activation=True,
                lite=lite,
            )
        else:
            self.num_emb = None
        # Categorical: embed + project to d_emb so it stacks with numeric tokens
        self.cat_embs = nn.ModuleList()
        self.cat_projs = nn.ModuleList()
        for card in cat_cardinalities:
            actual_emb = max(2, min(cat_emb_dim, int(math.sqrt(card)) + 1))
            self.cat_embs.append(nn.Embedding(card, actual_emb))
            nn.init.normal_(self.cat_embs[-1].weight, std=0.05)
            self.cat_projs.append(nn.Linear(actual_emb, d_emb))
        # Mixer: (d_num + n_cat) × d_emb → d_out
        d_feat = d_num + self.n_cat
        self.mix = nn.Linear(d_feat * d_emb, d_out)
        nn.init.normal_(self.mix.weight, std=0.02)
        nn.init.zeros_(self.mix.bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def d_out(self) -> int:
        return self._d_out

    def forward(self, X_num: torch.Tensor,
                cat_ids: list[torch.Tensor]) -> torch.Tensor:
        device = (self.mix.weight.device if X_num.numel() == 0 else X_num.device)
        parts = []
        if self.d_num and X_num.numel():
            num_out = self.num_emb(X_num)  # (N, d_num, d_emb)
            parts.append(num_out.reshape(X_num.shape[0], self.d_num * self.d_emb))
        for k, ids in enumerate(cat_ids):
            ids = ids.to(device)
            emb = self.cat_embs[k](ids)               # (N, actual_emb)
            parts.append(self.cat_projs[k](emb))      # (N, d_emb)
        if not parts:
            raise ValueError("UnifiedRTDLEncoder got no features.")
        # Need (N, (d_num + n_cat) * d_emb); cat parts are (N, d_emb), already flat
        h = torch.cat(parts, dim=1)
        h = self.mix(h)
        h = F.relu(h)
        return self.dropout(h)


class _PLREncoder(nn.Module):
    """Periodic-Linear-ReLU embedding for numeric features.

    Thin wrapper around Yandex's ``rtdl_num_embeddings.PeriodicEmbeddings``
    (Gorishniy et al. 2023, "On Embeddings for Numerical Features in Tabular
    Deep Learning"). The official package provides per-feature frequencies
    + tuned defaults (``frequency_init_scale=0.01``); we just flatten the
    last two dims so the rest of the Retouche pipeline can treat numerics
    as a flat tensor.

    Input  (N, d_num) → Output (N, d_num × d_emb).
    """

    def __init__(self, d_num: int, n_freqs: int = 48, d_emb: int = 24,
                 sigma: float = 0.01, lite: bool = False):
        super().__init__()
        try:
            from rtdl_num_embeddings import PeriodicEmbeddings
        except ImportError as e:
            raise ImportError(
                "PLR numeric encoder requires the `rtdl_num_embeddings` package: "
                "pip install rtdl_num_embeddings"
            ) from e
        self.d_num = d_num
        self.d_emb = d_emb
        self.n_freqs = n_freqs
        self.sigma = sigma
        self.emb = PeriodicEmbeddings(
            n_features=d_num,
            d_embedding=d_emb,
            n_frequencies=n_freqs,
            frequency_init_scale=sigma,
            activation=True,  # the "R" (ReLU) in PLR
            lite=lite,
        )

    @property
    def d_out(self) -> int:
        return self.d_num * self.d_emb

    def forward(self, X_num: torch.Tensor) -> torch.Tensor:
        if X_num.numel() == 0:
            return torch.empty(X_num.shape[0], 0, device=X_num.device)
        h = self.emb(X_num)  # (N, d_num, d_emb)
        return h.reshape(X_num.shape[0], self.d_num * self.d_emb)


class _Preprocessor(nn.Module):
    """Converts raw rows to a continuous, fixed-width tensor for the adapter.

    - Numeric columns: median-imputed, standard-scaled (stats fit on train).
      Optionally further passed through PLR (periodic-linear-ReLU) embeddings.
    - Categorical columns: encoded according to ``cat_encoder``.

    The output dimension ``d_in`` depends on the categorical and numeric
    encoding choices and is computed in ``fit``.
    """

    def __init__(
        self,
        cat_encoder: str = "embedding",
        cat_embed_dim: object = "auto",
        onehot_max_card: int = 8,
        num_encoder: str = "standard",
        plr_freqs: int = 48,
        plr_emb_dim: object = "auto",
        plr_sigma: float = 0.01,
        plr_lite: bool = False,
        nam_hidden: int = 32,
        nam_d_out: object = "auto",
        nam_dropout: float = 0.2,
        nam_slots: int = 1,
    ):
        super().__init__()
        if cat_encoder not in {"ordinal", "onehot", "embedding", "mixed"}:
            raise ValueError(
                f"cat_encoder must be one of ordinal/onehot/embedding/mixed, got {cat_encoder!r}"
            )
        if num_encoder not in {"standard", "plr", "nam", "unified_nam", "rtdl"}:
            raise ValueError(
                f"num_encoder must be 'standard'/'plr'/'nam'/'unified_nam'/'rtdl', "
                f"got {num_encoder!r}"
            )
        self.cat_encoder = cat_encoder
        self.cat_embed_dim = cat_embed_dim
        self.onehot_max_card = onehot_max_card
        self.num_encoder = num_encoder
        self.plr_freqs = plr_freqs
        self.plr_emb_dim = plr_emb_dim
        self.plr_sigma = plr_sigma
        self.plr_lite = plr_lite
        self.nam_hidden = nam_hidden
        self.nam_d_out = nam_d_out
        self.nam_dropout = nam_dropout
        self.nam_slots = nam_slots

    @staticmethod
    def _default_embed_dim(cardinality: int) -> int:
        return max(2, min(50, (cardinality + 1) // 2))

    def fit(self, X: np.ndarray, cat_idx: list[int], max_d: int = 500):
        n, d_raw = X.shape
        self.cat_idx_ = list(cat_idx)
        self.num_idx_ = [i for i in range(d_raw) if i not in self.cat_idx_]
        self.unified_ = self.num_encoder in ("unified_nam", "rtdl")

        # Numeric: impute + scale
        if self.num_idx_:
            X_num = X[:, self.num_idx_].astype(np.float64)
            self.num_imputer_ = SimpleImputer(strategy="median").fit(X_num)
            self.num_scaler_ = StandardScaler().fit(self.num_imputer_.transform(X_num))
            # Optional learnable numeric encoder on top
            if self.num_encoder == "plr":
                if self.plr_emb_dim == "auto":
                    d_emb = max(2, min(24, max_d // max(1, len(self.num_idx_))))
                else:
                    d_emb = int(self.plr_emb_dim)
                self.num_enc_ = _PLREncoder(
                    d_num=len(self.num_idx_),
                    n_freqs=self.plr_freqs,
                    d_emb=d_emb,
                    sigma=self.plr_sigma,
                    lite=self.plr_lite,
                )
            elif self.num_encoder == "nam":
                # QuantumBind-style NAM + Mixer: per-feature MLP expands, then
                # a learnable mixer collapses to d_out. Keeps the post-encoder
                # width close to original d so TabICL doesn't blow up on the
                # column-attention step.
                if self.nam_d_out == "auto":
                    d_out = max(8, min(128, len(self.num_idx_) * 2))
                else:
                    d_out = int(self.nam_d_out)
                self.num_enc_ = _NAMMixerEncoder(
                    d_num=len(self.num_idx_),
                    hidden=self.nam_hidden,
                    d_out=d_out,
                    dropout=self.nam_dropout,
                )
            else:
                self.num_enc_ = None
        else:
            self.num_imputer_ = None
            self.num_scaler_ = None
            self.num_enc_ = None

        # Categorical: build per-column label encoders to map values → int IDs
        # (unknown category at predict time → mapped to an "unknown" index)
        self.cat_encoders_ = []
        self.cat_cards_ = []
        for j in self.cat_idx_:
            col = X[:, j]
            col_str = np.array([_safe_str(v) for v in col])
            le = LabelEncoder().fit(col_str)
            self.cat_encoders_.append(le)
            self.cat_cards_.append(len(le.classes_) + 1)  # +1 for unknown

        # Set up trainable cat parameters
        self.cat_embeds_ = nn.ModuleList()
        cat_widths = []
        for card in self.cat_cards_:
            if self.cat_encoder == "ordinal":
                width = 1
                self.cat_embeds_.append(nn.Identity())
            elif self.cat_encoder == "onehot":
                width = card
                self.cat_embeds_.append(nn.Identity())
            elif self.cat_encoder == "mixed":
                if card <= self.onehot_max_card + 1:
                    width = card
                    self.cat_embeds_.append(nn.Identity())
                else:
                    width = 1
                    self.cat_embeds_.append(nn.Identity())
            else:  # embedding
                if self.cat_embed_dim == "auto":
                    width = self._default_embed_dim(card)
                else:
                    width = int(self.cat_embed_dim)
                emb = nn.Embedding(card, width)
                nn.init.normal_(emb.weight, std=0.05)
                self.cat_embeds_.append(emb)
            cat_widths.append(width)
        self.cat_widths_ = cat_widths
        self.unknown_ids_ = [card - 1 for card in self.cat_cards_]

        if self.unified_:
            # Unified mode: a single encoder consumes both numerics and cat IDs
            # and outputs d_target directly. Overrides the per-modality path.
            if self.nam_d_out == "auto":
                # Default to original raw feature count (so TabICL sees the
                # same C as it would with no preproc) — column-attention cost
                # stays bounded.
                d_target = max(8, len(self.num_idx_) + len(self.cat_idx_))
            else:
                d_target = int(self.nam_d_out)
            if self.num_encoder == "unified_nam":
                self.unified_enc_ = _UnifiedNAMEncoder(
                    d_num=len(self.num_idx_),
                    cat_cardinalities=self.cat_cards_,
                    hidden=self.nam_hidden,
                    n_channels=self.nam_hidden,
                    feat_layers=2,
                    cat_emb_dim=16,
                    d_out=d_target,
                    dropout=self.nam_dropout,
                    n_slots=self.nam_slots,
                )
            else:  # "rtdl"
                # Auto-pick d_emb so that intermediate (d_total × d_emb) is
                # reasonable: cap at ~512 cells in the mixer's input.
                d_total = len(self.num_idx_) + len(self.cat_idx_)
                if self.plr_emb_dim == "auto":
                    d_emb = max(4, min(24, 512 // max(1, d_total)))
                else:
                    d_emb = int(self.plr_emb_dim)
                self.unified_enc_ = _UnifiedRTDLEncoder(
                    d_num=len(self.num_idx_),
                    cat_cardinalities=self.cat_cards_,
                    d_emb=d_emb,
                    n_freqs=self.plr_freqs,
                    sigma=self.plr_sigma,
                    lite=self.plr_lite,
                    cat_emb_dim=16,
                    d_out=d_target,
                    dropout=self.nam_dropout,
                )
            self.d_out_ = d_target
        else:
            self.unified_enc_ = None
            d_num_raw = len(self.num_idx_)
            if self.num_enc_ is not None:
                d_num = self.num_enc_.d_out
            else:
                d_num = d_num_raw
            d_cat = sum(self.cat_widths_)
            self.d_out_ = d_num + d_cat
        return self

    def transform_numeric(self, X: np.ndarray) -> torch.Tensor:
        if not self.num_idx_:
            return torch.empty(X.shape[0], 0)
        X_num = X[:, self.num_idx_].astype(np.float64)
        X_num = self.num_imputer_.transform(X_num)
        X_num = self.num_scaler_.transform(X_num)
        return torch.from_numpy(X_num).float()

    def transform_cat_ids(self, X: np.ndarray) -> list[torch.Tensor]:
        ids = []
        for k, j in enumerate(self.cat_idx_):
            le = self.cat_encoders_[k]
            unk = self.unknown_ids_[k]
            known = set(le.classes_)
            col = np.array([_safe_str(v) for v in X[:, j]])
            mapped = np.where(np.isin(col, list(known)),
                              col, le.classes_[0] if len(le.classes_) else "")
            out = np.where(np.isin(col, list(known)),
                           le.transform(mapped), unk).astype(np.int64)
            ids.append(torch.from_numpy(out))
        return ids

    def forward(self, X_num: torch.Tensor, cat_ids: list[torch.Tensor]) -> torch.Tensor:
        """Build the full continuous representation as a differentiable tensor.

        Numeric block: standardized features (non-trainable scaler stats) optionally
        passed through trainable PLR embeddings. Categorical embeddings are
        trainable when cat_encoder='embedding'; one-hot / ordinal paths have no
        parameters. Unified-NAM mode bypasses both and produces d_target directly.
        """
        device = X_num.device if X_num.numel() else (cat_ids[0].device if cat_ids else None)
        if device is None:
            raise ValueError("Empty input to preprocessor.")

        if self.unified_:
            # Single encoder consumes both modalities, outputs (N, d_target).
            return self.unified_enc_(X_num, cat_ids)

        if X_num.numel():
            X_num = X_num.to(device)
            if self.num_enc_ is not None:
                X_num_out = self.num_enc_(X_num)
            else:
                X_num_out = X_num
            parts = [X_num_out]
        else:
            parts = []

        for k, ids in enumerate(cat_ids):
            ids = ids.to(device)
            if self.cat_encoder == "embedding":
                parts.append(self.cat_embeds_[k](ids))
            elif self.cat_encoder == "onehot":
                parts.append(F.one_hot(ids, num_classes=self.cat_cards_[k]).float())
            elif self.cat_encoder == "mixed":
                card = self.cat_cards_[k]
                if card <= self.onehot_max_card + 1:
                    parts.append(F.one_hot(ids, num_classes=card).float())
                else:
                    parts.append(ids.unsqueeze(1).float())  # ordinal as-is
            else:  # ordinal
                parts.append(ids.unsqueeze(1).float())

        if not parts:
            raise ValueError("Preprocessor produced no features.")
        return torch.cat(parts, dim=1)


def _safe_str(v):
    if v is None:
        return "__nan__"
    if isinstance(v, float) and math.isnan(v):
        return "__nan__"
    return str(v)


def _coslog_lr(step: int, total: int, n_cycles: int = 4, min_factor: float = 0.01) -> float:
    """Multi-cycle log-spaced cosine LR schedule ('coslog4' in the paper).

    Cycle length grows log-spaced; within each cycle, lr follows a half-cosine
    from 1 → min_factor. Returns a multiplier in [min_factor, 1].
    """
    if total <= 1:
        return 1.0
    boundaries = [0]
    span = total
    for c in range(n_cycles):
        cycle_len = max(1, int(span * (2 ** (c - n_cycles + 1))))
        boundaries.append(boundaries[-1] + cycle_len)
    boundaries[-1] = total

    for c in range(n_cycles):
        a, b = boundaries[c], boundaries[c + 1]
        if a <= step < b:
            t = (step - a) / max(1, b - a)
            return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * t))
    return min_factor


# ─────────────────────────────────────────────────────────────────────────────
# Public estimator
# ─────────────────────────────────────────────────────────────────────────────


class RetoucheTabICLClassifier(BaseEstimator, ClassifierMixin):
    """TFM-Retouche on a frozen TabICLv2 backbone.

    Trains a gated near-identity residual adapter in input space, with a post-
    training identity guard that falls back to the unmodified TabICL whenever
    the adapter does not strictly improve held-out validation.

    Parameters
    ----------
    Adapter architecture
        block_type : 'cross' (default, DCNv2) or 'mlp' (residual bottleneck).
        num_layers : depth of the residual stack δ. Default 2.
        low_rank_ratio : ratio r so hidden width h = ⌊r·d⌋. ``None`` ⇒ full rank.
        hidden_dim : kept for MLP block compatibility (overrides low_rank_ratio
            when set; the paper's default Table 2 fixes h=64 only for the MLP path).
        use_batch_norm : BN inside each inner layer. Default True.
        activation : 'relu' / 'gelu' / 'tanh' / None. Cross-block default None;
            MLP-block default 'relu'.
        alpha_init : init value for the gate α. Default 0.02.
        alpha_shape : 'per-channel' (default) or 'global'.

    Categorical handling
        cat_features : 'auto' | list[int] | None. Indices of categorical columns.
            'auto' = pandas object/category dtype, or low-cardinality on numpy.
        cat_encoder : 'embedding' (default), 'ordinal', 'onehot', 'mixed'.
        cat_embed_dim : 'auto' (default) or int.
        onehot_max_card : threshold for the 'mixed' variant.

    Optimization
        lr : base learning rate for matrices. Default 5e-3.
        weight_decay : AdamW weight decay on matrices only. Default 3e-3.
        gate_lr_factor : α gets ``gate_lr_factor × lr``. Default 3.0.
        label_smoothing : CE label smoothing. Default 0.15.
        max_grad_norm : grad-norm clip. Default 2.0.
        betas : AdamW (β1, β2). Default (0.9, 0.97).
        epochs : training epochs. Default 150.
        patience : early-stopping patience on val log-loss. Default 10.
        lr_schedule : 'coslog4' (default), 'cosine', 'constant'.

    Other
        max_d : frozen orthogonal projection inserted when raw d_in > max_d.
            Default 500 (TabICLv2 input cap).
        identity_guard_tol : adapter must beat raw by more than this on val.
            Default 0.005 (= 0.5%).
        val_size : fraction of train used as the guard / early-stop split. Default 0.2.
        max_context : 'auto' | int | None. Cap on the number of training rows
            fed through the differentiable forward at each epoch (paper's
            Appendix G ``max_samples`` knob — they cap at 15k–60k). 'auto'
            picks a value that fits L40S-class memory for typical d (~8000).
            ``None`` disables subsampling. Reshuffled every epoch.
        max_query : int or None. Cap on the val rows scored per epoch (paper
            also chunks query side on the largest datasets). ``None`` = no cap.
        device : torch device or None (auto).
        random_state : seed for the val split.
        verbose : print per-epoch val loss when True.
        backbone_kwargs : dict, extra kwargs forwarded to TabICLClassifier.
    """

    def __init__(
        self,
        # adapter
        block_type: str = "cross",
        num_layers: int = 2,
        low_rank_ratio: Optional[float] = 0.25,
        hidden_dim: Optional[int] = None,
        use_batch_norm: bool = True,
        norm_type: str = "batch",
        activation: Optional[str] = None,
        alpha_init: float = 0.02,
        alpha_shape: str = "per-channel",
        # categorical
        cat_features: object = "auto",
        cat_encoder: str = "embedding",
        cat_embed_dim: object = "auto",
        onehot_max_card: int = 8,
        # numeric
        num_encoder: str = "standard",
        plr_freqs: int = 48,
        plr_emb_dim: object = "auto",
        plr_sigma: float = 0.01,
        plr_lite: bool = False,
        nam_hidden: int = 32,
        nam_d_out: object = "auto",
        nam_dropout: float = 0.2,
        nam_slots: int = 1,
        # optimization
        lr: float = 5e-3,
        weight_decay: float = 3e-3,
        gate_lr_factor: float = 3.0,
        label_smoothing: float = 0.15,
        max_grad_norm: float = 2.0,
        betas: tuple = (0.9, 0.97),
        epochs: int = 150,
        patience: int = 10,
        lr_schedule: str = "coslog4",
        # other
        max_d: int = 500,
        identity_guard_tol: float = 0.005,
        val_size: float = 0.2,
        max_context: object = "auto",
        max_query: Optional[int] = None,
        device: Optional[str] = None,
        random_state: Optional[int] = 42,
        verbose: bool = False,
        backbone_kwargs: Optional[dict] = None,
    ):
        self.block_type = block_type
        self.num_layers = num_layers
        self.low_rank_ratio = low_rank_ratio
        self.hidden_dim = hidden_dim
        self.use_batch_norm = use_batch_norm
        self.norm_type = norm_type
        self.activation = activation
        self.alpha_init = alpha_init
        self.alpha_shape = alpha_shape
        self.cat_features = cat_features
        self.cat_encoder = cat_encoder
        self.cat_embed_dim = cat_embed_dim
        self.onehot_max_card = onehot_max_card
        self.num_encoder = num_encoder
        self.plr_freqs = plr_freqs
        self.plr_emb_dim = plr_emb_dim
        self.plr_sigma = plr_sigma
        self.plr_lite = plr_lite
        self.nam_hidden = nam_hidden
        self.nam_d_out = nam_d_out
        self.nam_dropout = nam_dropout
        self.nam_slots = nam_slots
        self.lr = lr
        self.weight_decay = weight_decay
        self.gate_lr_factor = gate_lr_factor
        self.label_smoothing = label_smoothing
        self.max_grad_norm = max_grad_norm
        self.betas = betas
        self.epochs = epochs
        self.patience = patience
        self.lr_schedule = lr_schedule
        self.max_d = max_d
        self.identity_guard_tol = identity_guard_tol
        self.val_size = val_size
        self.max_context = max_context
        self.max_query = max_query
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.backbone_kwargs = backbone_kwargs

    # ─── public API ─────────────────────────────────────────────────────────

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the adapter, then run the identity guard.

        Parameters
        ----------
        X, y : training data.
        X_val, y_val : optional held-out validation set used both for early
            stopping during adapter training and as the identity guard's
            scoring set. When provided, the internal ``train_test_split`` is
            skipped — this is the protocol the paper actually uses (the
            AutoGluon bag-fold val split serves the same role). Strongly
            recommended when running under an outer cross-validation loop so
            that the guard scores the same distribution as the outer test.
        """
        X_np = _to_numpy(X)
        y_np = np.asarray(y)

        # Fit label encoder on combined train+val labels so val labels with
        # otherwise-unseen classes don't fall outside the class set
        if y_val is not None:
            y_np_combined = np.concatenate([y_np, np.asarray(y_val)])
        else:
            y_np_combined = y_np
        self.label_encoder_ = LabelEncoder().fit(y_np_combined)
        y_enc = self.label_encoder_.transform(y_np)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        self.n_classes_ = n_classes

        cat_idx = self._resolve_cat_features(X, X_np)

        # Fit preprocessor on train (val unseen — mimics test-time behavior)
        self.preproc_ = _Preprocessor(
            cat_encoder=self.cat_encoder,
            cat_embed_dim=self.cat_embed_dim,
            onehot_max_card=self.onehot_max_card,
            num_encoder=self.num_encoder,
            plr_freqs=self.plr_freqs,
            plr_emb_dim=self.plr_emb_dim,
            plr_sigma=self.plr_sigma,
            plr_lite=self.plr_lite,
            nam_hidden=self.nam_hidden,
            nam_d_out=self.nam_d_out,
            nam_dropout=self.nam_dropout,
            nam_slots=self.nam_slots,
        ).fit(X_np, cat_idx, max_d=self.max_d)

        # train / guard split
        if X_val is not None and y_val is not None:
            X_tr = X_np
            y_tr = y_enc
            X_va = _to_numpy(X_val)
            y_va = self.label_encoder_.transform(np.asarray(y_val))
            self._external_val_ = True
        else:
            try:
                X_tr, X_va, y_tr, y_va = train_test_split(
                    X_np, y_enc,
                    test_size=self.val_size,
                    stratify=y_enc,
                    random_state=self.random_state,
                )
            except ValueError:
                X_tr, X_va, y_tr, y_va = train_test_split(
                    X_np, y_enc, test_size=self.val_size,
                    random_state=self.random_state,
                )
            self._external_val_ = False
        self._guard_split_ = (X_va, y_va)

        device = _resolve_device(self.device)
        self.device_ = device

        d_in = self.preproc_.d_out_
        d_back = min(self.max_d, d_in)
        self._d_in_ = d_in
        self._d_back_ = d_back

        # adapter
        if self.block_type == "cross":
            block = CrossBlock(
                d=d_in, num_layers=self.num_layers,
                low_rank_ratio=self.low_rank_ratio,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                norm_type=self.norm_type,
            )
        elif self.block_type == "mlp":
            block = ResMLPBlock(
                d=d_in, num_layers=self.num_layers,
                low_rank_ratio=self.low_rank_ratio,
                activation=self.activation if self.activation is not None else "relu",
                use_batch_norm=self.use_batch_norm,
                norm_type=self.norm_type,
            )
        else:
            raise ValueError(f"block_type must be 'cross' or 'mlp', got {self.block_type!r}")

        self.adapter_ = GatedAdapter(
            d=d_in, block=block,
            alpha_init=self.alpha_init, alpha_shape=self.alpha_shape,
        ).to(device)

        # frozen orthogonal projection for d_in > max_d
        if d_in > self.max_d:
            proj = torch.empty(d_in, d_back)
            nn.init.orthogonal_(proj)
            self._proj_ = proj.to(device)
            self.use_proj_ = True
        else:
            self._proj_ = None
            self.use_proj_ = False

        # move preproc parameters to device
        self.preproc_.to(device)

        # backbone (differentiable mode)
        clf_kwargs = dict(device=str(device), differentiable_input=True)
        if self.backbone_kwargs:
            clf_kwargs.update(self.backbone_kwargs)
        self.clf_ = TabICLClassifier(**clf_kwargs)

        # pre-build numeric / cat tensors for the whole train set
        X_tr_num = self.preproc_.transform_numeric(X_tr).to(device)
        X_tr_cat = [t.to(device) for t in self.preproc_.transform_cat_ids(X_tr)]
        X_va_num = self.preproc_.transform_numeric(X_va).to(device)
        X_va_cat = [t.to(device) for t in self.preproc_.transform_cat_ids(X_va)]
        y_tr_t = torch.from_numpy(y_tr).long().to(device)
        y_va_t = torch.from_numpy(y_va).long().to(device)

        # optimizer with three param groups (matrices / biases / α)
        mat_params, bias_params, alpha_params, emb_params = [], [], [], []
        for n, p in self.adapter_.named_parameters():
            if not p.requires_grad:
                continue
            if "alpha" in n:
                alpha_params.append(p)
            elif p.ndim == 1:
                bias_params.append(p)
            else:
                mat_params.append(p)
        for n, p in self.preproc_.named_parameters():
            if p.requires_grad:
                emb_params.append(p)

        param_groups = [
            {"params": mat_params, "weight_decay": self.weight_decay, "lr": self.lr},
            {"params": bias_params, "weight_decay": 0.0, "lr": self.lr},
            {"params": alpha_params, "weight_decay": 0.0, "lr": self.lr * self.gate_lr_factor},
            {"params": emb_params, "weight_decay": 0.0, "lr": self.lr},
        ]
        param_groups = [g for g in param_groups if g["params"]]
        optimizer = torch.optim.AdamW(param_groups, betas=self.betas)

        # Training data only — the guard val (X_va) is held out throughout
        # training and used solely by the identity guard at the end (§3.5).
        # Each epoch we partition the TRAINING data into context C and query
        # Q (§3.4: "the training data are randomly partitioned into a context
        # set C and a query set Q, matching the in-context-learning setting").
        # The query LABELS provide the loss target; query INPUTS are scored
        # via TabICL.predict_differentiable conditioned on C.

        # Resolve max_context. Paper Appendix G caps at 15k–60k; for L40S
        # (44GB) safety we scale a target memory budget against the adapter
        # output width d_back. With d_back=d_in (no projection) memory grows
        # ~linearly with n_context × d_back × num_layers, plus a fixed Adam
        # state cost from the adapter params. Empirically 6k×500 ≈ OOM
        # without head-room for the BatchEnsemble path (which holds K copies).
        n_train = X_tr_num.shape[0] if X_tr_num.numel() else len(y_tr_t)
        if self.max_context == "auto":
            # Soft budget. TabICL's row-attention is O(N²·d) per layer with ~6
            # attention layers in the ICL transformer plus the diff-forward
            # backward pass holding activations. Empirical L40S 44GB headroom:
            # n_context × d_back × n_query  ≲  600M ⇒ at n_query≈1k that's
            # n_context × d_back ≲ 600k. Plus PLR expansion increases d_back;
            # bagged path multiplies the memory cost across K members.
            cap = max(300, int(600_000 / max(1, d_back)))
            max_ctx = min(n_train - 1, cap, 6000)
        elif self.max_context is None:
            max_ctx = n_train - 1
        else:
            max_ctx = min(n_train - 1, int(self.max_context))
        # Always leave at least a few rows for the query
        max_ctx = max(max_ctx, n_train // 2)
        max_q_train = n_train - max_ctx
        self._max_context_ = max_ctx
        self._max_query_ = max_q_train

        # For early stopping we score a fixed-shuffle held-out slice of train
        # (separate from the per-epoch reshuffled C/Q). This monitors val-like
        # loss without leaking into gradients. The slice is held in memory
        # but NEVER appears in the query set used for training loss.
        rng = np.random.RandomState(self.random_state or 0)
        # Hold out at most ~30% of train for early stopping; on tiny datasets
        # cap small enough to leave a usable train pool (≥ 4 rows).
        max_es = max(0, n_train - 4)
        n_es = min(max_es, max(0, min(int(0.1 * n_train), 1000)))
        if n_es == 0:
            es_idx_np = np.empty(0, dtype=np.int64)
        else:
            es_idx_np = _stratified_sample(rng, y_tr, n_es)
        es_mask = np.zeros(n_train, dtype=bool)
        es_mask[es_idx_np] = True
        train_pool_idx = np.where(~es_mask)[0]
        es_idx = torch.from_numpy(es_idx_np).long().to(device)
        train_pool_idx_t = torch.from_numpy(train_pool_idx).long().to(device)
        y_train_pool_np = y_tr[train_pool_idx]
        n_train_pool = len(train_pool_idx)
        # Re-tighten max_ctx to fit in the train pool (excluding es slice).
        # Leave at least 2 rows for the query (BN won't crash, plus single-row
        # query has no useful gradient signal).
        max_ctx = min(max_ctx, max(1, n_train_pool - 2))
        max_q_train = max(2, n_train_pool - max_ctx)

        # init backbone with one dummy forward — just to set up z-norm stats
        with torch.no_grad():
            x_pre = self.preproc_(X_tr_num[train_pool_idx_t],
                                  [t[train_pool_idx_t] for t in X_tr_cat])
            x_post = self.adapter_(x_pre)
            x_back = x_post @ self._proj_ if self.use_proj_ else x_post
        self.clf_.fit_with_differentiable_input(x_back.detach(), y_tr_t[train_pool_idx_t])

        # training loop
        best_es = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(self.epochs):
            lr_mult = self._lr_mult(epoch, self.epochs)
            for g, base in zip(optimizer.param_groups,
                               [self.lr, self.lr, self.lr * self.gate_lr_factor, self.lr]):
                if g["params"]:
                    g["lr"] = base * lr_mult

            self.adapter_.train()
            self.preproc_.train()

            # Random partition of train pool into context C and query Q.
            # Stratify-sample the context, then the query is what's left.
            ctx_local = _stratified_sample(rng, y_train_pool_np, max_ctx)
            ctx_set = set(ctx_local.tolist())
            q_local = np.array([i for i in range(n_train_pool) if i not in ctx_set],
                               dtype=np.int64)
            if len(q_local) > max_q_train:
                q_local = rng.choice(q_local, size=max_q_train, replace=False)
            ctx_global = train_pool_idx[ctx_local]
            q_global = train_pool_idx[q_local]

            ctx_t = torch.from_numpy(ctx_global).long().to(device)
            q_t = torch.from_numpy(q_global).long().to(device)

            X_ctx_num = X_tr_num[ctx_t] if X_tr_num.numel() else X_tr_num
            X_ctx_cat = [t[ctx_t] for t in X_tr_cat]
            y_ctx = y_tr_t[ctx_t]
            X_q_num = X_tr_num[q_t] if X_tr_num.numel() else X_tr_num
            X_q_cat = [t[q_t] for t in X_tr_cat]
            y_q = y_tr_t[q_t]

            x_ctx_pre = self.preproc_(X_ctx_num, X_ctx_cat)
            x_ctx_post = self.adapter_(x_ctx_pre)
            x_q_pre = self.preproc_(X_q_num, X_q_cat)
            x_q_post = self.adapter_(x_q_pre)

            x_ctx_back = x_ctx_post @ self._proj_ if self.use_proj_ else x_ctx_post
            x_q_back = x_q_post @ self._proj_ if self.use_proj_ else x_q_post

            self.clf_.fit_with_differentiable_input(x_ctx_back, y_ctx)
            logits = self.clf_.predict_differentiable(x_q_back, return_logits=True)

            loss = F.cross_entropy(logits, y_q, label_smoothing=self.label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            params_for_clip = []
            for g in optimizer.param_groups:
                params_for_clip.extend(g["params"])
            torch.nn.utils.clip_grad_norm_(params_for_clip, self.max_grad_norm)
            optimizer.step()

            # Early stopping: score the fixed ES slice with the train pool
            # as context (no gradient).
            self.adapter_.eval()
            self.preproc_.eval()
            with torch.no_grad():
                x_pool_pre = self.preproc_(X_tr_num[train_pool_idx_t],
                                           [t[train_pool_idx_t] for t in X_tr_cat])
                x_pool_post = self.adapter_(x_pool_pre)
                x_pool_back = x_pool_post @ self._proj_ if self.use_proj_ else x_pool_post
                x_es_pre = self.preproc_(X_tr_num[es_idx],
                                         [t[es_idx] for t in X_tr_cat])
                x_es_post = self.adapter_(x_es_pre)
                x_es_back = x_es_post @ self._proj_ if self.use_proj_ else x_es_post
                self.clf_.fit_with_differentiable_input(x_pool_back, y_tr_t[train_pool_idx_t])
                es_logits = self.clf_.predict_differentiable(x_es_back, return_logits=True)
                es_loss = F.cross_entropy(es_logits, y_tr_t[es_idx]).item()

            if self.verbose:
                print(f"[Retouche] ep {epoch:3d}  q_loss={loss.item():.4f}  "
                      f"es_loss={es_loss:.4f}  lr×{lr_mult:.3f}")

            if es_loss + 1e-6 < best_es:
                best_es = es_loss
                best_state = {
                    "adapter": {k: v.detach().clone() for k, v in self.adapter_.state_dict().items()},
                    "preproc": {k: v.detach().clone() for k, v in self.preproc_.state_dict().items()},
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    if self.verbose:
                        print(f"[Retouche] early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.adapter_.load_state_dict(best_state["adapter"])
            self.preproc_.load_state_dict(best_state["preproc"])

        # identity guard: compare adapter vs raw on the held-out val
        self._run_identity_guard(X_tr, y_tr, X_va, y_va)

        return self

    def predict_proba(self, X):
        X_np = _to_numpy(X)
        if self._use_adapter_:
            return self._predict_proba_adapter(X_np)
        return self._predict_proba_raw(X_np)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(idx)

    # ─── internals ──────────────────────────────────────────────────────────

    def _resolve_cat_features(self, X, X_np) -> list[int]:
        if self.cat_features == "auto":
            try:
                import pandas as pd
                if isinstance(X, pd.DataFrame):
                    cats = []
                    for i, col in enumerate(X.columns):
                        if X[col].dtype == object or str(X[col].dtype) == "category":
                            cats.append(i)
                    return cats
            except ImportError:
                pass
            # numpy: low-cardinality int columns
            cats = []
            for i in range(X_np.shape[1]):
                col = X_np[:, i]
                try:
                    arr = col.astype(np.float64)
                    if np.all(np.isfinite(arr)) and np.all(arr == arr.astype(int)):
                        if len(np.unique(arr)) <= 20:
                            cats.append(i)
                except (ValueError, TypeError):
                    cats.append(i)
            return cats
        if self.cat_features is None:
            return []
        return list(self.cat_features)

    def _lr_mult(self, step: int, total: int) -> float:
        if self.lr_schedule == "constant":
            return 1.0
        if self.lr_schedule == "cosine":
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / max(1, total)))
        return _coslog_lr(step, total, n_cycles=4)

    def _build_train_features_eval(self, X_np: np.ndarray) -> torch.Tensor:
        """Build adapted, backbone-ready features for inference (no_grad)."""
        device = self.device_
        X_num = self.preproc_.transform_numeric(X_np).to(device)
        X_cat = [t.to(device) for t in self.preproc_.transform_cat_ids(X_np)]
        self.adapter_.eval()
        self.preproc_.eval()
        with torch.no_grad():
            x = self.preproc_(X_num, X_cat)
            x = self.adapter_(x)
            if self.use_proj_:
                x = x @ self._proj_
        return x

    def _predict_proba_adapter(self, X_test_np: np.ndarray) -> np.ndarray:
        # Refit train features through the adapter
        X_tr_np = self._train_X_full_np_
        y_tr_t = self._train_y_full_t_

        x_tr_back = self._build_train_features_eval(X_tr_np)
        self.clf_.fit_with_differentiable_input(x_tr_back, y_tr_t)
        x_te_back = self._build_train_features_eval(X_test_np)
        with torch.no_grad():
            probs = self.clf_.predict_differentiable(
                x_te_back, return_logits=False, softmax_temperature=0.9,
            )
        probs_np = probs.detach().cpu().numpy()
        if not np.all(np.isfinite(probs_np)):
            # Adapter blew up at inference: fall back to a raw TabICL on the
            # train set. Build it lazily once.
            if getattr(self, "_emergency_raw_clf_", None) is None:
                raw_kwargs = dict()
                if self.backbone_kwargs:
                    raw_kwargs.update(self.backbone_kwargs)
                raw_kwargs.pop("differentiable_input", None)
                emergency = TabICLClassifier(device=str(self.device_), **raw_kwargs)
                emergency.fit(X_tr_np, y_tr_t.detach().cpu().numpy())
                self._emergency_raw_clf_ = emergency
            return self._emergency_raw_clf_.predict_proba(X_test_np)
        return probs_np

    def _predict_proba_raw(self, X_test_np: np.ndarray) -> np.ndarray:
        clf = self._raw_clf_
        return clf.predict_proba(X_test_np)

    def _run_identity_guard(self, X_tr_np, y_tr, X_va_np, y_va):
        """Score adapter vs raw on val. Decide whether to deploy the adapter."""
        # After the guard scores adapter vs raw on (X_va, y_va), the val set
        # has served its purpose; we refit the deployed model on train+val for
        # inference. The outer-CV test fold is what stays unseen — whether the
        # val was internal (from train_test_split) or external (passed in),
        # using it for the final refit doesn't leak into outer test.
        full_X = np.concatenate([X_tr_np, X_va_np], axis=0)
        full_y = np.concatenate([y_tr, y_va], axis=0)
        self._train_X_full_np_ = full_X
        self._train_y_full_t_ = torch.from_numpy(full_y).long().to(self.device_)
        refit_X_for_raw = full_X
        refit_y_for_raw = full_y

        # adapter-path val score
        x_tr_back = self._build_train_features_eval(X_tr_np)
        self.clf_.fit_with_differentiable_input(x_tr_back, torch.from_numpy(y_tr).long().to(self.device_))
        x_va_back = self._build_train_features_eval(X_va_np)
        with torch.no_grad():
            probs_adapter = self.clf_.predict_differentiable(
                x_va_back, return_logits=False, softmax_temperature=0.9,
            ).detach().cpu().numpy()
        adapter_nan = not np.all(np.isfinite(probs_adapter))

        # raw-path val score (fresh TabICLClassifier on raw inputs, no adapter)
        raw_kwargs = dict()
        if self.backbone_kwargs:
            raw_kwargs.update(self.backbone_kwargs)
        raw_kwargs.pop("differentiable_input", None)
        raw_clf = TabICLClassifier(device=str(self.device_), **raw_kwargs)
        raw_clf.fit(X_tr_np, y_tr)
        probs_raw = raw_clf.predict_proba(X_va_np)

        if adapter_nan:
            adapter_score = float("inf")
        else:
            adapter_score = self._guard_metric(probs_adapter, y_va)
        raw_score = self._guard_metric(probs_raw, y_va)

        # both scores are "lower is better" (1-AUC for binary, log-loss for multi)
        improved = (not adapter_nan) and adapter_score < raw_score * (1.0 - self.identity_guard_tol)

        self._guard_metrics_ = {
            "adapter": adapter_score, "raw": raw_score,
            "improved": improved, "adapter_nan": adapter_nan,
        }
        self._use_adapter_ = improved
        if not improved:
            raw_clf_full = TabICLClassifier(device=str(self.device_), **raw_kwargs)
            raw_clf_full.fit(refit_X_for_raw, refit_y_for_raw)
            self._raw_clf_ = raw_clf_full
        else:
            self._raw_clf_ = None

        if self.verbose:
            tag = "USE ADAPTER" if improved else (
                "FALLBACK TO RAW (NaN)" if adapter_nan else "FALLBACK TO RAW"
            )
            print(f"[Retouche] guard: adapter={adapter_score:.4f}  raw={raw_score:.4f}  → {tag}")

    def _guard_metric(self, proba: np.ndarray, y: np.ndarray) -> float:
        """1 − AUC for binary; log-loss otherwise. Lower is better."""
        if proba.shape[1] == 2:
            try:
                return 1.0 - roc_auc_score(y, proba[:, 1])
            except ValueError:
                pass
        return log_loss(y, proba, labels=np.arange(proba.shape[1]))


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def _stratified_sample(rng: np.random.RandomState, y: np.ndarray, n: int) -> np.ndarray:
    """Sample n indices stratified by y. If n ≥ len(y), return a permutation
    of all indices."""
    N = len(y)
    if n >= N:
        return rng.permutation(N)
    classes, counts = np.unique(y, return_counts=True)
    quotas = np.maximum(1, np.round(counts / counts.sum() * n).astype(int))
    # Trim/extend so they sum to exactly n
    diff = n - quotas.sum()
    if diff != 0:
        order = np.argsort(-counts)
        i = 0
        while diff != 0:
            k = order[i % len(order)]
            if diff > 0:
                quotas[k] += 1
                diff -= 1
            elif quotas[k] > 1:
                quotas[k] -= 1
                diff += 1
            i += 1
    out = []
    for c, q in zip(classes, quotas):
        idxs = np.where(y == c)[0]
        pick = rng.choice(idxs, size=min(q, len(idxs)), replace=False)
        out.append(pick)
    return rng.permutation(np.concatenate(out))


def _to_numpy(X):
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X.values
    except ImportError:
        pass
    return np.asarray(X)


def _resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


# ─────────────────────────────────────────────────────────────────────────────
# BatchEnsemble: K independently-seeded Retouche adapters per AG bag-fold
# ─────────────────────────────────────────────────────────────────────────────


def _offload_to_cpu(m: "RetoucheTabICLClassifier") -> None:
    """Move a fitted Retouche member's GPU state to CPU to free memory."""
    if hasattr(m, "adapter_"):
        m.adapter_.cpu()
    if hasattr(m, "preproc_"):
        m.preproc_.cpu()
    if hasattr(m, "_proj_") and m._proj_ is not None:
        m._proj_ = m._proj_.cpu()
    # Backbone (TabICL): both diff path and raw path can hold GPU weights.
    if hasattr(m, "clf_") and getattr(m.clf_, "model_", None) is not None:
        try:
            m.clf_.model_.cpu()
        except Exception:
            pass
        # Drop diff-input train cache (the big GPU tensors from
        # fit_with_differentiable_input)
        for attr in ("X_train_", "y_train_", "diff_mean_", "diff_std_"):
            if hasattr(m.clf_, attr):
                try:
                    setattr(m.clf_, attr, getattr(m.clf_, attr).detach().cpu())
                except Exception:
                    pass
    if hasattr(m, "_raw_clf_") and m._raw_clf_ is not None and getattr(m._raw_clf_, "model_", None) is not None:
        try:
            m._raw_clf_.model_.cpu()
        except Exception:
            pass
    # Cached refit tensor on the wrapper
    if hasattr(m, "_train_y_full_t_") and m._train_y_full_t_ is not None:
        try:
            m._train_y_full_t_ = m._train_y_full_t_.detach().cpu()
        except Exception:
            pass


def _restore_to_device(m: "RetoucheTabICLClassifier") -> None:
    """Move a previously offloaded member back to its original device."""
    dev = getattr(m, "device_", None)
    if dev is None:
        return
    if hasattr(m, "adapter_"):
        m.adapter_.to(dev)
    if hasattr(m, "preproc_"):
        m.preproc_.to(dev)
    if hasattr(m, "_proj_") and m._proj_ is not None:
        m._proj_ = m._proj_.to(dev)
    if hasattr(m, "clf_") and getattr(m.clf_, "model_", None) is not None:
        try:
            m.clf_.model_.to(dev)
        except Exception:
            pass
        for attr in ("X_train_", "y_train_", "diff_mean_", "diff_std_"):
            if hasattr(m.clf_, attr):
                try:
                    setattr(m.clf_, attr, getattr(m.clf_, attr).to(dev))
                except Exception:
                    pass
    if hasattr(m, "_raw_clf_") and m._raw_clf_ is not None and getattr(m._raw_clf_, "model_", None) is not None:
        try:
            m._raw_clf_.model_.to(dev)
        except Exception:
            pass
    if hasattr(m, "_train_y_full_t_") and m._train_y_full_t_ is not None:
        try:
            m._train_y_full_t_ = m._train_y_full_t_.to(dev)
        except Exception:
            pass


class BaggedRetoucheTabICLClassifier(BaseEstimator, ClassifierMixin):
    """K independently-seeded RetoucheTabICLClassifier adapters, prediction
    averaged across them.

    Paper Appendix F suggests "BETA-style bagging of the trained adapter at
    inference to further damp prediction variance." This is the cheap version:
    K full Retouche fits with seeds (random_state, random_state+1, ...) and
    inference-time averaging of class probabilities.

    The frozen backbone weights are shared across all K (they don't depend on
    the seed); only the adapter+preproc parameters are independent. The
    identity guard runs *per-member*; each member can independently fall back
    to raw.

    Parameters
    ----------
    n_estimators : int, default=4
        Number of independent adapters to fit and average.
    **kwargs :
        Forwarded to each RetoucheTabICLClassifier instance.
    """

    def __init__(self, n_estimators: int = 4, **kwargs):
        self.n_estimators = n_estimators
        self.kwargs = kwargs

    def fit(self, X, y, X_val=None, y_val=None):
        base_seed = self.kwargs.get("random_state", 42) or 42
        self.members_ = []
        for k in range(self.n_estimators):
            seed = base_seed + 1000 * k
            kw = dict(self.kwargs)
            kw["random_state"] = seed
            m = RetoucheTabICLClassifier(**kw)
            m.fit(X, y, X_val=X_val, y_val=y_val)
            # Offload backbones+state to CPU to free GPU between members
            # (the next member's fit reallocates a fresh TabICL on GPU).
            _offload_to_cpu(m)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.members_.append(m)
        self.classes_ = self.members_[0].classes_
        self.n_classes_ = self.members_[0].n_classes_
        self.label_encoder_ = self.members_[0].label_encoder_
        # Ensure outer AG fold-holder receives a CPU-resident bagged model.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc; gc.collect()
            torch.cuda.empty_cache()
        return self

    def predict_proba(self, X):
        probs = []
        for m in self.members_:
            _restore_to_device(m)
            probs.append(m.predict_proba(X))
            _offload_to_cpu(m)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.mean(np.stack(probs, axis=0), axis=0)

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder_.inverse_transform(idx)
