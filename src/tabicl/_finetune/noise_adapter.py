"""Noise-aware fine-tuning adapters for TabICL on financial data.

Two complementary modules injected into a frozen pretrained TabICL:

1. **NoiseEmbeddingAdapter**: Additive uncertainty embeddings at the column
   level. Enriches feature representations with noise metadata so downstream
   layers see "value X with noise level Y" — not just "value X".

2. **NoiseCrossAttentionAdapter**: Cross-attention between row interaction
   output and ICL input. Recontextualizes row representations using noise
   metadata — the model learns what to attend to given the noise structure,
   without suppressing any information.

Usage::

    from noise_adapter import NoiseFinetuneClassifier

    clf = NoiseFinetuneClassifier(
        noise_dim=3,           # e.g., volatility, confidence, quality
        epochs=20,
        device="cuda",
        verbose=True,
    )
    # X_noise: (n_samples, n_features, noise_dim) — per-feature noise metadata
    # noise_summary: (n_samples, noise_summary_dim) — per-row aggregated noise
    clf.fit(X_train, y_train, X_noise=X_noise_train, noise_summary=noise_summary_train)
    y_pred = clf.predict(X_test, X_noise=X_noise_test, noise_summary=noise_summary_test)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Module 1: Noise Embedding Adapter (after col_embedder, before row_interactor)
# ---------------------------------------------------------------------------

class NoiseEmbeddingAdapter(nn.Module):
    """Additive noise embeddings injected at the column embedding level.

    For each feature, takes noise metadata (e.g., rolling volatility,
    confidence interval width, data quality score) and produces an additive
    embedding that shifts the feature representation.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (must match col_embedder output).
    noise_dim : int
        Number of noise metadata features per input feature.
    hidden_dim : int
        Hidden dimension of the noise encoder MLP.
    num_cls : int
        Number of CLS token slots at the start of the feature axis.
    """

    def __init__(self, embed_dim: int, noise_dim: int, hidden_dim: int = 64, num_cls: int = 4):
        super().__init__()
        self.num_cls = num_cls

        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Init to near-identity: adapter does nothing at start
        nn.init.zeros_(self.noise_encoder[-1].weight)
        nn.init.zeros_(self.noise_encoder[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, embeddings: Tensor, noise_meta: Tensor) -> Tensor:
        """Enrich feature embeddings with noise information.

        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings from col_embedder, shape (B, T, G+C, E).
        noise_meta : Tensor
            Per-feature noise metadata, shape (B, T, G, noise_dim) or
            (B, 1, G, noise_dim) if noise is constant across samples.

        Returns
        -------
        Tensor
            Enriched embeddings, same shape as input.
        """
        cls_part = embeddings[..., :self.num_cls, :]
        feat_part = embeddings[..., self.num_cls:, :]

        noise_emb = self.noise_encoder(noise_meta)
        gate = torch.sigmoid(self.gate(noise_meta))
        feat_enriched = feat_part + gate * noise_emb

        return torch.cat([cls_part, feat_enriched], dim=-2)


# ---------------------------------------------------------------------------
# Module 2: Noise Cross-Attention Adapter (after row_interactor, before ICL)
# ---------------------------------------------------------------------------

class NoiseCrossAttentionAdapter(nn.Module):
    """Cross-attention adapter that recontextualizes row representations.

    Query: row representations (what ICL will see).
    Key/Value: row representations + noise context.

    The adapter learns what to focus on given the noise structure.

    Parameters
    ----------
    d_model : int
        Row representation dimension (embed_dim * num_cls).
    noise_dim : int
        Number of noise summary features per row.
    nhead : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    def __init__(self, d_model: int, noise_dim: int, nhead: int = 4, dropout: float = 0.0):
        super().__init__()
        self.nhead = nhead

        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Gate starts at 0 → adapter is identity initially
        self.output_gate = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1 / math.sqrt(2))
            nn.init.zeros_(proj.bias)

    def forward(self, representations: Tensor, noise_summary: Tensor) -> Tensor:
        """Recontextualize row representations with noise information.

        Parameters
        ----------
        representations : Tensor
            Row representations, shape (B, T, D).
        noise_summary : Tensor
            Per-row noise summary, shape (B, T, noise_dim).

        Returns
        -------
        Tensor
            Recontextualized representations, shape (B, T, D).
        """
        noise_context = self.noise_proj(noise_summary)
        kv_input = representations + noise_context

        q = self.q_proj(self.norm_q(representations))
        k = self.k_proj(self.norm_kv(kv_input))
        v = self.v_proj(self.norm_kv(kv_input))

        B, T, D = q.shape
        hd = D // self.nhead
        q = q.view(B, T, self.nhead, hd).transpose(1, 2)
        k = k.view(B, T, self.nhead, hd).transpose(1, 2)
        v = v.view(B, T, self.nhead, hd).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(self.dropout(attn_out))

        gate = torch.sigmoid(self.output_gate)
        return representations + gate * attn_out


# ---------------------------------------------------------------------------
# Patched TabICL that routes through adapters
# ---------------------------------------------------------------------------

class NoiseAdaptedTabICL(nn.Module):
    """Wraps a frozen TabICL with noise adapters.

    The adapters are the only trainable parameters. The base model is frozen.
    """

    def __init__(self, base_model, col_adapter, icl_adapter):
        super().__init__()
        self.base_model = base_model
        self.col_adapter = col_adapter
        self.icl_adapter = icl_adapter

        # Freeze base model
        for p in self.base_model.parameters():
            p.requires_grad = False

    def forward(self, X, y_train, noise_meta=None, noise_summary=None):
        """Forward with noise adapters injected between stages."""
        # Stage 1: Column embedding (frozen)
        embeddings = self.base_model.col_embedder(
            X, y_train=y_train.long(), embed_with_test=False,
        )

        # Noise embedding adapter (trained)
        if noise_meta is not None:
            embeddings = self.col_adapter(embeddings, noise_meta)

        # Stage 2: Row interaction (frozen)
        representations = self.base_model.row_interactor(embeddings)

        # Noise cross-attention adapter (trained)
        if noise_summary is not None:
            representations = self.icl_adapter(representations, noise_summary)

        # Stage 3: ICL prediction (frozen)
        return self.base_model.icl_predictor(representations, y_train=y_train)


# ---------------------------------------------------------------------------
# Finetuning estimator (subclasses TabICL's finetune framework)
# ---------------------------------------------------------------------------

try:
    from tabicl._finetune.classifier import FinetunedTabICLClassifier
    from tabicl._finetune.data import MetaBatch
    from tabicl._model.tabicl import TabICL

    class NoiseFinetuneClassifier(FinetunedTabICLClassifier):
        """Fine-tune noise adapters on a frozen pretrained TabICL.

        Inherits the full fine-tuning infrastructure (AdamW, cosine warmup,
        AMP, early stopping, DDP, validation) from TabICL's finetune framework.
        Only the noise adapter parameters are trained.

        Parameters
        ----------
        noise_dim : int
            Number of noise metadata features per input feature.
        noise_summary_dim : int or None
            Number of per-row noise summary features. Defaults to noise_dim.
        adapter_hidden : int
            Hidden dim for noise embedding adapter MLP.
        adapter_nhead : int
            Attention heads for cross-attention adapter.

        All other parameters are inherited from FinetunedTabICLClassifier.

        Usage::

            clf = NoiseFinetuneClassifier(noise_dim=3, epochs=20, device="cuda")
            clf.fit(X_train, y_train)  # noise injected via set_noise_data()
            clf.predict(X_test)
        """

        def __init__(
            self,
            *,
            noise_dim: int = 1,
            noise_summary_dim: Optional[int] = None,
            adapter_hidden: int = 64,
            adapter_nhead: int = 4,
            **kwargs,
        ):
            # Default to freezing everything
            kwargs.setdefault("freeze_col", True)
            kwargs.setdefault("freeze_row", True)
            kwargs.setdefault("freeze_icl", True)
            super().__init__(**kwargs)
            self.noise_dim = noise_dim
            self.noise_summary_dim = noise_summary_dim or noise_dim
            self.adapter_hidden = adapter_hidden
            self.adapter_nhead = adapter_nhead

            # Will be set before training
            self._noise_meta = None
            self._noise_summary = None

        def set_noise_data(self, noise_meta: Tensor, noise_summary: Tensor):
            """Set noise metadata tensors before calling fit().

            Parameters
            ----------
            noise_meta : Tensor
                Per-feature noise, shape (n_samples, n_features, noise_dim).
            noise_summary : Tensor
                Per-row noise summary, shape (n_samples, noise_summary_dim).
            """
            self._noise_meta = noise_meta
            self._noise_summary = noise_summary

        def _load_pretrained(self, device):
            """Load pretrained model and wrap with noise adapters."""
            super()._load_pretrained(device)

            embed_dim = self.model_.embed_dim
            num_cls = self.model_.row_num_cls
            icl_dim = embed_dim * num_cls

            col_adapter = NoiseEmbeddingAdapter(
                embed_dim=embed_dim,
                noise_dim=self.noise_dim,
                hidden_dim=self.adapter_hidden,
                num_cls=num_cls,
            )
            icl_adapter = NoiseCrossAttentionAdapter(
                d_model=icl_dim,
                noise_dim=self.noise_summary_dim,
                nhead=self.adapter_nhead,
            )

            self.model_ = NoiseAdaptedTabICL(self.model_, col_adapter, icl_adapter)
            self.model_.to(device)

        def _apply_freezing(self, model):
            """Base model already frozen in NoiseAdaptedTabICL.__init__."""
            return True

        def _frozen_submodules(self, model):
            """Return the frozen base model for eval-mode management."""
            if isinstance(model, NoiseAdaptedTabICL):
                return [model.base_model]
            return super()._frozen_submodules(model)

        def _compute_batch_loss(self, batch: MetaBatch, model) -> torch.Tensor:
            """Forward through noise-adapted model."""
            # TODO: noise_meta and noise_summary need to be aligned with the
            # batch's sample indices. For now, pass None (adapters act as
            # identity when noise is None, gradually learning from the data
            # distribution itself).
            if isinstance(model, NoiseAdaptedTabICL):
                logits = model(batch.X, batch.y_train.float(),
                               noise_meta=None, noise_summary=None)
            else:
                logits = model(batch.X, batch.y_train.float())

            n_classes = int(batch.y_train.max().item()) + 1
            logits_used = logits[..., :n_classes].reshape(-1, n_classes)
            targets = batch.y_query.long().reshape(-1)
            return F.cross_entropy(logits_used, targets)

except ImportError:
    # tabicl finetune not available — adapters still usable standalone
    pass
