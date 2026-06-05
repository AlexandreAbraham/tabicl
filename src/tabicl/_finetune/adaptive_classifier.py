"""Adaptive fine-tuning wrapper that adds NAM and / or NoiseFilm adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch import nn

from tabicl._finetune.classifier import FinetunedTabICLClassifier
from tabicl._model.tabicl import TabICL
from tabicl._model.nam import NAMEncoder
from tabicl._model.noise_adapter import NoiseFilm


class AdaptiveFinetunedTabICLClassifier(FinetunedTabICLClassifier):
    """Fine-tune TabICL with optional NAM and noise-FiLM adapters.

    Both adapters are native ``nn.Module`` children of ``TabICL``:

    * **NAM** (``use_nam=True``) sits before ``col_embedder`` and learns a
      per-feature MLP + categorical embedding from raw ``(B, T, H)`` tables.
      Initialized as identity (gate=0) so a freshly-inserted NAM has zero
      impact on the pretrained backbone at the start of training.

    * **Noise FiLM** (``use_noise=True``) sits between ``col_embedder`` and
      ``row_interactor`` and applies a gated FiLM modulation on the
      ``(B, T, G+C, E)`` embeddings, conditioned on a per-table scalar
      (``None`` defaults to zeros — for plain FT). Identity at init.

    Both adapters are trained jointly with the (optionally frozen) backbone
    via the standard fine-tuning loop. No differentiable-input path needed.

    Parameters
    ----------
    use_nam : bool, default=False
        Insert a learnable :class:`NAMEncoder` before ``col_embedder``.
    use_noise : bool, default=False
        Insert a learnable :class:`NoiseFilm` between ``col_embedder`` and
        ``row_interactor``.
    nam_hidden : int, default=32
        Hidden width of each per-feature MLP inside NAM.
    nam_n_channels : int, default=8
        Output channels per feature in NAM (before the mixer).
    nam_n_layers : int, default=2
        Total Linear layers per per-feature MLP (>=1).
    nam_dropout : float, default=0.0
        Dropout applied to NAM's residual delta.
    noise_film_hidden : int, default=32
        Hidden width of the NoiseFilm conditioning MLP.
    freeze_nam : bool, default=False
        Freeze the NAM adapter (does nothing if ``use_nam=False``).
    freeze_noise : bool, default=False
        Freeze the NoiseFilm adapter (does nothing if ``use_noise=False``).

    All other parameters are forwarded to :class:`FinetunedTabICLClassifier`.
    """

    def __init__(
        self,
        *,
        # Adapter switches
        use_nam: bool = False,
        use_noise: bool = False,
        # NAM hyperparams
        nam_hidden: int = 32,
        nam_n_channels: int = 8,
        nam_n_layers: int = 2,
        nam_dropout: float = 0.0,
        # NoiseFilm hyperparams
        noise_film_hidden: int = 32,
        # Adapter freezing
        freeze_nam: bool = False,
        freeze_noise: bool = False,
        # Everything else forwarded to FinetunedTabICLClassifier
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.use_nam = use_nam
        self.use_noise = use_noise
        self.nam_hidden = nam_hidden
        self.nam_n_channels = nam_n_channels
        self.nam_n_layers = nam_n_layers
        self.nam_dropout = nam_dropout
        self.noise_film_hidden = noise_film_hidden
        self.freeze_nam = freeze_nam
        self.freeze_noise = freeze_noise

    # ---- adapter construction ----

    def _build_nam(self, n_features: int) -> NAMEncoder:
        """Construct a NAM sized to the loaded dataset's feature count.

        Categorical columns are not auto-detected here; pretrained TabICL
        consumes already-numeric inputs (the upstream sklearn preprocessing
        pipeline ordinal-encodes categoricals before they reach the model).
        We treat all columns as numeric — the per-feature MLPs still learn
        usable representations from the encoded integers.
        """
        return NAMEncoder(
            n_features=n_features,
            cat_idx=None,
            cat_cardinalities=None,
            hidden=self.nam_hidden,
            n_channels=self.nam_n_channels,
            n_layers=self.nam_n_layers,
            h_out=n_features,
            mix=True,
            dropout=self.nam_dropout,
        )

    def _build_noise_film(self, embed_dim: int) -> NoiseFilm:
        return NoiseFilm(embed_dim=embed_dim, hidden=self.noise_film_hidden)

    # ---- lifecycle overrides ----

    def _load_pretrained(self, device: torch.device) -> None:
        """Load the pretrained backbone, then attach the requested adapters.

        Adapters live as direct children of ``self.model_`` (``model_.nam`` and
        ``model_.noise_film``). The backbone's ``state_dict`` is therefore the
        canonical place to save and reload the adapter weights — no extra
        plumbing required for checkpoints, DDP, or the inner estimator.
        """
        super()._load_pretrained(device)

        if self.use_nam:
            n_features = int(self.X_raw_.shape[1])
            self.model_.nam = self._build_nam(n_features).to(device)

        if self.use_noise:
            embed_dim = int(self.model_config_["embed_dim"])
            self.model_.noise_film = self._build_noise_film(embed_dim).to(device)

    def _frozen_submodules(self, model: TabICL) -> list[nn.Module]:
        """Extend the base freeze set with NAM and NoiseFilm when requested."""
        frozen = super()._frozen_submodules(model)
        if self.use_nam and self.freeze_nam and model.nam is not None:
            frozen.append(model.nam)
        if self.use_noise and self.freeze_noise and model.noise_film is not None:
            frozen.append(model.noise_film)
        return frozen
