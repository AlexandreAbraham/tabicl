"""Adaptive fine-tuning wrapper that adds NAM and / or NoiseFilm adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted
from torch import nn

from tabicl._finetune.classifier import FinetunedTabICLClassifier
from tabicl._model.tabicl import TabICL
from tabicl._model.nam import NAMEncoder
from tabicl._model.noise_adapter import NoiseFilm
from tabicl._sklearn.preprocessing import EnsembleGenerator


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

    # ---- noise-aware predict path ----

    def _predict_proba_with_noise(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        noise_train: Optional[np.ndarray],
        noise_val: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """One-shot forward through the model with per-row noise threaded in.

        Bypasses the official ensemble-generator predict path. Builds a single
        normalization view, concatenates ``X_train`` + ``X_val`` along the row
        axis (with noise likewise concatenated), and runs one
        :meth:`TabICL.forward` to produce probabilities on the val rows.

        Returns
        -------
        proba : (n_val, n_classes)
        classes_ : (n_classes,)
        """
        if not self.use_noise:
            raise RuntimeError("_predict_proba_with_noise called without use_noise=True")

        if noise_train is None or noise_val is None:
            raise ValueError(
                "Both noise_train and noise_val must be provided for the "
                "noise-aware predict path."
            )

        device = next(self.model_.parameters()).device

        # Single-norm ensemble generator for preprocessing consistency. We pick
        # one (the default 'none' normalization with a single estimator) to
        # keep the forward shape (B=1, T, H) — no per-norm ensembling here.
        gen = EnsembleGenerator(
            classification=True,
            n_estimators=1,
            norm_methods=self.norm_methods,
            feat_shuffle_method="none",
            class_shuffle_method="none",
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        gen.fit(X_train, y_train)
        variants = gen.transform(X_val, mode="both")
        # variants is a dict; take the first entry deterministically.
        norm_method, (X_variant, y_variant) = next(iter(variants.items()))
        # X_variant: (E=1, T, H); y_variant: (E=1, train_size)
        X_t = torch.from_numpy(X_variant).float().to(device)
        y_t = torch.from_numpy(y_variant).float().to(device)
        n_train = X_train.shape[0]
        n_val = X_val.shape[0]
        T = n_train + n_val
        assert X_t.shape[1] == T, f"expected T={T}, got {X_t.shape[1]}"

        # Build (E=1, T) noise tensor matching X_t row order: train block then val block.
        noise_full = np.concatenate([noise_train, noise_val], axis=0).astype(np.float32)
        noise_t = torch.from_numpy(noise_full).unsqueeze(0).to(device)

        self.model_.eval()
        with torch.no_grad():
            out = self.model_(
                X=X_t,
                y_train=y_t,
                noise=noise_t,
                return_logits=False,
                softmax_temperature=self.softmax_temperature,
            )
        # out shape: (E=1, n_val, max_classes). Slice to actual class count.
        n_classes = int(self._label_encoder_.classes_.shape[0])
        probs = out[0, :, :n_classes].float().cpu().numpy()
        # Renormalize after slicing (sliced logits may not sum to 1 after softmax).
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs, np.arange(n_classes)

    # ---- public surface ----

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        output_dir: Optional[str | Path] = None,
        noise=None,
        noise_val=None,
    ) -> "AdaptiveFinetunedTabICLClassifier":
        """Fine-tune with optional per-row noise from a teacher model.

        See :meth:`FinetunedTabICLBase.fit` for the base signature. The
        ``noise`` (shape ``(n_samples,)``) and ``noise_val`` arguments are
        per-row scalars threaded through to the :class:`NoiseFilm` adapter
        when ``use_noise=True``. Must both be supplied if ``use_noise=True``
        and ``X_val`` is also supplied; otherwise an auto-split also splits
        ``noise``.
        """
        if self.use_noise and noise is None:
            raise ValueError(
                "use_noise=True but no `noise` array passed to fit(). "
                "Pre-compute per-row teacher scores (e.g. OOF predictions) "
                "and pass them as fit(X, y, noise=...)"
            )
        if not self.use_noise and noise is not None:
            raise ValueError(
                "noise was passed but use_noise=False — set use_noise=True "
                "to enable the FiLM adapter that consumes the noise signal."
            )
        return super().fit(
            X, y, X_val=X_val, y_val=y_val, output_dir=output_dir,
            noise=noise, noise_val=noise_val,
        )

    def predict_proba(self, X, noise_test=None):
        """Predict class probabilities, optionally conditioned on per-row noise.

        When ``use_noise=True``, ``noise_test`` (shape ``(n_samples,)``)
        must be supplied — typically teacher predictions on ``X``. Uses the
        custom one-shot predict path that bypasses the inner ensemble
        estimator; no ensemble averaging is done.

        When ``use_noise=False``, this falls back to the official
        :class:`FinetunedTabICLClassifier.predict_proba` which goes through
        the ensemble generator.
        """
        if not self.use_noise:
            if noise_test is not None:
                raise ValueError(
                    "noise_test was provided but use_noise=False — the model "
                    "has no NoiseFilm adapter to consume it."
                )
            return super().predict_proba(X)

        if noise_test is None:
            raise ValueError(
                "use_noise=True requires noise_test at predict time. "
                "Compute teacher predictions on X (e.g. teacher.predict_proba(X)[:, 1]) "
                "and pass them as noise_test=..."
            )

        check_is_fitted(self, "noise_raw_")
        X_arr = np.asarray(X)
        noise_test_arr = np.asarray(noise_test, dtype=np.float32)
        if noise_test_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"noise_test has length {noise_test_arr.shape[0]} but X has {X_arr.shape[0]} rows"
            )
        proba, _ = self._predict_proba_with_noise(
            X_train=np.asarray(self.X_raw_),
            y_train=np.asarray(self._label_encoder_.transform(self.y_raw_)).astype(np.int64),
            X_val=X_arr,
            noise_train=self.noise_raw_,
            noise_val=noise_test_arr,
        )
        return proba

    @property
    def classes_(self):
        check_is_fitted(self, "_label_encoder_")
        return self._label_encoder_.classes_
