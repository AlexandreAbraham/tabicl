"""Unified TabICL adaptation estimator.

One sklearn-compatible class that covers:

  Architecture (constructor-only):
    * vanilla TabICL
    * + NAM2a preprocessor (input-space, per-feature MLP + slots + mixer to d_raw)
    * + Noise adapter (post-col-embed FiLM gate + post-row-interact cross-attn)
    * + Both

  Training modes (methods called after construction):
    * .fit(X, y)                          — plain ICL (no gradients)
    * .fine_tune(X, y, freeze=..., ...)   — single-dataset training
    * .cpt(datasets, freeze=..., ...)     — multi-dataset continued pretraining
    * .predict_proba(X), .predict(X)      — use current state

The `freeze=` argument on training methods takes either a list of module names
(``col_embedder``, ``row_interactor``, ``icl_predictor``, ``nam``, ``noise``)
or a shorthand (``"all"``, ``"none"``, ``"backbone"``, ``"adapters"``).

Defaults mirror successful experiments in the ASOS POC and the NAM2a 30-OpenML
bench.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .classifier import TabICLClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Vendored noise-adapter modules (originally in
# neuralk_model.training.modules.noise_adapter — vendored here to avoid the
# cross-repo dependency).
# ─────────────────────────────────────────────────────────────────────────────


class NoiseEmbeddingAdapter(nn.Module):
    """Additive gated embedding injected after TabICL's col_embedder.

    Sits between col_embed → row_interact. FiLM-style: encodes per-feature
    noise metadata into an additive shift, gated by sigmoid(MLP(noise)) so the
    adapter starts as a small, learnable perturbation and grows / shrinks via
    backprop. When ``noise_meta`` is all zeros, the adapter is still in the
    autograd graph (the gate ≈ 0.5 at init, encoder output ≈ 0).
    """

    def __init__(self, embed_dim: int, noise_dim: int, hidden_dim: int = 64,
                 num_cls: int = 4):
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

    def forward(self, embeddings: torch.Tensor,
                noise_meta: torch.Tensor) -> torch.Tensor:
        cls_part = embeddings[..., :self.num_cls, :]
        feat_part = embeddings[..., self.num_cls:, :]
        noise_emb = self.noise_encoder(noise_meta)
        gate = torch.sigmoid(self.gate(noise_meta))
        feat_enriched = feat_part + gate * noise_emb
        return torch.cat([cls_part, feat_enriched], dim=-2)


class NoiseCrossAttentionAdapter(nn.Module):
    """Cross-attention adapter injected after TabICL's row_interactor.

    Sits between row_interact → icl_predictor. Q from row representations,
    KV from row representations + noise context. ``output_gate`` starts at
    zero so the adapter is identity at init, with gradients flowing through
    the gate parameter.
    """

    def __init__(self, d_model: int, noise_dim: int, nhead: int = 4,
                 dropout: float = 0.0):
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
        self.output_gate = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj.weight, gain=1 / math.sqrt(2))
            nn.init.zeros_(proj.bias)

    def forward(self, representations: torch.Tensor,
                noise_summary: torch.Tensor) -> torch.Tensor:
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


# ─────────────────────────────────────────────────────────────────────────────
# NAM2a unified encoder (port of the winning variant from feature/retouche).
# ─────────────────────────────────────────────────────────────────────────────


class _UnifiedNAMEncoder(nn.Module):
    """Per-feature MLP for numerics + entity embedding for cats, then a single
    dense mixer collapses to d_out. ``n_slots`` gives k independent embeddings
    per feature (Cabin-style) that are concatenated into the mixer input.

    Input  (N, d_num) numerics + list of (N,) cat ID tensors.
    Output (N, d_out).
    """

    def __init__(self, d_num: int, cat_cardinalities: list[int],
                 hidden: int = 32, n_channels: int = 32, feat_layers: int = 2,
                 cat_emb_dim: int = 16, d_out: int = 64, dropout: float = 0.2,
                 n_slots: int = 2):
        super().__init__()
        self.d_num = d_num
        self.n_cat = len(cat_cardinalities)
        self.hidden = hidden
        self.n_channels = n_channels
        self.n_slots = n_slots
        self._d_out = d_out

        def _make_num_mlp():
            layers = [nn.Linear(1, hidden), nn.ReLU()]
            for _ in range(feat_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU()]
            layers.append(nn.Linear(hidden, n_channels))
            return nn.Sequential(*layers)
        self.num_mlps = nn.ModuleList(
            [_make_num_mlp() for _ in range(d_num * n_slots)]
        )
        self.cat_embs = nn.ModuleList()
        self.cat_projs = nn.ModuleList()
        for card in cat_cardinalities:
            actual_emb = max(2, min(cat_emb_dim, int(math.sqrt(card)) + 1))
            for _ in range(n_slots):
                emb = nn.Embedding(card, actual_emb)
                nn.init.normal_(emb.weight, std=0.05)
                self.cat_embs.append(emb)
                self.cat_projs.append(nn.Linear(actual_emb, n_channels))
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
        h = torch.cat(parts, dim=1)
        h = self.mix(h)
        h = F.relu(h)
        return self.dropout(h)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


_FREEZE_NAMES = ("col_embedder", "row_interactor", "icl_predictor", "nam", "noise")


def _normalize_freeze(freeze) -> set[str]:
    """Resolve a freeze= argument into a concrete set of module names."""
    if freeze is None or freeze == "none":
        return set()
    if isinstance(freeze, bool):
        return set(_FREEZE_NAMES) if freeze else set()
    if isinstance(freeze, str):
        if freeze == "all":
            return set(_FREEZE_NAMES)
        if freeze == "backbone":
            return {"col_embedder", "row_interactor", "icl_predictor"}
        if freeze == "adapters":
            return {"nam", "noise"}
        if freeze in _FREEZE_NAMES:
            return {freeze}
        raise ValueError(
            f"Unknown freeze keyword {freeze!r}. "
            f"Use one of: all, none, backbone, adapters, or a name in {_FREEZE_NAMES}."
        )
    if isinstance(freeze, (list, tuple, set)):
        out = set()
        for name in freeze:
            out |= _normalize_freeze(name)
        return out
    raise ValueError(f"Unsupported freeze= value: {freeze!r}")


def _resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def _to_numpy(X):
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X.values
    except ImportError:
        pass
    return np.asarray(X)


def _infer_cat_features(X, X_np, cat_features):
    """Resolve a cat_features argument to a list of column indices."""
    if cat_features is None:
        return []
    if cat_features != "auto":
        return list(cat_features)
    # auto: pandas object/category dtype, else low-cardinality int columns
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return [
                i for i, col in enumerate(X.columns)
                if X[col].dtype == object or str(X[col].dtype) == "category"
            ]
    except ImportError:
        pass
    cats = []
    for i in range(X_np.shape[1]):
        col = X_np[:, i]
        try:
            arr = col.astype(np.float64)
            if (np.all(np.isfinite(arr)) and np.all(arr == arr.astype(int))
                    and len(np.unique(arr)) <= 20):
                cats.append(i)
        except (ValueError, TypeError):
            cats.append(i)
    return cats


def _safe_str(v):
    if v is None:
        return "__nan__"
    if isinstance(v, float) and math.isnan(v):
        return "__nan__"
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# The estimator
# ─────────────────────────────────────────────────────────────────────────────


class TabICLAdaptClassifier(ClassifierMixin, BaseEstimator):
    """Unified TabICL adaptation classifier.

    Combines vanilla TabICL with optional architectural enhancements (NAM2a
    input-space preproc and/or post-col-embed noise adapter), exposed through
    a single sklearn-style API:

      * ``__init__`` sets architecture only.
      * ``.fit(X, y)`` runs plain ICL (no gradient updates) — valid only when
        no adapters are enabled.
      * ``.fine_tune(X, y, …)`` trains on a single dataset, with explicit
        freezing knobs.
      * ``.cpt(datasets, …)`` runs continued pretraining over multiple
        datasets, with low default LR.
      * ``.predict_proba(X)`` uses the current model state.

    Parameters
    ----------
    use_nam2a : bool, default=False
        Enable the NAM2a input-space preprocessor (per-feature MLP + slots +
        mixer collapsing to ``d_raw``).
    use_noise_adapter : bool, default=False
        Enable the noise adapter (FiLM gate after col_embedder, cross-attn
        after row_interactor).
    nam_slots, nam_hidden, nam_d_out, nam_dropout :
        NAM2a hyperparameters. Defaults are the winning NAM2a config from the
        30-OpenML bench. ``nam_d_out="auto"`` keeps the post-preproc width
        equal to the raw feature count (or 8, whichever is larger).
    noise_dim, adapter_hidden, adapter_nhead :
        Noise-adapter hyperparameters. Match the values used in ASOS exp 13/14.
    cat_features : 'auto', list[int] or None, default='auto'
        Indices of categorical columns. Used only when NAM2a is enabled (it
        builds entity embeddings for these columns).
    device : torch device or None, default=None
        ``None`` auto-selects CUDA.
    random_state : int, default=42
        Seed for splits and shuffles.
    verbose : bool, default=False
        Print training progress.
    """

    def __init__(
        self,
        # Architecture switches
        use_nam2a: bool = False,
        use_noise_adapter: bool = False,
        # NAM2a hyperparams
        nam_slots: int = 2,
        nam_hidden: int = 32,
        nam_d_out: Union[int, Literal["auto"]] = "auto",
        nam_dropout: float = 0.2,
        cat_features: Union[Literal["auto"], list[int], None] = "auto",
        # Noise-adapter hyperparams
        noise_dim: int = 1,
        adapter_hidden: int = 64,
        adapter_nhead: int = 4,
        # Misc
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.use_nam2a = use_nam2a
        self.use_noise_adapter = use_noise_adapter
        self.nam_slots = nam_slots
        self.nam_hidden = nam_hidden
        self.nam_d_out = nam_d_out
        self.nam_dropout = nam_dropout
        self.cat_features = cat_features
        self.noise_dim = noise_dim
        self.adapter_hidden = adapter_hidden
        self.adapter_nhead = adapter_nhead
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ─── public API ─────────────────────────────────────────────────────

    def fit(self, X, y):
        """Plain in-context fit (no gradient updates).

        Valid only when no adapter is enabled. With adapters, call
        ``fine_tune`` or ``cpt`` instead — the adapters are randomly
        initialised and would corrupt predictions.
        """
        if self.use_nam2a or self.use_noise_adapter:
            raise RuntimeError(
                "TabICLAdaptClassifier.fit is for plain ICL only. "
                "With use_nam2a or use_noise_adapter enabled, call "
                "fine_tune(X, y) or cpt(datasets) instead so the adapter "
                "parameters get trained."
            )
        self._init_state(X, y)
        # Plain ICL → just stash a fitted TabICLClassifier on raw features.
        self._raw_clf_ = TabICLClassifier(
            device=str(self.device_),
            random_state=self.random_state,
            verbose=False,
        )
        self._raw_clf_.fit(self._train_X_np_, self._train_y_enc_)
        self._is_fitted_ = True
        self._mode_ = "icl"
        return self

    def fine_tune(
        self,
        X, y, *,
        X_val=None, y_val=None,
        freeze: Union[str, list, bool, None] = "backbone",
        lr: float = 5e-3,
        backbone_lr_factor: float = 0.1,
        weight_decay: float = 3e-3,
        epochs: int = 150,
        patience: int = 10,
        max_context: Union[int, Literal["auto"]] = "auto",
        label_smoothing: float = 0.15,
        max_grad_norm: float = 2.0,
        identity_guard_tol: float = 0.005,
    ):
        """Single-dataset fine-tune.

        Builds adapters (if requested), then trains via the differentiable
        forward through TabICL using the same context/query meta-batch loop
        as Retouche / NoiseFinetuneClassifier. ``freeze`` controls which
        modules receive gradients.

        Returns self.
        """
        self._init_state(X, y, X_val=X_val, y_val=y_val)
        frozen = _normalize_freeze(freeze)
        self._build_modules()
        self._apply_freezing(frozen)
        self._run_training_loop(
            mode="fine_tune",
            lr=lr,
            backbone_lr_factor=backbone_lr_factor,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            max_context=max_context,
            label_smoothing=label_smoothing,
            max_grad_norm=max_grad_norm,
        )
        # Identity guard: keep a pristine raw TabICL refit on full train, in
        # case our adapter underperforms.
        self._run_identity_guard(identity_guard_tol)
        self._is_fitted_ = True
        self._mode_ = "fine_tune"
        return self

    def cpt(
        self,
        datasets, *,
        freeze: Union[str, list, bool, None] = "none",
        lr: float = 1e-5,
        backbone_lr_factor: float = 1.0,   # at CPT lr the backbone is small enough
        weight_decay: float = 0.01,
        epochs: int = 30,
        patience: int = 10,
        max_context: Union[int, Literal["auto"]] = "auto",
        label_smoothing: float = 0.15,
        max_grad_norm: float = 1.0,
        sample_weights: Optional[list[float]] = None,
    ):
        """Continued pre-training on multiple datasets.

        Parameters
        ----------
        datasets : list[(X, y)]
            List of (X, y) tuples. Each iteration samples one dataset
            (uniformly or by ``sample_weights``) and runs a meta-batch step.
        freeze : default 'none'
            For CPT you usually want the whole stack trainable.
        lr : default 1e-5
            CPT-appropriate small learning rate.

        Returns self.
        """
        if not datasets:
            raise ValueError("cpt() requires a non-empty list of (X, y) datasets.")
        # Initialize state from the first dataset to pick up class set, etc.
        X0, y0 = datasets[0]
        self._init_state(X0, y0)
        frozen = _normalize_freeze(freeze)
        self._build_modules()
        self._apply_freezing(frozen)

        rng = np.random.RandomState(self.random_state)
        weights = (np.asarray(sample_weights, dtype=np.float64)
                   if sample_weights is not None else None)
        if weights is not None:
            weights = weights / weights.sum()

        self._run_training_loop_multi(
            datasets=datasets,
            mode="cpt",
            rng=rng,
            weights=weights,
            lr=lr,
            backbone_lr_factor=backbone_lr_factor,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            max_context=max_context,
            label_smoothing=label_smoothing,
            max_grad_norm=max_grad_norm,
        )
        self._is_fitted_ = True
        self._mode_ = "cpt"
        return self

    def predict_proba(self, X):
        if not getattr(self, "_is_fitted_", False):
            raise RuntimeError("Call fit, fine_tune, or cpt before predict_proba.")
        X_np = _to_numpy(X)
        if self._mode_ == "icl":
            return self._raw_clf_.predict_proba(X_np)

        # Adapter / FT / CPT path: use the trained model end-to-end through
        # the differentiable forward.
        if getattr(self, "_use_adapter_", True) is False:
            return self._raw_clf_.predict_proba(X_np)

        device = self.device_
        # Build train context tensor (with NAM2a preproc if enabled)
        x_tr_back = self._forward_preproc(self._train_X_np_)
        self._diff_clf_.fit_with_differentiable_input(
            x_tr_back, torch.from_numpy(self._train_y_enc_).long().to(device),
        )
        x_te_back = self._forward_preproc(X_np)
        with torch.no_grad():
            probs = self._diff_clf_.predict_differentiable(
                x_te_back, return_logits=False, softmax_temperature=0.9,
            )
        probs_np = probs.detach().cpu().numpy()
        if not np.all(np.isfinite(probs_np)):
            # Fallback: pristine raw TabICL
            return self._raw_clf_.predict_proba(X_np)
        return probs_np

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(idx)

    # ─── internals ──────────────────────────────────────────────────────

    def _init_state(self, X, y, X_val=None, y_val=None):
        X_np = _to_numpy(X)
        y_np = np.asarray(y)
        if y_val is not None:
            y_combined = np.concatenate([y_np, np.asarray(y_val)])
        else:
            y_combined = y_np
        self.label_encoder_ = LabelEncoder().fit(y_combined)
        y_enc = self.label_encoder_.transform(y_np)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ != 2:
            warnings.warn(
                f"TabICLAdaptClassifier benched only on binary tasks; got "
                f"{self.n_classes_} classes.",
            )

        self._cat_idx_ = _infer_cat_features(X, X_np, self.cat_features)
        self._num_idx_ = [i for i in range(X_np.shape[1]) if i not in self._cat_idx_]

        # Numeric impute + standardize (NAM2a path needs differentiable, stable inputs)
        if self._num_idx_:
            X_num = X_np[:, self._num_idx_].astype(np.float64)
            self._num_imputer_ = SimpleImputer(strategy="median").fit(X_num)
            self._num_scaler_ = StandardScaler().fit(self._num_imputer_.transform(X_num))
        else:
            self._num_imputer_ = None
            self._num_scaler_ = None

        # Categorical: per-column LabelEncoder + cardinality with +1 unknown slot
        self._cat_encoders_ = []
        self._cat_cards_ = []
        for j in self._cat_idx_:
            col = X_np[:, j]
            col_str = np.array([_safe_str(v) for v in col])
            le = LabelEncoder().fit(col_str)
            self._cat_encoders_.append(le)
            self._cat_cards_.append(len(le.classes_) + 1)

        self._train_X_np_ = X_np
        self._train_y_enc_ = y_enc.astype(np.int64)
        self._val_X_np_ = _to_numpy(X_val) if X_val is not None else None
        self._val_y_enc_ = (self.label_encoder_.transform(np.asarray(y_val))
                            if y_val is not None else None)
        self.device_ = _resolve_device(self.device)

    def _build_modules(self):
        """Build NAM2a / noise adapter / diff classifier per the architecture flags."""
        device = self.device_

        # NAM2a preproc
        if self.use_nam2a:
            d_target = (max(8, len(self._num_idx_) + len(self._cat_idx_))
                        if self.nam_d_out == "auto" else int(self.nam_d_out))
            self._nam_ = _UnifiedNAMEncoder(
                d_num=len(self._num_idx_),
                cat_cardinalities=self._cat_cards_,
                hidden=self.nam_hidden,
                n_channels=self.nam_hidden,
                feat_layers=2,
                cat_emb_dim=16,
                d_out=d_target,
                dropout=self.nam_dropout,
                n_slots=self.nam_slots,
            ).to(device)
            self._d_in_ = d_target
        else:
            self._nam_ = None
            self._d_in_ = len(self._num_idx_) + len(self._cat_idx_)

        # Noise adapter (built inside the diff classifier path; we just stash
        # the modules here so they're known to the optimizer / freezer).
        if self.use_noise_adapter:
            # Build placeholders; the actual injection into the TabICL forward
            # happens in _forward_preproc via a wrapped TabICLClassifier.
            # We need embed_dim from the loaded TabICL model — set lazily on
            # first forward.
            self._noise_col_ = "PENDING"   # built after we know embed_dim
            self._noise_icl_ = "PENDING"
        else:
            self._noise_col_ = None
            self._noise_icl_ = None

        # Differentiable-input classifier (does the diff forward through TabICL)
        clf_kwargs = dict(device=str(device), differentiable_input=True)
        self._diff_clf_ = TabICLClassifier(**clf_kwargs)

    def _build_noise_modules(self):
        """Build the actual NoiseEmbeddingAdapter and NoiseCrossAttentionAdapter
        once we know the backbone's embed_dim / row_num_cls.
        """
        if not self.use_noise_adapter or self._noise_col_ != "PENDING":
            return
        device = self.device_
        # _load_model populates clf_.model_ with the TabICL instance.
        self._diff_clf_._load_model()
        self._diff_clf_.model_.to(device)
        m = self._diff_clf_.model_
        embed_dim = m.embed_dim
        num_cls = m.row_num_cls
        icl_dim = embed_dim * num_cls
        self._noise_col_ = NoiseEmbeddingAdapter(
            embed_dim=embed_dim, noise_dim=self.noise_dim,
            hidden_dim=self.adapter_hidden, num_cls=num_cls,
        ).to(device)
        self._noise_icl_ = NoiseCrossAttentionAdapter(
            d_model=icl_dim, noise_dim=self.noise_dim,
            nhead=self.adapter_nhead,
        ).to(device)

    def _apply_freezing(self, frozen: set[str]):
        """Apply requires_grad flags based on the freeze set."""
        self._frozen_ = frozen

        # Make sure the backbone is loaded AND the diff classifier has been
        # primed via its "first call" branch in fit_with_differentiable_input.
        # That branch unconditionally sets every backbone param's requires_grad
        # to False — we have to let it run *before* applying our own freeze
        # flags, otherwise the user's freeze="none" gets silently overwritten
        # on the first forward call.
        if getattr(self._diff_clf_, "diff_mean_", None) is None:
            # First-time priming with a tiny dummy batch. Two rows are enough.
            d = self._d_in_
            dummy_X = torch.zeros(2, d, device=self.device_,
                                   dtype=torch.float32, requires_grad=False)
            dummy_y = torch.tensor([0, 1], device=self.device_, dtype=torch.long)
            self._diff_clf_.fit_with_differentiable_input(dummy_X, dummy_y)
            # The diff classifier just set requires_grad=False on all backbone
            # params and set diff_mean_/std_ from the dummy batch. We'll fix
            # both below: requires_grad here, diff_mean_/std_ on the next
            # *real* fit_with_differentiable_input call inside training.
            # Force diff_mean_/std_ back to "uninitialized" so the next real
            # call recomputes them from real data. The diff classifier uses
            # `hasattr(self, "diff_mean_")` to detect the first call, so we
            # have to delete (not None-set) the attributes.
            for attr in ("diff_mean_", "diff_std_"):
                if hasattr(self._diff_clf_, attr):
                    delattr(self._diff_clf_, attr)

        # If noise adapter is requested, build the modules now.
        if self.use_noise_adapter:
            self._build_noise_modules()

        # Backbone
        m = self._diff_clf_.model_
        for name, submodule_name in [
            ("col_embedder", "col_embedder"),
            ("row_interactor", "row_interactor"),
            ("icl_predictor", "icl_predictor"),
        ]:
            mod = getattr(m, submodule_name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = (name not in frozen)

        # NAM2a
        if self._nam_ is not None:
            for p in self._nam_.parameters():
                p.requires_grad = ("nam" not in frozen)

        # Noise adapter
        if self.use_noise_adapter and self._noise_col_ is not None:
            for p in self._noise_col_.parameters():
                p.requires_grad = ("noise" not in frozen)
            for p in self._noise_icl_.parameters():
                p.requires_grad = ("noise" not in frozen)

    def _backbone_param_groups(self, lr: float, backbone_lr_factor: float,
                                weight_decay: float):
        """Build optimizer param groups, with backbone params receiving
        lr × backbone_lr_factor (effective only for unfrozen backbone params).
        """
        backbone_mods = ["col_embedder", "row_interactor", "icl_predictor"]
        m = self._diff_clf_.model_

        backbone_params = []
        for sn in backbone_mods:
            mod = getattr(m, sn, None)
            if mod is None:
                continue
            for p in mod.parameters():
                if p.requires_grad:
                    backbone_params.append(p)

        adapter_params = []
        if self._nam_ is not None:
            adapter_params.extend(p for p in self._nam_.parameters() if p.requires_grad)
        if self.use_noise_adapter and self._noise_col_ is not None:
            adapter_params.extend(p for p in self._noise_col_.parameters() if p.requires_grad)
            adapter_params.extend(p for p in self._noise_icl_.parameters() if p.requires_grad)

        groups = []
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": lr * backbone_lr_factor,
                "weight_decay": weight_decay,
            })
        if adapter_params:
            groups.append({
                "params": adapter_params,
                "lr": lr,
                "weight_decay": weight_decay,
            })
        if not groups:
            raise RuntimeError(
                "Nothing to train: all parameters are frozen. "
                "Adjust freeze=… to leave at least one component unfrozen."
            )
        return groups

    def _forward_preproc(self, X_np: np.ndarray) -> torch.Tensor:
        """Build the tensor that goes into TabICL's differentiable forward.

        With NAM2a: numerics imputed+scaled, cats mapped to IDs, both fed
        through _UnifiedNAMEncoder.

        Without NAM2a: numerics imputed+scaled, cats ordinal-scaled, concatenated.
        """
        device = self.device_
        if self._num_idx_:
            X_num = X_np[:, self._num_idx_].astype(np.float64)
            X_num = self._num_imputer_.transform(X_num)
            X_num = self._num_scaler_.transform(X_num).astype(np.float32)
            X_num_t = torch.from_numpy(X_num).to(device)
        else:
            X_num_t = torch.empty(X_np.shape[0], 0, device=device)

        cat_ids = []
        for k, j in enumerate(self._cat_idx_):
            le = self._cat_encoders_[k]
            unk = len(le.classes_)
            known = set(le.classes_)
            col = np.array([_safe_str(v) for v in X_np[:, j]])
            mapped = np.where(np.isin(col, list(known)),
                              col, le.classes_[0] if len(le.classes_) else "")
            ids = np.where(np.isin(col, list(known)),
                           le.transform(mapped), unk).astype(np.int64)
            cat_ids.append(torch.from_numpy(ids).to(device))

        if self._nam_ is not None:
            return self._nam_(X_num_t, cat_ids)

        # No NAM2a: concatenate numerics + ordinal-scaled cat IDs
        parts = [X_num_t] if X_num_t.numel() else []
        for ids in cat_ids:
            parts.append(ids.unsqueeze(1).float())
        if not parts:
            raise ValueError("No features after preproc.")
        return torch.cat(parts, dim=1)

    def _diff_classifier_forward(self, x_back_train, y_train, x_back_query):
        """One full diff forward, returning logits. Routes through the noise
        adapter if enabled (by patching the TabICL forward stages here).
        """
        if not self.use_noise_adapter:
            self._diff_clf_.fit_with_differentiable_input(x_back_train, y_train)
            return self._diff_clf_.predict_differentiable(
                x_back_query, return_logits=True,
            )

        # With noise adapter: do the forward manually so we can inject adapters.
        # Pattern from ASOS exp 13/14 — pass zero noise, the adapter still has
        # gradient flow through its gate.
        # NOTE: This is a simplified path; for full performance we may want to
        # mirror TabICLClassifier.predict_differentiable's z-norm handling.
        # For now we rely on the diff classifier's own z-norm by calling it
        # in a wrapped mode.
        # First do the standard fit (sets z-norm stats on x_back_train).
        self._diff_clf_.fit_with_differentiable_input(x_back_train, y_train)
        # Build the combined train+query tensor the diff classifier expects:
        # we replicate the predict_differentiable z-norm + concat then route
        # through the wrapped TabICL forward.
        X_train = self._diff_clf_.X_train_
        X_train_normed = (X_train - self._diff_clf_.diff_mean_) / self._diff_clf_.diff_std_
        X_test_normed = (x_back_query - self._diff_clf_.diff_mean_) / self._diff_clf_.diff_std_
        X_combined = torch.cat([X_train_normed, X_test_normed], dim=0).unsqueeze(0).to(self.device_)
        y_train_t = self._diff_clf_.y_train_.unsqueeze(0).float().to(self.device_)

        m = self._diff_clf_.model_
        # Stage 1
        embeddings = m.col_embedder(X_combined, y_train=y_train_t, embed_with_test=False)
        # Inject col-side noise adapter (zero noise → adapter is identity-ish at init)
        B, T = embeddings.shape[:2]
        G = embeddings.size(-2) - m.row_num_cls
        noise_meta = torch.zeros(B, T, G, self.noise_dim,
                                  device=embeddings.device, dtype=embeddings.dtype)
        embeddings = self._noise_col_(embeddings, noise_meta)
        # Stage 2
        representations = m.row_interactor(embeddings)
        # Inject ICL-side noise adapter
        noise_summary = torch.zeros(B, T, self.noise_dim,
                                     device=representations.device,
                                     dtype=representations.dtype)
        representations = self._noise_icl_(representations, noise_summary)
        # Stage 3
        out = m.icl_predictor(representations, y_train=y_train_t)
        # Slice to actual num classes, like predict_differentiable does
        logits = out[0, :, :self.n_classes_]
        return logits

    def _run_training_loop(
        self, *, mode, lr, backbone_lr_factor, weight_decay, epochs, patience,
        max_context, label_smoothing, max_grad_norm,
    ):
        """Standard single-dataset fine-tune loop with context/query meta-batches."""
        device = self.device_
        n_train = len(self._train_y_enc_)
        y_train_np = self._train_y_enc_

        # Resolve max_context
        if max_context == "auto":
            max_ctx = min(n_train - 1, max(500, int(600_000 / max(1, self._d_in_))), 6000)
        elif max_context is None:
            max_ctx = n_train - 1
        else:
            max_ctx = min(n_train - 1, int(max_context))

        # Hold out a small ES slice
        rng = np.random.RandomState(self.random_state)
        n_es = max(0, min(int(0.1 * n_train), 1000))
        es_idx = (rng.choice(n_train, size=n_es, replace=False)
                  if n_es > 0 else np.empty(0, dtype=np.int64))
        es_mask = np.zeros(n_train, dtype=bool)
        es_mask[es_idx] = True
        pool_idx = np.where(~es_mask)[0]
        n_pool = len(pool_idx)
        max_ctx = min(max_ctx, max(1, n_pool - 2))

        # Optimizer
        param_groups = self._backbone_param_groups(lr, backbone_lr_factor, weight_decay)
        optimizer = torch.optim.AdamW(param_groups)

        best_loss = float("inf")
        best_state = None
        bad = 0
        for epoch in range(epochs):
            # Random C/Q from pool
            perm = rng.permutation(n_pool)
            ctx_local = perm[:max_ctx]
            q_local = perm[max_ctx:]
            if len(q_local) < 2:
                q_local = perm[-2:]
            ctx_global = pool_idx[ctx_local]
            q_global = pool_idx[q_local]

            self._set_train_mode(training=True)
            x_ctx = self._forward_preproc(self._train_X_np_[ctx_global])
            x_q = self._forward_preproc(self._train_X_np_[q_global])
            y_ctx = torch.from_numpy(y_train_np[ctx_global]).long().to(device)
            y_q = torch.from_numpy(y_train_np[q_global]).long().to(device)

            logits = self._diff_classifier_forward(x_ctx, y_ctx, x_q)
            loss = F.cross_entropy(logits, y_q, label_smoothing=label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            params_for_clip = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params_for_clip, max_grad_norm)
            optimizer.step()

            # Early-stop check on the ES slice (if present)
            if n_es > 0:
                self._set_train_mode(training=False)
                with torch.no_grad():
                    x_pool = self._forward_preproc(self._train_X_np_[pool_idx])
                    x_es = self._forward_preproc(self._train_X_np_[es_idx])
                    y_pool = torch.from_numpy(y_train_np[pool_idx]).long().to(device)
                    y_es = torch.from_numpy(y_train_np[es_idx]).long().to(device)
                    es_logits = self._diff_classifier_forward(x_pool, y_pool, x_es)
                    es_loss = float(F.cross_entropy(es_logits, y_es).item())
                if self.verbose:
                    print(f"[adapt {mode}] ep {epoch:3d}  q_loss={loss.item():.4f}  es_loss={es_loss:.4f}",
                          flush=True)
                if es_loss + 1e-6 < best_loss:
                    best_loss = es_loss
                    best_state = self._snapshot_state()
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        if self.verbose:
                            print(f"[adapt {mode}] early stop @ epoch {epoch}", flush=True)
                        break
            else:
                if self.verbose and epoch % 5 == 0:
                    print(f"[adapt {mode}] ep {epoch:3d}  q_loss={loss.item():.4f}",
                          flush=True)

        if best_state is not None:
            self._restore_state(best_state)

    def _run_training_loop_multi(
        self, *, datasets, mode, rng, weights, lr, backbone_lr_factor,
        weight_decay, epochs, patience, max_context, label_smoothing, max_grad_norm,
    ):
        """CPT loop: each epoch sample one dataset, take a single step."""
        device = self.device_
        param_groups = self._backbone_param_groups(lr, backbone_lr_factor, weight_decay)
        optimizer = torch.optim.AdamW(param_groups)

        # Re-encode each dataset's labels using a fresh LabelEncoder so they
        # all match self.classes_ (assumes same class set across datasets).
        encoded = []
        for X_i, y_i in datasets:
            Xi_np = _to_numpy(X_i)
            yi_enc = self.label_encoder_.transform(np.asarray(y_i))
            encoded.append((Xi_np, yi_enc))

        best_loss = float("inf")
        best_state = None
        bad = 0

        for epoch in range(epochs):
            # Sample dataset
            if weights is None:
                idx = rng.randint(0, len(encoded))
            else:
                idx = rng.choice(len(encoded), p=weights)
            X_d, y_d = encoded[idx]
            n_d = len(y_d)
            if n_d < 4:
                continue

            # Resolve per-step max_context for THIS dataset
            if max_context == "auto":
                max_ctx = min(n_d - 2, max(500, int(600_000 / max(1, self._d_in_))), 6000)
            elif max_context is None:
                max_ctx = n_d - 2
            else:
                max_ctx = min(n_d - 2, int(max_context))

            perm = rng.permutation(n_d)
            ctx_local = perm[:max_ctx]
            q_local = perm[max_ctx:]
            if len(q_local) < 2:
                q_local = perm[-2:]

            self._set_train_mode(training=True)
            x_ctx = self._forward_preproc(X_d[ctx_local])
            x_q = self._forward_preproc(X_d[q_local])
            y_ctx = torch.from_numpy(y_d[ctx_local]).long().to(device)
            y_q = torch.from_numpy(y_d[q_local]).long().to(device)

            logits = self._diff_classifier_forward(x_ctx, y_ctx, x_q)
            loss = F.cross_entropy(logits, y_q, label_smoothing=label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            params_for_clip = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params_for_clip, max_grad_norm)
            optimizer.step()

            # Patience based on running query loss
            if self.verbose and epoch % 5 == 0:
                print(f"[adapt {mode}] ep {epoch:3d} ds {idx}  q_loss={loss.item():.4f}",
                      flush=True)
            if loss.item() + 1e-6 < best_loss:
                best_loss = loss.item()
                best_state = self._snapshot_state()
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    if self.verbose:
                        print(f"[adapt {mode}] early stop @ epoch {epoch}", flush=True)
                    break

        if best_state is not None:
            self._restore_state(best_state)

    def _set_train_mode(self, training: bool):
        """Set train/eval mode on trainable modules, respecting freezes.

        Important: the TabICL backbone submodules stay in train mode at all
        times during training/eval. Putting them in eval mode would route their
        forward through ``_inference_forward``, which contains internal
        ``torch.no_grad()`` blocks (see ``model/inference.py``) — that
        permanently detaches gradient and breaks fine-tuning on the next
        iteration. Whether the backbone trains is controlled by
        ``requires_grad`` (set in ``_apply_freezing``), not by ``.train()``
        / ``.eval()``. Only the NAM2a / noise-adapter modules are toggled by
        this method, since their forward path doesn't have inference-only
        branches.
        """
        if self._nam_ is not None:
            self._nam_.train(training and "nam" not in self._frozen_)
        if self.use_noise_adapter and self._noise_col_ is not None:
            self._noise_col_.train(training and "noise" not in self._frozen_)
            self._noise_icl_.train(training and "noise" not in self._frozen_)
        # Backbone: keep in train mode unconditionally. The diff classifier
        # set model_.train() on its first-call branch; we don't override.

    def _snapshot_state(self) -> dict:
        snap = {}
        if self._nam_ is not None:
            snap["nam"] = {k: v.detach().clone() for k, v in self._nam_.state_dict().items()}
        if self.use_noise_adapter and self._noise_col_ is not None:
            snap["noise_col"] = {k: v.detach().clone() for k, v in self._noise_col_.state_dict().items()}
            snap["noise_icl"] = {k: v.detach().clone() for k, v in self._noise_icl_.state_dict().items()}
        # Backbone (only if any backbone param is being trained)
        m = self._diff_clf_.model_
        for sn in ("col_embedder", "row_interactor", "icl_predictor"):
            mod = getattr(m, sn, None)
            if mod is None or all(not p.requires_grad for p in mod.parameters()):
                continue
            snap[sn] = {k: v.detach().clone() for k, v in mod.state_dict().items()}
        return snap

    def _restore_state(self, snap: dict):
        if "nam" in snap:
            self._nam_.load_state_dict(snap["nam"])
        if "noise_col" in snap:
            self._noise_col_.load_state_dict(snap["noise_col"])
            self._noise_icl_.load_state_dict(snap["noise_icl"])
        m = self._diff_clf_.model_
        for sn in ("col_embedder", "row_interactor", "icl_predictor"):
            if sn in snap:
                getattr(m, sn).load_state_dict(snap[sn])

    def _run_identity_guard(self, tol: float):
        """Hold a pristine raw TabICL refit on full train, used at predict time
        if the adapter produces NaN or otherwise fails. (Cheaper than the
        full Retouche identity guard — no held-out val comparison.)
        """
        raw_clf = TabICLClassifier(
            device=str(self.device_),
            random_state=self.random_state,
            verbose=False,
        )
        raw_clf.fit(self._train_X_np_, self._train_y_enc_)
        self._raw_clf_ = raw_clf
        # We don't auto-decide use_adapter here — predict_proba uses adapter
        # by default, falls back to raw only on NaN. The user can manually
        # set self._use_adapter_ = False to force the fallback.
        self._use_adapter_ = True
