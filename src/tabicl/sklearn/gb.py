"""Gradient-boosted TabICL classifier.

Uses TabICLRegressor as the weak learner in a standard gradient-boosting loop
on log-odds. Each round fits to the prior round's pseudo-residuals and adds
its prediction (scaled by the learning rate) to the running log-odds estimate.

No gradients, no fine-tuning. Just ICL composition.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from .regressor import TabICLRegressor


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _log_loss_from_logodds(F: np.ndarray, y: np.ndarray) -> float:
    """Stable binary cross-entropy from raw log-odds."""
    # max(F, 0) - F * y + log(1 + exp(-|F|))
    return float(np.mean(np.maximum(F, 0) - F * y + np.log1p(np.exp(-np.abs(F)))))


class TabICLGBClassifier(ClassifierMixin, BaseEstimator):
    """Gradient-boosted TabICL for binary classification.

    Each round fits a fresh ``TabICLRegressor`` on the current pseudo-
    residuals (gradients of the binary log-loss in log-odds space) and
    accumulates its scaled prediction into the running estimate. Final
    prediction is ``sigmoid(F_T)``.

    Algorithm
    ---------
    F_0(x) = log(p / (1-p)),  p = mean(y_train)
    for t = 1..n_rounds:
        p_t       = sigmoid(F_{t-1}(x))                # current proba
        r_t,i     = y_i - p_t(x_i)                     # pseudo-residual on train
        h_t       = TabICLRegressor().fit(X_tr, r_t)    # fresh weak learner
        F_t(x)    = F_{t-1}(x) + learning_rate * h_t(x)

    Parameters
    ----------
    n_rounds : int, default=20
        Maximum number of boosting rounds.
    learning_rate : float, default=0.1
        Step size applied to each weak learner's prediction.
    sample_weight_mode : 'none' | 'residual_resample', default='none'
        How the weak learner sees per-sample importance:

        * ``'none'``: every context row is treated equally — vanilla GB-ICL.
        * ``'residual_resample'``: each round, resample the train rows with
          replacement, ``p ∝ |residual| + sample_weight_eps``. High-residual
          samples appear more often in the context; the regressor's in-context
          averaging gets biased toward them. Approximation of true sample
          weighting without touching TabICL's attention.
    sample_weight_eps : float, default=1e-3
        Floor added to |residual| before normalizing into sampling
        probabilities (avoid degenerate 0-probability rows).
    early_stopping_rounds : int or None, default=5
        If ``X_val``/``y_val`` are passed to ``fit``, stop after this many
        rounds without improvement on the val log-loss. ``None`` disables.
    n_estimators : int, default=8
        Inner ``TabICLRegressor`` n_estimators (ensemble size of each round).
        Smaller = faster, less accurate per round; larger = slower, more accurate.
    batch_size : int, default=8
        Inner ``TabICLRegressor`` batch size.
    device : str or None, default=None
        Forwarded to each round's ``TabICLRegressor``. ``None`` = auto-detect.
    random_state : int, default=42
        Seed for reproducibility of per-round ``TabICLRegressor`` ensembles.
    verbose : bool, default=False
        Print per-round train/val log-loss.

    Notes
    -----
    Binary classification only — checks for two classes during fit.
    """

    def __init__(
        self,
        n_rounds: int = 20,
        learning_rate: float = 0.1,
        sample_weight_mode: str = "none",
        sample_weight_eps: float = 1e-3,
        early_stopping_rounds: Optional[int] = 5,
        n_estimators: int = 8,
        batch_size: int = 8,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.sample_weight_mode = sample_weight_mode
        self.sample_weight_eps = sample_weight_eps
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ─── public API ─────────────────────────────────────────────────────

    def fit(self, X, y, X_val=None, y_val=None):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y)
        self.label_encoder_ = LabelEncoder().fit(y_np)
        y_enc = self.label_encoder_.transform(y_np)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                f"TabICLGBClassifier is binary-only; got {len(self.classes_)} classes."
            )

        # Cache for prediction-time refit-on-full-train
        self._X_train_ = X_np
        self._y_train_enc_ = y_enc.astype(np.float64)

        # Base log-odds (mean(y) in [eps, 1-eps])
        p0 = float(self._y_train_enc_.mean())
        self.f0_ = _logit(p0)

        # Initialize running log-odds on train and (optional) val
        F_tr = np.full(len(y_enc), self.f0_, dtype=np.float64)
        if X_val is not None:
            X_val_np = np.asarray(X_val, dtype=np.float32)
            y_val_enc = self.label_encoder_.transform(np.asarray(y_val))
            F_va = np.full(len(y_val_enc), self.f0_, dtype=np.float64)
        else:
            X_val_np = None
            y_val_enc = None
            F_va = None

        # The fit-time predictions on X_train serve only as pseudo-residual
        # targets; we don't store them. We do store the per-round trained
        # weak learners so predict() can reapply them on test data.
        if self.sample_weight_mode not in ("none", "residual_resample"):
            raise ValueError(
                f"sample_weight_mode must be 'none' or 'residual_resample', "
                f"got {self.sample_weight_mode!r}"
            )
        rng_master = np.random.RandomState(self.random_state)

        self.weak_learners_ = []
        best_val = float("inf")
        best_round = -1
        bad_rounds = 0

        for t in range(self.n_rounds):
            p_tr = _sigmoid(F_tr)
            r_t = self._y_train_enc_ - p_tr   # pseudo-residual = y - p

            # Optionally resample (X, r_t) with p ∝ |r_t| + eps for this round.
            if self.sample_weight_mode == "residual_resample":
                w = np.abs(r_t) + self.sample_weight_eps
                p = w / w.sum()
                idx = rng_master.choice(len(r_t), size=len(r_t), replace=True, p=p)
                X_round = X_np[idx]
                r_round = r_t[idx]
            else:
                X_round = X_np
                r_round = r_t

            # Fresh weak learner on (X_round, r_round)
            reg = TabICLRegressor(
                device=self.device,
                n_estimators=self.n_estimators,
                batch_size=self.batch_size,
                random_state=self.random_state + t,
                verbose=False,
            )
            reg.fit(X_round, r_round)

            # Predict on train (for next round's residuals) and val (for ES)
            h_tr = reg.predict(X_np).astype(np.float64)
            F_tr = F_tr + self.learning_rate * h_tr

            if F_va is not None:
                h_va = reg.predict(X_val_np).astype(np.float64)
                F_va = F_va + self.learning_rate * h_va
                val_ll = _log_loss_from_logodds(F_va, y_val_enc.astype(np.float64))
                tr_ll = _log_loss_from_logodds(F_tr, self._y_train_enc_)
                if self.verbose:
                    print(f"[gb] round {t+1:>3d}: train_logloss={tr_ll:.4f}  val_logloss={val_ll:.4f}",
                          flush=True)

                if self.early_stopping_rounds is not None:
                    if val_ll + 1e-7 < best_val:
                        best_val = val_ll
                        best_round = t
                        bad_rounds = 0
                    else:
                        bad_rounds += 1
                        if bad_rounds >= self.early_stopping_rounds:
                            if self.verbose:
                                print(f"[gb] early stop @ round {t+1}; best @ round {best_round+1} val_ll={best_val:.4f}",
                                      flush=True)
                            # Truncate to best round (inclusive)
                            self.weak_learners_.append(reg)
                            self.weak_learners_ = self.weak_learners_[: best_round + 1]
                            break
            else:
                tr_ll = _log_loss_from_logodds(F_tr, self._y_train_enc_)
                if self.verbose:
                    print(f"[gb] round {t+1:>3d}: train_logloss={tr_ll:.4f}", flush=True)

            self.weak_learners_.append(reg)

        # Stash final round counts for inspection
        self.n_rounds_used_ = len(self.weak_learners_)
        self._is_fitted_ = True
        return self

    def predict_proba(self, X):
        if not getattr(self, "_is_fitted_", False):
            raise RuntimeError("Call fit before predict_proba.")
        X_np = np.asarray(X, dtype=np.float32)

        F = np.full(X_np.shape[0], self.f0_, dtype=np.float64)
        for reg in self.weak_learners_:
            h = reg.predict(X_np).astype(np.float64)
            F = F + self.learning_rate * h

        p1 = _sigmoid(F)
        return np.stack([1.0 - p1, p1], axis=1)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(idx)

    def staged_predict_proba(self, X):
        """Yield predict_proba at each boosting round (useful for ES diagnostics).

        Yields ``n_rounds_used_`` arrays of shape (n_samples, 2), starting from
        round 1 (after the first weak learner).
        """
        if not getattr(self, "_is_fitted_", False):
            raise RuntimeError("Call fit before staged_predict_proba.")
        X_np = np.asarray(X, dtype=np.float32)
        F = np.full(X_np.shape[0], self.f0_, dtype=np.float64)
        for reg in self.weak_learners_:
            h = reg.predict(X_np).astype(np.float64)
            F = F + self.learning_rate * h
            p1 = _sigmoid(F)
            yield np.stack([1.0 - p1, p1], axis=1)
