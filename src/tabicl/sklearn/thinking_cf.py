"""Counterfactual-based 'thinking mode' for TabICL.

Pipeline per sample x:
1. Predict class c with frozen TabICL.
2. Optimize x toward target (1-c) via differentiable-input gradient descent.
   Record n_steps_to_flip = first step at which the argmax flipped. Snapshot
   the flipped point x_cf.
3. Re-optimize a separate copy of x toward target c. Run for exactly n_steps_to_flip
   steps. Snapshot x_proto (a "canonical" / amplified version of x in its class).
4. Build a pair meta-classifier from (x, x_proto, x_cf) triplets on the train
   set. Each pair is (concat(x_a, x_b), label="same class"). For each test
   sample i, query the pair classifier on (x_i, x_proto_i) -- expect SAME --
   and (x_i, x_cf_i) -- expect DIFF.
5. Coherence flag: if both queries contradict expectation, the original
   class prediction is suspect (likely misclassification).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .classifier import TabICLClassifier


class ThinkingCFTabICLClassifier(BaseEstimator, ClassifierMixin):
    """Counterfactual 'thinking-mode' wrapper around TabICLClassifier.

    The wrapper exposes a standard sklearn API; the underlying class predictions
    are the same as a base ``TabICLClassifier``. The novel output is the
    per-sample ``coherence_flag_`` array (after ``predict``/``predict_proba``):
    True ⇒ the pair-classifier endorsed the original prediction, False ⇒ the
    pair-classifier disagreed (suspect misclassification).

    Parameters
    ----------
    cf_lr : float, default=5e-2
        Learning rate for the input-space gradient descent that builds the
        counterfactual and the prototype.

    cf_max_steps : int, default=50
        Hard cap on the number of optimization steps. Samples that don't flip
        by this point get ``n_steps_to_flip[i] = cf_max_steps`` and a
        ``not_converged[i] = True`` flag.

    cf_target_proba : float, default=0.5
        Probability of the target class at which we consider the flip "done"
        (in addition to the argmax check). Default 0.5 = standard decision
        threshold.

    pair_context_size : 'auto' or int, default='auto'
        Number of pairs to put in the pair-classifier's context. ``'auto'``
        uses ``n_train`` (matches one pair per train sample → does not blow
        up the pair classifier's context).

    pair_classifier_n_estimators : int, default=8
        Ensemble size for the pair classifier's inner TabICLClassifier.

    device : str or None, default=None
        Forwarded to the inner classifiers. ``None`` auto-selects CUDA.

    random_state : int or None, default=42
        Seed for pair sampling.

    verbose : bool, default=False
        Print progress for the CF / prototype optimization.
    """

    def __init__(
        self,
        cf_lr: float = 5e-2,
        cf_max_steps: int = 50,
        cf_target_proba: float = 0.5,
        pair_context_size: object = "auto",
        pair_classifier_n_estimators: int = 8,
        device: Optional[str] = None,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.cf_lr = cf_lr
        self.cf_max_steps = cf_max_steps
        self.cf_target_proba = cf_target_proba
        self.pair_context_size = pair_context_size
        self.pair_classifier_n_estimators = pair_classifier_n_estimators
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ─── public API ───────────────────────────────────────────────────────

    def fit(self, X, y):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y)
        self.label_encoder_ = LabelEncoder().fit(y_np)
        y_enc = self.label_encoder_.transform(y_np)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                f"ThinkingCFTabICLClassifier is binary-only; got {len(self.classes_)} classes."
            )

        # Standardize for stable input-space gradients
        self.imputer_ = SimpleImputer(strategy="median").fit(X_np)
        X_imp = self.imputer_.transform(X_np)
        self.scaler_ = StandardScaler().fit(X_imp)
        X_scaled = self.scaler_.transform(X_imp).astype(np.float32)

        self._X_train_np_ = X_scaled
        self._y_train_np_ = y_enc.astype(np.int64)
        self._d_ = X_scaled.shape[1]
        self._n_train_ = X_scaled.shape[0]

        device = _resolve_device(self.device)
        self.device_ = device

        # Build the base differentiable-input classifier
        # (uses fit_with_differentiable_input / predict_differentiable on the
        # current branch's TabICLClassifier).
        self.base_clf_ = TabICLClassifier(
            device=str(device),
            differentiable_input=True,
        )
        Xt = torch.from_numpy(X_scaled).to(device)
        yt = torch.from_numpy(self._y_train_np_).to(device)
        self.base_clf_.fit_with_differentiable_input(Xt, yt)

        # Stage 1+2+3 on TRAIN: predicted classes, counterfactuals, prototypes
        self._train_pred_, self._train_cf_, self._train_proto_, \
            self._train_n_flip_, self._train_not_converged_ = self._build_cf_and_proto(X_scaled)

        return self

    def predict_proba(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(self.imputer_.transform(X_np)).astype(np.float32)

        # Stage 1+2+3 on TEST: predicted classes, counterfactuals, prototypes
        test_pred, test_cf, test_proto, test_n_flip, test_nc = self._build_cf_and_proto(X_scaled)

        # Stage 4: build pair classifier on train, query on test
        coherence_flag, pair_probs = self._run_pair_classifier(
            X_scaled, test_pred, test_cf, test_proto,
        )

        # Record per-test diagnostics
        self.test_pred_ = test_pred
        self.test_n_flip_ = test_n_flip
        self.test_not_converged_ = test_nc
        self.coherence_flag_ = coherence_flag
        self.pair_probs_ = pair_probs  # (n_test, 2) — [same_pair_prob, diff_pair_prob]
        self.pair_case_ = self._pair_case_  # (n_test,) of 'A'/'B'/'C'/'D'

        # Convert hard test_pred to a soft proba consistent with the original
        # base-classifier confidence on the unmodified inputs.
        Xt = torch.from_numpy(X_scaled).to(self.device_)
        with torch.no_grad():
            base_probs = self.base_clf_.predict_differentiable(
                Xt, return_logits=False, softmax_temperature=0.9
            )
        return base_probs.detach().cpu().numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(idx)

    # ─── core machinery ───────────────────────────────────────────────────

    def _build_cf_and_proto(self, X_np: np.ndarray):
        """For each row x_i:
          - get c_i (base prediction)
          - find x_cf_i + n_steps_to_flip_i by gradient descent toward (1-c_i)
          - find x_proto_i by gradient descent toward c_i, for n_steps_to_flip_i steps

        Returns (pred_classes, cf, proto, n_flip, not_converged) — all numpy.
        """
        device = self.device_
        N, d = X_np.shape

        x_orig = torch.from_numpy(X_np).to(device)

        # Stage 1: base predictions
        with torch.no_grad():
            base_probs = self.base_clf_.predict_differentiable(
                x_orig, return_logits=False, softmax_temperature=0.9,
            )
        pred_classes = base_probs.argmax(dim=1)  # (N,)

        # Stage 2: counterfactual = optimize x toward target (1 - pred_class)
        target_cf = 1 - pred_classes  # (N,)
        x_cf, n_flip, not_converged = self._optimize_toward_target(
            x_orig, target_cf, stop_on_flip=True,
        )

        # Stage 3: prototype = optimize x toward target = pred_class, for n_flip[i] steps
        # We run the loop in lockstep on all samples but snapshot x_proto_i at step n_flip_i.
        target_proto = pred_classes
        x_proto = self._optimize_for_n_steps(
            x_orig, target_proto, n_flip,
        )

        return (
            pred_classes.detach().cpu().numpy(),
            x_cf.detach().cpu().numpy(),
            x_proto.detach().cpu().numpy(),
            n_flip.detach().cpu().numpy().astype(np.int64),
            not_converged.detach().cpu().numpy().astype(bool),
        )

    def _optimize_toward_target(
        self, x_init: torch.Tensor, target: torch.Tensor,
        stop_on_flip: bool,
    ):
        """Per-row gradient descent on input toward target class.

        Returns:
            x_final : (N, d) tensor — each row's snapshot at flip (or final if not flipped).
            n_flip  : (N,) int tensor — step at which each row flipped (or cf_max_steps).
            not_converged : (N,) bool tensor — True if didn't flip by cf_max_steps.
        """
        device = x_init.device
        N, d = x_init.shape
        x = x_init.detach().clone().requires_grad_(True)

        flipped = torch.zeros(N, dtype=torch.bool, device=device)
        n_flip = torch.full((N,), self.cf_max_steps, dtype=torch.long, device=device)
        x_snapshot = x_init.detach().clone()

        for step in range(1, self.cf_max_steps + 1):
            # Per-step forward
            probs = self.base_clf_.predict_differentiable(
                x, return_logits=False, softmax_temperature=0.9,
            )
            # CE loss toward target
            log_probs = torch.log(probs.clamp_min(1e-12))
            loss = F.nll_loss(log_probs, target, reduction="mean")

            grads = torch.autograd.grad(loss, x, create_graph=False)[0]
            with torch.no_grad():
                x = (x - self.cf_lr * grads).detach().requires_grad_(True)

                # Check which rows just flipped
                probs_now = self.base_clf_.predict_differentiable(
                    x, return_logits=False, softmax_temperature=0.9,
                )
                pred_now = probs_now.argmax(dim=1)
                newly_flipped = (pred_now == target) & (~flipped)
                if newly_flipped.any():
                    # Snapshot those rows
                    x_snapshot[newly_flipped] = x.detach()[newly_flipped]
                    n_flip[newly_flipped] = step
                    flipped = flipped | newly_flipped

            if stop_on_flip and flipped.all():
                break

            if self.verbose and step % 10 == 0:
                print(f"  [CF step {step}] flipped={int(flipped.sum())}/{N}", flush=True)

        not_converged = ~flipped
        # For non-converged rows, the snapshot is the original x (we didn't flip).
        return x_snapshot, n_flip, not_converged

    def _optimize_for_n_steps(
        self, x_init: torch.Tensor, target: torch.Tensor, n_steps_per_row: torch.Tensor,
    ):
        """Run optimization on all rows, snapshot row i at exactly n_steps_per_row[i].

        Lockstep: do up to max(n_steps_per_row) full passes, but each row
        captures its snapshot at its own step count.
        """
        device = x_init.device
        N, d = x_init.shape
        max_steps = int(n_steps_per_row.max().item())
        if max_steps == 0:
            return x_init.detach().clone()

        x = x_init.detach().clone().requires_grad_(True)
        x_snapshot = x_init.detach().clone()
        captured = torch.zeros(N, dtype=torch.bool, device=device)

        for step in range(1, max_steps + 1):
            probs = self.base_clf_.predict_differentiable(
                x, return_logits=False, softmax_temperature=0.9,
            )
            log_probs = torch.log(probs.clamp_min(1e-12))
            loss = F.nll_loss(log_probs, target, reduction="mean")

            grads = torch.autograd.grad(loss, x, create_graph=False)[0]
            with torch.no_grad():
                x = (x - self.cf_lr * grads).detach().requires_grad_(True)

                # Capture rows whose step count == step
                to_capture = (n_steps_per_row == step) & (~captured)
                if to_capture.any():
                    x_snapshot[to_capture] = x.detach()[to_capture]
                    captured = captured | to_capture

            if self.verbose and step % 10 == 0:
                print(f"  [Proto step {step}] captured={int(captured.sum())}/{N}", flush=True)

        # Any rows with n_steps_per_row == 0 keep their original x
        return x_snapshot

    # ─── pair classifier ──────────────────────────────────────────────────

    def _run_pair_classifier(
        self,
        X_test_np: np.ndarray,
        test_pred: np.ndarray,
        test_cf: np.ndarray,
        test_proto: np.ndarray,
    ):
        """Build pair meta-dataset from train (x, x_proto, x_cf) triplets,
        train TabICLClassifier on it, query on test pairs.

        Returns
        -------
        coherence_flag : (n_test,) bool — True ⇒ pair classifier confirms
            original prediction (same-pair → SAME, cf-pair → DIFF).
        pair_probs : (n_test, 2) float — [P(same|x,x_proto), P(diff|x,x_cf)]
            interpreted as the *expected* labels, so coherence = both > 0.5.
        """
        rng = np.random.RandomState(self.random_state or 0)

        # Resolve K = n_train pairs to build context
        if self.pair_context_size == "auto":
            K = self._n_train_
        else:
            K = int(self.pair_context_size)

        # Sample pairs for the meta-classifier context. Three sources, ~1/3 each:
        K_self = K // 3
        K_cf = K // 3
        K_cross = K - K_self - K_cf

        # (a) (x_i, x_proto_i) on train — label SAME (1)
        idx_a = rng.choice(self._n_train_, size=K_self, replace=True)
        pairs_a = np.concatenate(
            [self._X_train_np_[idx_a], self._train_proto_[idx_a]], axis=1,
        )
        labels_a = np.ones(K_self, dtype=np.int64)

        # (b) (x_i, x_cf_i) on train — label DIFF (0) — but only if i actually flipped
        ok_b = ~self._train_not_converged_
        idx_b_pool = np.where(ok_b)[0]
        if len(idx_b_pool) == 0:
            # Edge case: nothing flipped on train. Fall back to (x, x) same-pairs.
            idx_b = rng.choice(self._n_train_, size=K_cf, replace=True)
            pairs_b = np.concatenate(
                [self._X_train_np_[idx_b], self._X_train_np_[idx_b]], axis=1,
            )
            labels_b = np.ones(K_cf, dtype=np.int64)
        else:
            idx_b = rng.choice(idx_b_pool, size=K_cf, replace=True)
            pairs_b = np.concatenate(
                [self._X_train_np_[idx_b], self._train_cf_[idx_b]], axis=1,
            )
            labels_b = np.zeros(K_cf, dtype=np.int64)

        # (c) (x_proto_i, x_proto_j) cross — label = 1[c_i == c_j]
        idx_c1 = rng.choice(self._n_train_, size=K_cross, replace=True)
        idx_c2 = rng.choice(self._n_train_, size=K_cross, replace=True)
        pairs_c = np.concatenate(
            [self._train_proto_[idx_c1], self._train_proto_[idx_c2]], axis=1,
        )
        labels_c = (self._train_pred_[idx_c1] == self._train_pred_[idx_c2]).astype(np.int64)

        # Random order per pair: swap the two halves with 50% prob
        all_pairs = np.concatenate([pairs_a, pairs_b, pairs_c], axis=0)
        all_labels = np.concatenate([labels_a, labels_b, labels_c], axis=0)
        d = self._d_
        swap = rng.rand(len(all_pairs)) < 0.5
        if swap.any():
            tmp = all_pairs[swap, :d].copy()
            all_pairs[swap, :d] = all_pairs[swap, d:]
            all_pairs[swap, d:] = tmp

        # Shuffle for good measure
        perm = rng.permutation(len(all_pairs))
        all_pairs = all_pairs[perm]
        all_labels = all_labels[perm]

        # Build the test queries
        n_test = X_test_np.shape[0]
        query_same = np.concatenate([X_test_np, test_proto], axis=1)
        query_diff = np.concatenate([X_test_np, test_cf], axis=1)
        # Random swap for queries too
        sw_s = rng.rand(n_test) < 0.5
        if sw_s.any():
            tmp = query_same[sw_s, :d].copy()
            query_same[sw_s, :d] = query_same[sw_s, d:]
            query_same[sw_s, d:] = tmp
        sw_d = rng.rand(n_test) < 0.5
        if sw_d.any():
            tmp = query_diff[sw_d, :d].copy()
            query_diff[sw_d, :d] = query_diff[sw_d, d:]
            query_diff[sw_d, d:] = tmp

        # Train a fresh TabICLClassifier on the pair meta-task
        pair_clf = TabICLClassifier(
            device=str(self.device_),
            n_estimators=self.pair_classifier_n_estimators,
            random_state=self.random_state or 0,
            verbose=False,
        )
        pair_clf.fit(all_pairs.astype(np.float32), all_labels)

        p_same = pair_clf.predict_proba(query_same.astype(np.float32))[:, 1]
        p_diff_is_same = pair_clf.predict_proba(query_diff.astype(np.float32))[:, 1]
        p_diff = 1.0 - p_diff_is_same

        # Coherence: same-pair endorsed (p_same > 0.5) AND diff-pair endorsed
        S = (p_same > 0.5)
        D = (p_diff > 0.5)
        coherence_flag = S & D
        # Four-way case labels:
        #   A: coherent   (S=T, D=T)
        #   B: cf-doubt   (S=T, D=F) — CF judged same-class as x
        #   C: proto-doubt(S=F, D=T) — proto judged different-class from x
        #   D: total      (S=F, D=F) — both inverted
        case = np.full(len(p_same), "A", dtype="<U1")
        case[S & ~D] = "B"
        case[~S & D] = "C"
        case[~S & ~D] = "D"
        pair_probs = np.stack([p_same, p_diff], axis=1)
        self._pair_case_ = case  # exposed for analysis
        return coherence_flag, pair_probs


# ─── helpers ──────────────────────────────────────────────────────────────


def _resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device
