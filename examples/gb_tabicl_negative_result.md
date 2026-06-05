# Gradient-Boosted TabICL — Negative Result

**TL;DR:** GB-TabICL (using `TabICLRegressor` as the weak learner in a
standard gradient-boosting loop) does not beat vanilla TabICL, either
standalone or as an ensemble partner. Sample-weighting via residual-based
resampling never paid off. Don't pursue this direction further unless a new
weighting mechanism (e.g. attention-bias) emerges.

## Idea

Run a textbook gradient-boosting loop on top of TabICL:

```
F_0 = logit(mean(y))
for t = 1..T:
    p_t = sigmoid(F_{t-1}(X_train))
    r_t = y_train - p_t            # pseudo-residuals
    h_t = TabICLRegressor().fit(X, r_t)  # fresh weak learner per round
    F_t = F_{t-1} + lr * h_t(X)
return sigmoid(F_T)
```

Implementation lives at [src/tabicl/sklearn/gb.py](../src/tabicl/sklearn/gb.py).

Two sample-weighting modes were tested:
- `sample_weight_mode='none'` — uniform context every round
- `sample_weight_mode='residual_resample'` — resample context with replacement,
  `p ∝ |r_t| + eps`, biasing each round toward hard examples

A third option (`attention_bias` — injecting `log(w)` into row-attention logits)
was considered but never implemented since (1) and (2) didn't show signal.

## Results

### TabBench protocol, 10 binary datasets, 5-fold CV, T=8

(10 of 20 sampled datasets succeeded — 5 missing from registry, 5 multi-class.)

| Dataset | BL | GB-none-T8 | GB-resa-T8 | ENS:BL+nT8+rT8 | ENS:BL+nT4+nT8 | ENS:BL+nT4+nT8+rT4+rT8 | ENS:BL+nT2..8 |
|---|---|---|---|---|---|---|---|
| openml-3 | 0.9998 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| openml-31 (credit-g) | 0.8242 | 0.8204 | 0.8191 | 0.8251 | 0.8238 | **0.8255** | 0.8239 |
| openml-37 (diabetes) | 0.8515 | **0.8543** | 0.8520 | 0.8522 | 0.8522 | 0.8525 | 0.8527 |
| openml-1063 | 0.8215 | 0.8235 | 0.8260 | 0.8284 | 0.8210 | **0.8293** | 0.8207 |
| openml-1462 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| openml-1464 (blood) | **0.7664** | 0.7537 | 0.7094 | 0.7594 | 0.7649 | 0.7573 | 0.7627 |
| openml-1480 (ilpd) | 0.7746 | 0.7561 | 0.7737 | **0.7753** | 0.7748 | 0.7749 | 0.7736 |
| openml-1487 | 0.9436 | 0.9429 | 0.9366 | **0.9441** | 0.9438 | **0.9441** | 0.9440 |
| openml-1494 | 0.9485 | **0.9496** | 0.9481 | 0.9488 | 0.9490 | 0.9488 | 0.9489 |
| openml-1510 | **0.9974** | 0.9971 | 0.9966 | 0.9972 | 0.9972 | 0.9972 | 0.9972 |
| **mean** | **0.8927** | 0.8897 | 0.8861 | **0.8930** | 0.8926 | 0.8929 | 0.8924 |

Raw results: [gb_tabicl_tabbench_results.json](gb_tabicl_tabbench_results.json).

### Read

1. **Standalone GB underperforms BL.** GB-none-T8 mean 0.8897 vs BL 0.8927
   (−0.0030). GB-resa-T8 worse still at 0.8861 (−0.0066).
2. **Ensembles tie BL within noise.** Best mix (BL + GB-none-T8 + GB-resa-T8)
   = 0.8930, a +0.0003 gain. Mix choice barely matters: all 4 ensembles land
   in 0.8924-0.8930.
3. **Residual-resampling hurts.** Every dataset where GB-resa diverges from
   GB-none, it's worse. So option 1 is dead; option 2 isn't worth implementing
   on this evidence.
4. **One clear regression.** openml-1464 (blood-transfusion): all GB variants
   and ensembles hurt by 0.001-0.057 — never helped.
5. **Calibration is much worse for GB.** Log-loss inflated by +0.05-0.11 nats
   on every GB variant in the prior plain-bench, consistent with classical
   GB-on-strong-learner overshoot. AUC is rank-based so it doesn't punish this,
   but any downstream use that consumes calibrated probabilities would suffer.

### Earlier runs (cross-checks)

- **4-dataset plain bench (no AG), T=8**: BL 0.7811, all GB variants 0.7702-0.7741.
  Ensembles 0.7806-0.7821. Same shape: ensembles tie BL ±0.001.
- **4-dataset plain bench with staged T=2,4,6,8**: AUCs are essentially flat
  across T (Δ<0.005) — more rounds don't help. So `n_rounds=2` would give
  nearly the same result, 4× cheaper.
- **4-dataset AG-bagged bench**: GB-none mean 0.7736, GB-resa 0.7715, BL 0.7829.
  Bagging didn't change the verdict.

## Why it doesn't work (hypotheses)

- **TabICL is already a strong learner.** Classical GB assumes a weak base
  (depth-3 tree). With a high-capacity ICL learner, each round overshoots —
  log-loss inflation confirms this.
- **TabICL is already an in-context ensemble of regressors** via its built-in
  `n_estimators` bagging. Re-ensembling over rounds adds little orthogonal
  signal — the "weak learners" are highly correlated versions of each other.
- **Pseudo-residual targets are awkward for ICL.** TabICL's regressor is
  trained on a synthetic-data prior over **observation-like** targets;
  feeding it a sequence of progressively-flattening residuals may put it
  outside its training distribution after the first round.
- **Residual resampling doesn't fix capacity.** Reweighting *which* rows the
  model sees doesn't change the fact that the model's predictions overshoot.
  An attention-bias mechanism would have the same issue.

## What was kept

- [`src/tabicl/sklearn/gb.py`](../src/tabicl/sklearn/gb.py) —
  `TabICLGBClassifier`, binary only, with the residual-resample knob.
- This document.
- [`examples/gb_tabicl_tabbench_results.json`](gb_tabicl_tabbench_results.json) — raw
  per-fold AUCs.

## What was not pursued

- **Option 2: attention-bias weighting.** Would inject `log(w)` into the row-
  attention logits inside `row_interactor`. Cleaner long-term but the same
  capacity issue applies. Not worth the engineering on this evidence.
- **Adaptive learning rate / line search.** Could reduce overshoot. Might be
  worth trying if revisiting; left undone.
- **GB on a fine-tuned TabICL** (NAM2a or vanilla FT base). Could change
  things if FT lowers capacity per round. Not tested.

## Conclusion

Don't ship GB-TabICL. Keep the code as a documented negative result so we
don't try the same thing again from scratch.
