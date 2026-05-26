# Thinking-CF: negative result

## What was tried

Counterfactual-based "thinking mode" for TabICL. Per-sample pipeline:

1. Predict class `c` with frozen TabICL.
2. Gradient-descent on the input toward target `1−c`, snapshot `x_cf` at the
   first step where argmax flipped, record `n_steps_to_flip`.
3. Gradient-descent on a fresh copy of the input toward target `c` for exactly
   `n_steps_to_flip` steps, snapshot `x_proto` (the "amplified" / canonical
   version of the sample in its predicted class).
4. Build a *pair meta-classifier* from train `(x, x_proto, x_cf)` triplets:
   training pairs are `concat(a, b)` with binary label "same class". The pair
   classifier itself is a `TabICLClassifier` fit on the meta-dataset.
5. For each test sample, query the pair classifier on `(x, x_proto)` (expect
   "same") and `(x, x_cf)` (expect "different").

Four cases per test sample based on whether each query returns the expected
label:

- **A** (same=✓, diff=✓) — "coherent"
- **B** (same=✓, diff=✗) — CF judged same-class as x (CF didn't really flip)
- **C** (same=✗, diff=✓) — proto judged different-class from x
- **D** (same=✗, diff=✗) — both inverted

## What we found

Bench on 20 OpenML-CC18 binary classification datasets (17 completed, 3
failed on remote disk), val-tuned threshold for each method, applied on a
held-out outer test split.

**For misclassification detection / selective prediction**:
- Case B = "model was confident" with **+0.50 to +0.81 Pearson correlation
  with `max(predict_proba)`** across datasets.
- In other words, the pair-classifier's coherence is mostly a noisy proxy for
  TabICL's own confidence.

**For prediction swapping** (flip the top-k samples by some "doubt" score):
- Case-based score: `(1−p_S) + (1−p_D)`
- Confidence-based score: `1 − max(predict_proba)`
- Both methods sweep top-k on a val split, pick `k*` that maximises val
  accuracy after flipping, apply on test.

Aggregate test results on 17 datasets:

| Metric | Case-swap (val-tuned) | Confidence-swap (val-tuned) |
|---|---|---|
| Mean Δ vs BL | **−0.0065** | **−0.0039** |
| Wins | 4 | **9** |
| Ties | 4 | 4 |
| Losses | 9 | 4 |

Both methods hurt on average after val-tuning. The val→test transfer of
`k*` is too noisy on small datasets — overfitting the val signal doesn't
generalise. Crucially, **case-swap is worse than confidence-swap on average**
(mean Δ(case − conf) = −0.0027), and the pair-classifier signal brings nothing
useful over confidence.

The earlier "ilpd D-swap +3.4pp" win that motivated this further work was a
cherry-picked manual swap rule on a single dataset; with proper val-tuning on
the same dataset, case-swap gets −0.85pp and confidence gets +0.85pp.

## Why it failed

1. **The pair classifier learnt confidence.** Most train pairs in its context
   ended up being "(x, x)" fallbacks (because few samples actually flipped),
   so it learnt "samples that look similar are same-class" — which is exactly
   what `max(proba)` already encodes.

2. **The CF perturbations were adversarial-like, not semantic.** Gradient
   descent at lr=0.1 for ≤50 steps mostly produces small input perturbations
   that cross the decision boundary without becoming a realistic class-`(1−c)`
   example. The pair classifier saw through them and judged `x_cf` as still
   same-class.

3. **Threshold discovery is the actual hard problem**, and we discovered it
   the hard way. The pair-classifier framing didn't crack it; if anything it
   made the threshold harder to choose because there are two probabilities
   (p_S, p_D) instead of one.

## Files

- `thinking_cf.py` — the `ThinkingCFTabICLClassifier` implementation
- `examples/thinking_cf_initial_results.json` — initial 4-dataset bench
- `examples/thinking_cf_swap_val_tuned_results.json` — 17-dataset val-tuned bench

## Take-away

When you need a "should I trust this prediction?" signal, **start with**
`max(predict_proba)`. It's free, it's available, and it's hard to beat
without much more compute and a real semantic improvement on top.

Branch `feature/thinking-cf` is preserved as documentation of what doesn't
work, so future-me doesn't reinvent it.
