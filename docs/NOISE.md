# Noise-Aware Adapters for TabICL

Lightweight adapters that make a frozen pretrained TabICL noise-aware for financial data, without retraining.

## Problem

Financial data is noisy (heavy tails, regime changes, non-stationary features). The pretrained model was trained on synthetic priors that don't model this. We want to inject noise awareness **without modifying or retraining the base model**.

## Approach

Two complementary adapters inserted into the frozen TabICL pipeline:

```
X ──→ [col_embedder] ──→ [NoiseEmbeddingAdapter] ──→ [row_interactor] ──→ [NoiseCrossAttentionAdapter] ──→ [icl_predictor] ──→ predictions
          frozen              TRAINED                    frozen                    TRAINED                      frozen
```

### 1. NoiseEmbeddingAdapter (after column embedding)

**What it does:** For each feature, takes noise metadata (e.g., rolling volatility, confidence interval width) and produces an additive embedding shift. The downstream model sees "value X with noise level Y" — not just "value X".

**Key design choices:**
- **Additive, not multiplicative** — doesn't suppress noisy features, enriches them
- **Gated** — a learned sigmoid gate controls how much noise info to blend in
- **Identity at init** — gate output and noise encoder are zero-initialized, so the adapter starts as a no-op and gradually learns

**Parameters:** ~17K (tiny)

### 2. NoiseCrossAttentionAdapter (after row interaction)

**What it does:** Cross-attention where queries are the row representations (what ICL will see) and keys/values are noise-conditioned versions. The model learns what to focus on given the noise structure of each sample.

**Key design choices:**
- **Recontextualizes, doesn't filter** — a noisy-but-informative feature gets attended to differently, not suppressed
- **Output gate** — starts at 0 (identity), learns to blend in the noise-aware signal
- **Pre-norm** — LayerNorm on Q and KV paths for stable training

**Parameters:** ~1.3M (~5% of the 28M base model)

## Usage

### Standalone adapters

```python
from noise_adapter import NoiseEmbeddingAdapter, NoiseCrossAttentionAdapter

# After column embedding
col_adapter = NoiseEmbeddingAdapter(
    embed_dim=128,      # must match model's embed_dim
    noise_dim=3,        # your noise features: e.g., volatility, confidence, quality
    num_cls=4,          # model's row_num_cls
)
# embeddings: (B, T, G+C, E) from col_embedder
# noise_meta: (B, T, G, 3) per-feature noise
enriched = col_adapter(embeddings, noise_meta)

# After row interaction
icl_adapter = NoiseCrossAttentionAdapter(
    d_model=512,        # embed_dim * num_cls = 128 * 4
    noise_dim=3,        # per-row noise summary dimension
)
# representations: (B, T, 512) from row_interactor
# noise_summary: (B, T, 3) per-row aggregated noise
recontextualized = icl_adapter(representations, noise_summary)
```

### With TabICL's fine-tuning framework

```python
from noise_adapter import NoiseFinetuneClassifier

clf = NoiseFinetuneClassifier(
    noise_dim=3,                # per-feature noise metadata dimension
    noise_summary_dim=3,        # per-row noise summary dimension
    adapter_hidden=64,          # noise encoder hidden dim
    adapter_nhead=4,            # cross-attention heads
    epochs=20,
    learning_rate=1e-4,
    device="cuda",
    verbose=True,
    # Base model is frozen by default (freeze_col=True, freeze_row=True, freeze_icl=True)
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

## Noise metadata examples for finance

**Per-feature noise (`noise_meta`):**
- Rolling volatility of each feature (e.g., 20-day std)
- Bid-ask spread (for price features)
- Data staleness (seconds since last update)
- Source reliability score

**Per-row noise summary (`noise_summary`):**
- Mean/max feature volatility across all features for that row
- Market regime indicator (VIX level, spread index)
- Data completeness ratio (fraction of non-stale features)

## Architecture details

```
NoiseEmbeddingAdapter (17K params):
  noise_encoder: Linear(noise_dim, 64) → GELU → Linear(64, embed_dim)
  gate:          Linear(noise_dim, 64) → GELU → Linear(64, embed_dim)
  forward:       feat + sigmoid(gate(noise)) * noise_encoder(noise)

NoiseCrossAttentionAdapter (1.3M params):
  noise_proj: Linear(noise_dim, d_model) → GELU → Linear(d_model, d_model)
  Q/K/V proj: Linear(d_model, d_model) each
  out_proj:   Linear(d_model, d_model)  [zero-init]
  output_gate: scalar param [init=0]
  forward:    repr + sigmoid(gate) * CrossAttn(Q=repr, KV=repr+noise_proj(noise))
```

Both adapters are initialized as identity functions and gradually learn to incorporate noise information during fine-tuning.

## Files

- `noise_adapter.py` — All modules + `NoiseFinetuneClassifier`
- `NOISE.md` — This file

## TODO

- Wire noise tensors through TabICL's `MetaBatch` data pipeline (currently noise is passed as None during fine-tuning — the adapter learns from data distribution alone)
- Add `NoiseFinetuneRegressor` variant
- Experiment with propagating uncertainty (mean, variance) pairs through layers instead of point estimates
