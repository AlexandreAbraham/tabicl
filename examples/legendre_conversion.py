"""Convert pretrained TabICL weights to Legendre-parameterized form.

Demonstrates:
1. Loading pretrained classifier
2. Converting ICL encoder to LegendreEncoder
3. Measuring parameter compression
4. Measuring reconstruction quality on a real dataset
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tabicl import TabICLClassifier
from tabicl.model.legendre import LegendreEncoder


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    # ── 1. Load pretrained model ──
    print("Loading pretrained TabICL classifier...")
    clf = TabICLClassifier(n_estimators=1)
    X, y = make_classification(n_samples=200, n_features=10, n_classes=5,
                               n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf.fit(X_train, y_train)
    baseline_preds = clf.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    model = clf.model_
    config = clf.model_config_

    # ── 2. Analyze the original ICL encoder ──
    icl_encoder = model.icl_predictor.tf_icl
    num_blocks = len(icl_encoder.blocks)
    d_model = config.get("embed_dim", 128) * config.get("row_num_cls", 4)

    print(f"\nOriginal ICL encoder:")
    print(f"  Blocks: {num_blocks}")
    print(f"  d_model (ICL dim): {d_model}")
    print(f"  Parameters: {count_params(icl_encoder):,}")

    # ── 3. Convert at different compression levels ──
    print(f"\n{'Coeffs':>8} {'Params':>12} {'Compression':>12} {'Recon Error':>12}")
    print("-" * 50)

    for num_coeffs in [num_blocks, num_blocks // 2, num_blocks // 3, 4, 3, 2]:
        if num_coeffs < 1 or num_coeffs > num_blocks:
            continue

        leg_enc = LegendreEncoder.from_encoder(icl_encoder, num_coeffs=num_coeffs)
        leg_enc.eval()

        orig_params = count_params(icl_encoder)
        leg_params = count_params(leg_enc)
        compression = orig_params / leg_params

        # Measure reconstruction error on a sample input
        with torch.no_grad():
            x = torch.randn(2, 50, d_model)
            out_orig = icl_encoder(x, train_size=35)
            out_leg = leg_enc(x, train_size=35)
            recon_err = (out_orig - out_leg).abs().mean().item()

        print(f"{num_coeffs:>8} {leg_params:>12,} {compression:>11.2f}x {recon_err:>12.6f}")

    # ── 4. Full end-to-end evaluation with compressed model ──
    # NOTE: clf.fit() reloads the model from checkpoint, so we must swap
    # the encoder AFTER fit() and call predict() WITHOUT re-fitting.
    print("\n── End-to-end accuracy with Legendre compression ──")
    print(f"{'Coeffs':>8} {'Accuracy':>10} {'Δ vs baseline':>14}")
    print("-" * 35)

    for num_coeffs in [num_blocks, num_blocks // 2, 4, 3, 2]:
        if num_coeffs < 1 or num_coeffs > num_blocks:
            continue

        # Replace the ICL encoder in-place (no re-fitting!)
        device = next(icl_encoder.parameters()).device
        leg_enc = LegendreEncoder.from_encoder(icl_encoder, num_coeffs=num_coeffs)
        leg_enc.eval().to(device)
        model.icl_predictor.tf_icl = leg_enc

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        delta = acc - baseline_acc
        sign = "+" if delta >= 0 else ""
        print(f"{num_coeffs:>8} {acc:>9.4f} {sign}{delta:>13.4f}")

    # Restore original encoder
    model.icl_predictor.tf_icl = icl_encoder

    # ── 5. Weight-space analysis ──
    print("\n── Weight smoothness analysis (explains compression quality) ──")
    print("Measuring how much layer weights vary along depth...")

    with torch.no_grad():
        # Stack in_proj weights across layers: (L, 3*d, d)
        in_proj_weights = torch.stack([b.attn.in_proj_weight for b in icl_encoder.blocks])
        L = in_proj_weights.shape[0]

        # Consecutive layer distance
        diffs = []
        for i in range(L - 1):
            diff = (in_proj_weights[i + 1] - in_proj_weights[i]).norm().item()
            diffs.append(diff)

        avg_weight_norm = in_proj_weights.flatten(1).norm(dim=1).mean().item()
        avg_diff = np.mean(diffs)
        max_diff = np.max(diffs)

        print(f"  Avg weight norm per layer:        {avg_weight_norm:.4f}")
        print(f"  Avg consecutive layer distance:   {avg_diff:.4f}")
        print(f"  Max consecutive layer distance:   {max_diff:.4f}")
        print(f"  Relative variation (avg/norm):    {avg_diff / avg_weight_norm:.4f}")

        # SVD of stacked weights to see effective rank along depth
        W_flat = in_proj_weights.flatten(1)  # (L, 3*d*d)
        U, S, V = torch.linalg.svd(W_flat, full_matrices=False)
        S_norm = S / S.sum()
        cumsum = S_norm.cumsum(0)
        print(f"\n  Singular value energy distribution along depth axis:")
        for k in range(min(L, 8)):
            print(f"    Top-{k+1} components capture {cumsum[k].item()*100:.1f}% of variance")

    print("\nConclusion:")
    print("  If weights vary sharply between layers, cold projection loses quality.")
    print("  For best results, train FROM SCRATCH with Legendre parameterization,")
    print("  or fine-tune the converted model to recover accuracy.")


if __name__ == "__main__":
    main()
