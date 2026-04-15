"""Evaluate Legendre-compressed TabICL on real OpenML datasets.

Loads the pretrained classifier, converts the ICL encoder to Legendre
parameterization at various compression levels, and compares accuracy
on multiple datasets from OpenML.

NOTE: clf.fit() reloads the model from checkpoint, so the encoder must
be swapped AFTER fit() and predict() called WITHOUT re-fitting.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from tabicl import TabICLClassifier
from tabicl.model.legendre import LegendreEncoder


DATASETS = [
    ("iris", 61),
    ("wine", 187),
    ("balance-scale", 11),
    ("vehicle", 54),
    ("segment", 36),
    ("steel-plates-fault", 1504),
    ("vowel", 307),
    ("letter", 6),
    ("pendigits", 32),
    ("optdigits", 28),
]


def load_dataset(name, openml_id):
    """Fetch an OpenML dataset and return train/test splits."""
    data = fetch_openml(data_id=openml_id, as_frame=False, parser="auto")
    X, y = data.data, data.target
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Cap dataset size for speed
    if len(X) > 1000:
        idx = np.random.RandomState(42).choice(len(X), 1000, replace=False)
        X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    n_classes = len(np.unique(y))
    return X_train, X_test, y_train, y_test, n_classes


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    np.random.seed(42)

    print("Loading pretrained TabICL classifier...")
    clf = TabICLClassifier(n_estimators=2)

    # Compression configs to test
    configs = [
        ("baseline", None),
        ("6 coeffs (2x)", 6),
        ("4 coeffs (3x)", 4),
        ("3 coeffs (4x)", 3),
    ]

    # Print header
    header = f"{'Dataset':<22} {'Classes':>7}"
    for label, _ in configs:
        header += f" {label:>16}"
    print(header)
    print("-" * len(header))

    # Aggregate results
    all_results = {label: [] for label, _ in configs}

    for ds_name, openml_id in DATASETS:
        X_train, X_test, y_train, y_test, n_classes = load_dataset(ds_name, openml_id)
        row = f"{ds_name:<22} {n_classes:>7}"

        for label, nc in configs:
            # fit() reloads model from checkpoint — always start fresh
            clf.fit(X_train, y_train)

            if nc is not None:
                # Swap encoder AFTER fit, then predict without re-fitting
                orig_enc = clf.model_.icl_predictor.tf_icl
                device = next(orig_enc.parameters()).device
                leg_enc = LegendreEncoder.from_encoder(orig_enc, num_coeffs=nc)
                leg_enc.eval().to(device)
                clf.model_.icl_predictor.tf_icl = leg_enc

            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            all_results[label].append(acc)
            row += f" {acc:>15.1%}"

        print(row)

    # Print summary
    print("-" * len(header))
    summary = f"{'MEAN':<22} {'':>7}"
    for label, _ in configs:
        mean_acc = np.mean(all_results[label])
        summary += f" {mean_acc:>15.1%}"
    print(summary)

    # Parameter counts (need a model to count from)
    X_tr, X_te, y_tr, y_te, _ = load_dataset(*DATASETS[0])
    clf.fit(X_tr, y_tr)
    orig_enc = clf.model_.icl_predictor.tf_icl
    orig_params = count_params(orig_enc)

    print(f"\nCompression summary:")
    print(f"  {'Config':<20} {'Params':>12} {'Ratio':>8}")
    print(f"  {'-'*42}")
    for label, nc in configs:
        if nc is None:
            print(f"  {label:<20} {orig_params:>12,} {'1.00x':>8}")
        else:
            leg_enc = LegendreEncoder.from_encoder(orig_enc, num_coeffs=nc)
            p = count_params(leg_enc)
            print(f"  {label:<20} {p:>12,} {orig_params / p:>.2f}x")


if __name__ == "__main__":
    main()
