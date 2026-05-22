"""Retouche-PEB (PLR + Embed + LN + 4-way BatchEnsemble) vs raw TabICL.

The 30-dataset bench using the best improvement variant. Comparable to the
paper's TALENT win rates (97/4/69 AUC = 58.4%, 109/0/61 LogLoss = 64.1%).
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

from autogluon.core.models import AbstractModel
from autogluon.tabular import TabularPredictor

from tabicl import TabICLClassifier
from tabicl.sklearn.retouche import (
    RetoucheTabICLClassifier,
    BaggedRetoucheTabICLClassifier,
)


def _peb_kwargs(cat_idx):
    return dict(
        block_type="cross", num_layers=2, low_rank_ratio=0.25,
        use_batch_norm=True, norm_type="layer",
        alpha_init=0.02, alpha_shape="per-channel",
        cat_features=cat_idx if cat_idx else None,
        cat_encoder="embedding",
        cat_embed_dim="auto",
        num_encoder="plr", plr_freqs=24, plr_emb_dim="auto", plr_sigma=1.0,
        lr=5e-3, weight_decay=3e-3, gate_lr_factor=3.0,
        label_smoothing=0.15, max_grad_norm=2.0, betas=(0.9, 0.97),
        epochs=150, patience=10, lr_schedule="coslog4",
        # Cap backbone-facing d at 200 — TabICL's col-attention is O(C²) so
        # we want PLR's expanded features to project back down before hitting
        # the frozen backbone. Adapter still operates in PLR-expanded space.
        max_d=200,
        max_context="auto", max_query=2000,
        identity_guard_tol=0.005,
        device="cuda", random_state=42, verbose=False,
    )


class PEBAGModel(AbstractModel):
    def _get_default_auxiliary_params(self):
        out = super()._get_default_auxiliary_params()
        out["valid_raw_types"] = ["int", "float", "category", "object", "bool"]
        return out
    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, **kwargs):
        Xn = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        Xvn = (X_val.values.astype(np.float32) if isinstance(X_val, pd.DataFrame)
               else np.asarray(X_val, dtype=np.float32)) if X_val is not None else None
        cat_idx = []
        if isinstance(X, pd.DataFrame):
            cat_idx = [i for i, c in enumerate(X.columns)
                       if X[c].dtype == object or str(X[c].dtype) == "category"]
        p = self._get_model_params()
        kw = _peb_kwargs(cat_idx)
        self.model = BaggedRetoucheTabICLClassifier(
            n_estimators=p.get("ensemble_k", 4), **kw)
        if Xvn is not None and y_val is not None:
            self.model.fit(Xn, y, X_val=Xvn, y_val=np.asarray(y_val))
        else:
            self.model.fit(Xn, y)
        return self
    def _predict_proba(self, X, **kw):
        Xn = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        p = self.model.predict_proba(Xn)
        return p[:, 1] if self.problem_type == "binary" else p


class RawAGModel(AbstractModel):
    def _get_default_auxiliary_params(self):
        out = super()._get_default_auxiliary_params()
        out["valid_raw_types"] = ["int", "float", "category", "object", "bool"]
        return out
    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, **kwargs):
        Xn = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        self.model = TabICLClassifier(device="cuda", n_estimators=8, batch_size=8,
                                       random_state=42, verbose=False)
        self.model.fit(Xn, np.asarray(y))
        return self
    def _predict_proba(self, X, **kw):
        Xn = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        p = self.model.predict_proba(Xn)
        return p[:, 1] if self.problem_type == "binary" else p


def _pos(proba):
    if isinstance(proba, pd.DataFrame):
        return proba.iloc[:, -1].values
    a = np.asarray(proba)
    return a[:, -1] if a.ndim == 2 else a


def load(did):
    ds = openml.datasets.get_dataset(did, download_data=True)
    df, *_ = ds.get_data()
    target = ds.default_target_attribute
    df[target] = LabelEncoder().fit_transform(df[target].astype(str)).astype(int)
    for c in df.columns:
        if c == target:
            continue
        if df[c].dtype == object or str(df[c].dtype) == "category":
            df[c] = LabelEncoder().fit_transform(df[c].astype(str).fillna("MISSING"))
    return df, target


DATASETS = [
    ("taiwanese_bankruptcy", 46962),
    ("kc1",                  1067),
    ("blood-transfusion",    1464),
    ("credit-g",             31),
    ("qsar-biodeg",          1494),
    ("churn",                40701),
    ("phoneme",              1489),
    ("electricity",          151),
    ("diabetes",             37),
    ("ilpd",                 1480),
    ("ionosphere",           59),
    ("kr-vs-kp",             3),
    ("spambase",             44),
    ("sick",                 38),
    ("breast-w",             15),
    ("heart-statlog",        53),
    ("hill-valley",          1479),
    ("madelon",              1485),
    ("MagicTelescope",       1120),
    ("eeg-eye-state",        1471),
    ("monks-problems-1",     333),
    ("monks-problems-2",     334),
    ("monks-problems-3",     335),
    ("tic-tac-toe",          50),
    ("banknote-authentication", 1462),
    ("climate-model-simulation-crashes", 1467),
    ("wdbc",                 1510),
    ("cylinder-bands",       6332),
    ("ozone-level-8hr",      1487),
]


def main():
    print("="*78, flush=True)
    print("Retouche-PEB (improved) vs raw TabICL — 30 datasets, AG 8-fold bagged", flush=True)
    print("="*78, flush=True)

    results = []
    for name, did in DATASETS:
        print(f"\n──── {name} (did={did})", flush=True)
        try:
            df, target = load(did)
        except Exception as e:
            print(f"  LOAD FAILED: {e}", flush=True)
            continue
        y = df[target].values
        n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
        if n_pos < 24 or n_neg < 24:
            print(f"  SKIP: n_pos={n_pos} n_neg={n_neg}", flush=True)
            continue
        print(f"  shape={df.shape} pos={n_pos} neg={n_neg}", flush=True)

        df_tr, df_te = train_test_split(df, test_size=0.2, stratify=y, random_state=42)
        df_tr = df_tr.reset_index(drop=True); df_te = df_te.reset_index(drop=True)
        y_te = df_te[target].values

        # Baseline
        bl_path = f"/tmp/ag_bl_{name}"
        os.system(f"rm -rf {bl_path}")
        pred_bl = TabularPredictor(label=target, eval_metric="roc_auc",
                                    path=bl_path, verbosity=0)
        t0 = time.time()
        try:
            pred_bl.fit(
                df_tr, time_limit=1800,
                hyperparameters={RawAGModel: [{}]},
                num_bag_folds=8, num_stack_levels=0,
                ag_args_fit={"num_gpus": 1},
                ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
            )
            proba_bl = _pos(pred_bl.predict_proba(df_te.drop(columns=[target])))
            bl_auc = roc_auc_score(y_te, proba_bl)
            bl_ll = log_loss(y_te, np.clip(proba_bl, 1e-7, 1-1e-7), labels=[0, 1])
            bl_t = time.time() - t0
            print(f"  BL  AUC={bl_auc:.4f} LL={bl_ll:.4f} ({bl_t:.0f}s)", flush=True)
        except Exception as e:
            print(f"  BL CRASH: {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        finally:
            os.system(f"rm -rf {bl_path}")

        # PEB
        rt_path = f"/tmp/ag_peb_{name}"
        os.system(f"rm -rf {rt_path}")
        pred_rt = TabularPredictor(label=target, eval_metric="roc_auc",
                                    path=rt_path, verbosity=0)
        t0 = time.time()
        try:
            pred_rt.fit(
                df_tr, time_limit=5400,
                hyperparameters={PEBAGModel: [{}]},
                num_bag_folds=8, num_stack_levels=0,
                ag_args_fit={"num_gpus": 1},
                ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
            )
            proba_rt = _pos(pred_rt.predict_proba(df_te.drop(columns=[target])))
            rt_auc = roc_auc_score(y_te, proba_rt)
            rt_ll = log_loss(y_te, np.clip(proba_rt, 1e-7, 1-1e-7), labels=[0, 1])
            rt_t = time.time() - t0
            d_auc = rt_auc - bl_auc
            d_ll = bl_ll - rt_ll
            arrow = "↑" if d_auc > 0 else ("=" if d_auc == 0 else "↓")
            print(f"  PEB AUC={rt_auc:.4f} LL={rt_ll:.4f} "
                  f"ΔAUC={d_auc:+.4f} {arrow} ΔLL={d_ll:+.4f}  ({rt_t:.0f}s)", flush=True)
            results.append({
                "name": name, "did": did, "shape": list(df.shape),
                "bl_auc": float(bl_auc), "bl_ll": float(bl_ll),
                "peb_auc": float(rt_auc), "peb_ll": float(rt_ll),
                "d_auc": float(d_auc), "d_ll": float(d_ll),
                "bl_secs": bl_t, "peb_secs": rt_t,
            })
        except Exception as e:
            print(f"  PEB CRASH: {type(e).__name__}: {str(e)[:120]}", flush=True)
            results.append({
                "name": name, "did": did,
                "bl_auc": float(bl_auc), "bl_ll": float(bl_ll),
                "peb_auc": None, "err": str(e)[:200]})
        finally:
            os.system(f"rm -rf {rt_path}")

        with open("/root/peb30_results.json", "w") as fh:
            json.dump(results, fh, indent=2)

    # Summary
    print("\n" + "="*78, flush=True)
    print("SUMMARY (held-out test AUC; Δ = PEB - BL)", flush=True)
    print("="*78, flush=True)
    print(f"{'dataset':<32}{'BL':>9}{'PEB':>9}{'ΔAUC':>10}{'ΔLL':>10}", flush=True)
    print("-"*78, flush=True)
    rows = [r for r in results if r.get("peb_auc") is not None]
    for r in rows:
        print(f"{r['name'][:30]:<32}{r['bl_auc']:>9.4f}{r['peb_auc']:>9.4f}"
              f"{r['d_auc']:>+10.4f}{r['d_ll']:>+10.4f}", flush=True)
    print("-"*78, flush=True)
    if rows:
        ds = np.array([r["d_auc"] for r in rows])
        dl = np.array([r["d_ll"] for r in rows])
        w_a = int((ds>0).sum()); l_a = int((ds<0).sum()); t_a = int((ds==0).sum())
        w_l = int((dl>0).sum()); l_l = int((dl<0).sum()); t_l = int((dl==0).sum())
        print(f"AUC      mean Δ={ds.mean():+.4f}  W/T/L = {w_a}/{t_a}/{l_a}  win_rate={w_a/max(1,w_a+l_a):.1%}", flush=True)
        print(f"LogLoss  mean Δ={dl.mean():+.4f}  W/T/L = {w_l}/{t_l}/{l_l}  win_rate={w_l/max(1,w_l+l_l):.1%}", flush=True)
        print(f"(Paper TALENT 170 ds: AUC 58.4%, LogLoss 64.1%)", flush=True)

    with open("/root/peb30_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved /root/peb30_results.json", flush=True)


if __name__ == "__main__":
    main()
