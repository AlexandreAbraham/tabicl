"""Joint training: adapter + model params in same optimizer. 5-day gap."""

import sys, warnings
sys.path.insert(0, "/root/neuralk_training")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from noise_adapter import NoiseEmbeddingAdapter


def build_features(df):
    close = df["Close"].squeeze(); high = df["High"].squeeze()
    low = df["Low"].squeeze(); opn = df["Open"].squeeze(); volume = df["Volume"].squeeze()
    f = {}
    for d in [5, 10, 20, 30, 60]:
        f[f"ROC_{d}"] = close.pct_change(d)
        f[f"MA_{d}"] = close.rolling(d).mean() / close - 1
        f[f"STD_{d}"] = close.rolling(d).std() / close
        f[f"MAX_{d}"] = close.rolling(d).max() / close - 1
        f[f"MIN_{d}"] = close.rolling(d).min() / close - 1
        f[f"RSV_{d}"] = (close - close.rolling(d).min()) / (close.rolling(d).max() - close.rolling(d).min() + 1e-8)
        f[f"CORR_{d}"] = close.rolling(d).corr(volume.rolling(d).mean())
    for d in [5, 10, 20, 60]:
        f[f"VROL_{d}"] = volume.rolling(d).mean() / (volume.rolling(max(d*2,60)).mean() + 1e-8)
        f[f"VSTD_{d}"] = volume.rolling(d).std() / (volume.rolling(d).mean() + 1e-8)
    f["KMID"] = (close - opn) / (opn + 1e-8)
    f["HIGH_LOW"] = (high - low) / (close + 1e-8)
    return pd.DataFrame(f, index=df.index)


def fetch_data(tickers):
    aX, ay, an, ad = [], [], [], []
    for t in tickers:
        try:
            df = yf.download(t, start="2015-01-01", end="2024-12-31", progress=False)
            if len(df) < 200: continue
            feat = build_features(df); close = df["Close"].squeeze()
            y = ((close.pct_change(5).shift(-5)) > 0).astype(int)
            noise = feat.rolling(20).std()
            v = feat.notna().all(1) & noise.notna().all(1) & y.notna()
            aX.append(feat[v].values.astype(np.float32)); ay.append(y[v].values.astype(np.int64))
            an.append(noise[v].values.astype(np.float32)); ad.append(feat[v].index)
        except: pass
    return (np.nan_to_num(np.concatenate(aX)), np.concatenate(ay),
            np.nan_to_num(np.concatenate(an)), np.concatenate(ad))


def forward_col(model, col_ad, X_t, y_t, noise_t):
    emb = model.col_embedder(X_t, y_train=y_t.long(), embed_with_test=False)
    G = emb.shape[2] - model.row_num_cls
    nm = noise_t.unsqueeze(0).unsqueeze(-1)[:, :, :G, :]
    if nm.shape[2] < G: nm = F.pad(nm, (0, 0, 0, G - nm.shape[2]))
    emb = col_ad(emb, nm)
    rep = model.row_interactor(emb)
    return model.icl_predictor(rep, y_train=y_t)


def train_joint(model, col_ad, X_tr, y_tr, noise_tr, epochs, device):
    """Train adapter + model jointly, different LRs via param groups."""
    for p in model.parameters(): p.requires_grad = True
    for p in col_ad.parameters(): p.requires_grad = True
    model.train()
    col_ad.train()

    optimizer = AdamW([
        {"params": col_ad.parameters(), "lr": 5e-4},
        {"params": model.parameters(), "lr": 1e-5},
    ], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    ctx_size = min(400, int(len(X_tr) * 0.8))

    for _ in range(epochs):
        for _ in range(15):
            perm = np.random.permutation(len(X_tr))
            ci, qi = perm[:ctx_size], perm[ctx_size:ctx_size+200]
            if len(qi) < 10: continue
            ai = np.concatenate([ci, qi])
            X_t = torch.from_numpy(X_tr[ai]).unsqueeze(0).to(device)
            y_t = torch.from_numpy(y_tr[ci]).float().unsqueeze(0).to(device)
            y_q = torch.from_numpy(y_tr[qi]).long().to(device)
            n_t = torch.from_numpy(noise_tr[ai]).to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = forward_col(model, col_ad, X_t, y_t, n_t)
                loss = F.cross_entropy(logits[0, :, :2], y_q)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()


def train_sequential(model, col_ad, X_tr, y_tr, noise_tr, epochs, device):
    """Train adapter first, then model. Same total epochs."""
    half = epochs // 2
    # Phase 1: adapter only
    for p in model.parameters(): p.requires_grad = False
    for p in col_ad.parameters(): p.requires_grad = True
    model.train(); col_ad.train()
    opt1 = AdamW(col_ad.parameters(), lr=5e-4, weight_decay=0.01)
    scaler1 = torch.amp.GradScaler("cuda")
    ctx_size = min(400, int(len(X_tr) * 0.8))
    for _ in range(half):
        for _ in range(15):
            perm = np.random.permutation(len(X_tr))
            ci, qi = perm[:ctx_size], perm[ctx_size:ctx_size+200]
            if len(qi) < 10: continue
            ai = np.concatenate([ci, qi])
            X_t = torch.from_numpy(X_tr[ai]).unsqueeze(0).to(device)
            y_t = torch.from_numpy(y_tr[ci]).float().unsqueeze(0).to(device)
            y_q = torch.from_numpy(y_tr[qi]).long().to(device)
            n_t = torch.from_numpy(noise_tr[ai]).to(device)
            opt1.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = forward_col(model, col_ad, X_t, y_t, n_t)
                loss = F.cross_entropy(logits[0, :, :2], y_q)
            scaler1.scale(loss).backward(); scaler1.step(opt1); scaler1.update()

    # Phase 2: model only
    for p in model.parameters(): p.requires_grad = True
    for p in col_ad.parameters(): p.requires_grad = False
    model.train(); col_ad.eval()
    opt2 = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.01)
    scaler2 = torch.amp.GradScaler("cuda")
    for _ in range(half):
        for _ in range(15):
            perm = np.random.permutation(len(X_tr))
            ci, qi = perm[:ctx_size], perm[ctx_size:ctx_size+200]
            if len(qi) < 10: continue
            ai = np.concatenate([ci, qi])
            X_t = torch.from_numpy(X_tr[ai]).unsqueeze(0).to(device)
            y_t = torch.from_numpy(y_tr[ci]).float().unsqueeze(0).to(device)
            y_q = torch.from_numpy(y_tr[qi]).long().to(device)
            n_t = torch.from_numpy(noise_tr[ai]).to(device)
            opt2.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = forward_col(model, col_ad, X_t, y_t, n_t)
                loss = F.cross_entropy(logits[0, :, :2], y_q)
            scaler2.scale(loss).backward(); scaler2.step(opt2); scaler2.update()


def evaluate(model, col_ad, Xc, yc, nc, Xt, yt, nt, device):
    model.train(); col_ad.eval()
    cs = min(400, len(Xc)); rng = np.random.default_rng(0)
    ci = rng.choice(len(Xc), cs, replace=False); preds = []
    with torch.no_grad():
        for s in range(0, len(Xt), 200):
            e = min(s+200, len(Xt))
            Xa = np.concatenate([Xc[ci], Xt[s:e]]); na = np.concatenate([nc[ci], nt[s:e]])
            XT = torch.from_numpy(Xa).unsqueeze(0).to(device)
            yT = torch.from_numpy(yc[ci]).float().unsqueeze(0).to(device)
            nT = torch.from_numpy(na).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = forward_col(model, col_ad, XT, yT, nT)
            preds.append(logits[0, :, :2].argmax(-1).cpu().numpy())
    return accuracy_score(yt[:sum(len(p) for p in preds)], np.concatenate(preds))


def main():
    print("Joint vs Sequential (Col+FT, 5-day gap, 20 epochs total)")
    tickers = ["AAPL","MSFT","GOOGL","AMZN","JPM","GS","XOM","PFE",
               "WMT","KO","BA","CAT","DIS","NFLX","NVDA","TSLA",
               "META","V","MA","UNH","HD","CRM","COST","ABT"]
    print("Fetching...")
    X_raw, y, noise_raw, dates = fetch_data(tickers)
    order = np.argsort(dates)
    X_raw, y, noise_raw = X_raw[order], y[order], noise_raw[order]
    n = len(X_raw); GAP = 5; device = "cuda"
    print(f"{n} samples")

    from tabicl import TabICLClassifier
    joint_res, seq_res = [], []

    for fold in range(5):
        tr_s = int(fold * 0.1 * n); tr_e = tr_s + int(0.5 * n)
        te_s = tr_e + GAP; te_e = te_s + int(0.1 * n)
        if te_e > n: break
        sc = StandardScaler()
        X_tr = np.nan_to_num(sc.fit_transform(X_raw[tr_s:tr_e])).astype(np.float32)
        X_te = np.nan_to_num(sc.transform(X_raw[te_s:te_e])).astype(np.float32)
        nsc = StandardScaler()
        n_tr = np.nan_to_num(nsc.fit_transform(noise_raw[tr_s:tr_e])).astype(np.float32)
        n_te = np.nan_to_num(nsc.transform(noise_raw[te_s:te_e])).astype(np.float32)
        y_tr, y_te = y[tr_s:tr_e], y[te_s:te_e]

        print(f"\nFold {fold+1}:", end="", flush=True)

        # Joint
        clf = TabICLClassifier(n_estimators=1, device=device); clf.fit(X_tr, y_tr)
        col_ad = NoiseEmbeddingAdapter(embed_dim=128, noise_dim=1, hidden_dim=64, num_cls=clf.model_.row_num_cls).to(device)
        train_joint(clf.model_, col_ad, X_tr, y_tr, n_tr, epochs=20, device=device)
        acc = evaluate(clf.model_, col_ad, X_tr, y_tr, n_tr, X_te, y_te, n_te, device)
        joint_res.append(acc)
        print(f"  Joint={acc:.1%}", end="", flush=True)

        # Sequential
        clf = TabICLClassifier(n_estimators=1, device=device); clf.fit(X_tr, y_tr)
        col_ad = NoiseEmbeddingAdapter(embed_dim=128, noise_dim=1, hidden_dim=64, num_cls=clf.model_.row_num_cls).to(device)
        train_sequential(clf.model_, col_ad, X_tr, y_tr, n_tr, epochs=20, device=device)
        acc = evaluate(clf.model_, col_ad, X_tr, y_tr, n_tr, X_te, y_te, n_te, device)
        seq_res.append(acc)
        print(f"  Seq={acc:.1%}")

    print(f"\n{'Config':<15} " + " ".join([f"F{i+1:>5}" for i in range(len(joint_res))]) + f" {'Mean':>7}")
    print("-" * 50)
    print(f"{'Joint':<15} " + " ".join([f"{a:>5.1%}" for a in joint_res]) + f" {np.mean(joint_res):>6.1%}")
    print(f"{'Sequential':<15} " + " ".join([f"{a:>5.1%}" for a in seq_res]) + f" {np.mean(seq_res):>6.1%}")
    print(f"\nRef: Col+FT(prev)=54.6% | LightGBM=54.7%")


if __name__ == "__main__":
    main()
