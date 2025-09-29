#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Day 4 — Đánh giá & tinh chỉnh:
- Thu logits/prob trên DEV & TEST
- Tìm threshold tối ưu per-label (tối đa F1 trên DEV)
- Vẽ PR curve per-label (TEST) + overlay điểm baseline lexical (nếu có)
- Xuất metrics (micro/macro F1, P, R) cho cả model & baseline
- Lưu ảnh vào report/figures/ và bảng vào report/metrics/
"""

import os, sys, json, argparse, numpy as np, pandas as pd
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")  # để không cần GUI
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader
import torch

# đảm bảo import được train_bert.py trong cùng thư mục
CUR = os.path.dirname(__file__)
if CUR not in sys.path: sys.path.append(CUR)
from train_bert import BertMultiLabel, ENTITIES, QueryDataset
from transformers import BertTokenizerFast

# =============== Lexical baseline (điểm tham chiếu) ===================
# Bạn có thể mở rộng rule set này theo Day 2 (dummy rules)
RULES = {
    "Genre": ["horror","comedy","romance","sci-fi","action","drama","thriller","fantasy","documentary","animation","crime"],
    "CastAndCrew": ["tom hanks","leonardo dicaprio","scarlett johansson","keanu reeves","nolan","tarantino","spielberg","scorsese","greta gerwig"],
    "AudioLanguage": ["arabic","bangla","chinese","english","french","german","hindi","italian","japanese","korean","portuguese","spanish","turkish","vietnamese"],
    "ReleaseYear": r"\b(19|20)\d{2}\b",
    "Decade": ["80s","90s","2000s","2010s","2020s"],
    "StreamingService": ["netflix","hbo max","amazon prime","apple tv","hulu","disney+","peacock","paramount+"],
    "FreeContent": ["free"],
    "Franchise": ["avengers","batman","spider-man","star wars","harry potter","mission impossible","fast and furious","james bond"],
    "Holiday": ["christmas","halloween","thanksgiving","easter","valentine"],
    "Sport": ["football","basketball","tennis","cricket","baseball","soccer","formula 1","rugby","golf","hockey","manchester united","serena williams","nadal","lebron"],
    "Character": ["batman","charlie brown","sherlock holmes","james bond","spider-man","wonder woman","superman"],
}

import re
def lexical_predict(text: str) -> List[int]:
    q = text.lower()
    y = {e:0 for e in ENTITIES}
    # intent
    if any(w in q for w in ["movie","movies","film","films"]): y["IntentMovie"]=1
    if any(w in q for w in ["tv","series","show","shows"]): y["IntentTvSeries"]=1
    # popularity/recency
    if any(w in q for w in ["popular","trending","top charts","most-watched"]): y["Popularity"]=1
    if any(w in q for w in ["new ","newly","recent","latest","released in "]): y["Recency"]=1
    # names (đơn giản)
    if '"' in q: y["MovieName"]=1
    # regex/list rules
    for e, rule in RULES.items():
        if isinstance(rule, str):
            if re.search(rule, q): y[e]=1
        else:
            if any(tok in q for tok in rule): y[e]=1
    # nếu query nhắc tới "like <series>" => TVSeriesName
    if "tv series like " in q or "shows like " in q: y["TVSeriesName"]=1
    return [y[e] for e in ENTITIES]

# =================== Helper metrics ======================
def binarize(P: np.ndarray, th_vec: Dict[str, float]) -> np.ndarray:
    # P: [N, L], th_vec: dict entity->th
    L = P.shape[1]
    Yhat = np.zeros_like(P, dtype=int)
    for j, name in enumerate(ENTITIES):
        t = th_vec.get(name, 0.5)
        Yhat[:, j] = (P[:, j] >= t).astype(int)
    return Yhat

def compute_micro_macro(Y: np.ndarray, Yhat: np.ndarray):
    micro_f1 = f1_score(Y, Yhat, average="micro", zero_division=0)
    macro_f1 = f1_score(Y, Yhat, average="macro", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(Y, Yhat, average="micro", zero_division=0)
    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "micro_p": p, "micro_r": r}

def per_label_f1(Y_true: np.ndarray, Y_pred: np.ndarray):
    p, r, f, s = precision_recall_fscore_support(Y_true, Y_pred, average=None, zero_division=0)
    rows = []
    for j, name in enumerate(ENTITIES):
        rows.append({"label": name, "precision": float(p[j]), "recall": float(r[j]), "f1": float(f[j]), "support": int(s[j])})
    return pd.DataFrame(rows)

# =================== Collect logits/probs =================
def collect_probs(model, dl, device):
    model.eval()
    Ys, Ps = [], []
    with torch.no_grad():
        for x, y in dl:
            x = {k: v.to(device) for k, v in x.items()}
            logits = model(**x).cpu().numpy()
            probs = 1/(1+np.exp(-logits))
            Ys.append(y.numpy()); Ps.append(probs)
    return np.vstack(Ys), np.vstack(Ps)

# =================== Search thresholds on DEV ============
def find_thresholds_dev(Y_dev: np.ndarray, P_dev: np.ndarray, strategy="f1", beta=1.0):
    th = {}
    for j, name in enumerate(ENTITIES):
        best, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            pred = (P_dev[:, j] >= t).astype(int)
            if strategy == "f1":
                p,r,f,_ = precision_recall_fscore_support(Y_dev[:,j], pred, average="binary", zero_division=0)
                score = f
            else:
                # F-beta
                p,r,_f,_ = precision_recall_fscore_support(Y_dev[:,j], pred, average="binary", zero_division=0)
                if (beta*beta*p + r) == 0:
                    score = 0
                else:
                    score = (1+beta*beta) * (p*r) / (beta*beta*p + r)
            if score > best:
                best, best_t = score, t
        th[name] = float(best_t)
    return th

# =================== Plot PR per-label (TEST) ============
def plot_pr_curves(Y_test: np.ndarray, P_test: np.ndarray, out_dir: str, baseline_points: Dict[str, Tuple[float,float]] = None):
    os.makedirs(out_dir, exist_ok=True)
    for j, name in enumerate(ENTITIES):
        y_true = Y_test[:, j]
        y_score = P_test[:, j]
        if y_true.sum() == 0:
            # không có dương, bỏ qua
            continue
        precisions, recalls, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        plt.figure(figsize=(5,4))
        plt.step(recalls, precisions, where="post")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"{name} — PR curve (AP={ap:.3f})")
        plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])

        # overlay baseline point nếu có
        if baseline_points and name in baseline_points:
            bp = baseline_points[name]
            if bp is not None:
                plt.scatter([bp[1]], [bp[0]], marker="x")  # (recall on x, precision on y)
                plt.text(bp[1], bp[0], " baseline", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pr_{name}.png"))
        plt.close()

# =================== Baseline metrics (TEST) =============
def baseline_eval(path_jsonl: str) -> Tuple[np.ndarray, np.ndarray]:
    # trả Y_true, Y_pred_baseline (0/1)
    rows = [json.loads(l) for l in open(path_jsonl, encoding="utf-8")]
    Y = []
    Yhat = []
    for r in rows:
        y = [int(r["labels"].get(e,0)) for e in ENTITIES]
        yhat = lexical_predict(r["text"])
        Y.append(y); Yhat.append(yhat)
    return np.vstack(Y), np.vstack(Yhat)

def per_label_baseline_points(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, Tuple[float,float]]:
    # trả precision, recall của baseline cho từng nhãn (để plot điểm)
    p, r, f, s = precision_recall_fscore_support(Y_true, Y_pred, average=None, zero_division=0)
    out = {}
    for j, name in enumerate(ENTITIES):
        out[name] = (float(p[j]), float(r[j]))
    return out

# =================== Main =================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--ckpt", type=str, default="models/bert_intent.pt")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--fig_dir", type=str, default="report/figures")
    ap.add_argument("--metrics_dir", type=str, default="report/metrics")
    ap.add_argument("--beta", type=float, default=1.0, help="dùng F-beta khi chọn threshold (beta=1 => F1)")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dev = QueryDataset(os.path.join(args.data_dir,"splits","dev.jsonl"), tok, args.max_len)
    test = QueryDataset(os.path.join(args.data_dir,"splits","test.jsonl"), tok, args.max_len)

    dl_dev = DataLoader(dev, batch_size=64, shuffle=False, num_workers=0)
    dl_test = DataLoader(test, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertMultiLabel()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    # thu prob
    Yd, Pd = collect_probs(model, dl_dev, device)
    Yt, Pt = collect_probs(model, dl_test, device)

    # chọn threshold theo DEV
    th = find_thresholds_dev(Yd, Pd, strategy=("f1" if args.beta==1.0 else "fbeta"), beta=args.beta)

    # LƯU thresholds.json 
    with open(os.path.join(args.metrics_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(th, f, indent=2)
    print("Saved per-label thresholds ->", os.path.join(args.metrics_dir, "thresholds.json"))

    # áp dụng threshold lên TEST
    Yhat = binarize(Pt, th)
    agg = compute_micro_macro(Yt, Yhat)
    df_per = per_label_f1(Yt, Yhat)

    # baseline lexical trên TEST
    Yt_base, Yhat_base = baseline_eval(os.path.join(args.data_dir,"splits","test.jsonl"))
    agg_base = compute_micro_macro(Yt_base, Yhat_base)
    df_base = per_label_f1(Yt_base, Yhat_base)

    # lưu metrics
    pd.DataFrame([agg]).to_csv(os.path.join(args.metrics_dir, "model_agg.csv"), index=False)
    df_per.to_csv(os.path.join(args.metrics_dir, "model_per_label.csv"), index=False)
    pd.DataFrame([agg_base]).to_csv(os.path.join(args.metrics_dir, "baseline_agg.csv"), index=False)
    df_base.to_csv(os.path.join(args.metrics_dir, "baseline_per_label.csv"), index=False)

    # plot PR per-label (TEST) + overlay baseline point
    base_points = per_label_baseline_points(Yt_base, Yhat_base)
    plot_pr_curves(Yt, Pt, args.fig_dir, baseline_points=base_points)

    # in tóm tắt
    print("== MODEL (threshold per-label on DEV) @ TEST ==")
    print(f"micro-F1={agg['micro_f1']:.4f} macro-F1={agg['macro_f1']:.4f} P={agg['micro_p']:.4f} R={agg['micro_r']:.4f}")
    print("== BASELINE (lexical) @ TEST ==")
    print(f"micro-F1={agg_base['micro_f1']:.4f} macro-F1={agg_base['macro_f1']:.4f} P={agg_base['micro_p']:.4f} R={agg_base['micro_r']:.4f}")
    print(f"Saved figures -> {args.fig_dir}")
    print(f"Saved metrics -> {args.metrics_dir}")
    print("Top few per-label (model):")
    print(df_per.sort_values('f1', ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    main()
