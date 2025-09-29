#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm import tqdm

ENTITIES = [
    "IntentMovie","IntentTvSeries","Theme","Genre","CastAndCrew","TVSeriesName","MovieName",
    "StreamingService","Recency","Popularity","ReleaseYear","Decade","FreeContent","AudioLanguage",
    "Franchise","Holiday","Sport","Character"
]

# ---------------- Dataset ----------------
class QueryDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        text = r["text"]
        labels = [int(r["labels"].get(e,0)) for e in ENTITIES]
        enc = self.tokenizer(text, truncation=True, padding="max_length",
                             max_length=self.max_len, return_tensors="pt")
        x = {k:v.squeeze(0) for k,v in enc.items()}
        y = torch.tensor(labels, dtype=torch.float)
        return x, y

# --------------- Model -------------------
class BertMultiLabel(nn.Module):
    def __init__(self, name="bert-base-uncased", num_labels=len(ENTITIES), dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(name, output_hidden_states=True)
        hidden = self.bert.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden*4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids  # <- thêm dòng này
        )
        hs = out.hidden_states
        cls_concat = torch.cat([hs[-1][:,0,:], hs[-2][:,0,:], hs[-3][:,0,:], hs[-4][:,0,:]], dim=1)
        logits = self.mlp(cls_concat)
        return logits

# ------------- Utils ---------------------
def ensure_splits(weak_path, out_dir):
    train_p = os.path.join(out_dir, "splits", "train.jsonl")
    if os.path.exists(train_p):
        return
    rows = [json.loads(l) for l in open(weak_path, encoding="utf-8")]
    random.seed(7); random.shuffle(rows)
    n = len(rows); n_tr = int(0.7*n); n_dv = int(0.1*n)
    os.makedirs(os.path.join(out_dir, "splits"), exist_ok=True)
    with open(os.path.join(out_dir,"splits","train.jsonl"),"w",encoding="utf-8") as f:
        for r in rows[:n_tr]: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with open(os.path.join(out_dir,"splits","dev.jsonl"),"w",encoding="utf-8") as f:
        for r in rows[n_tr:n_tr+n_dv]: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with open(os.path.join(out_dir,"splits","test.jsonl"),"w",encoding="utf-8") as f:
        for r in rows[n_tr+n_dv:]: f.write(json.dumps(r, ensure_ascii=False)+"\n")

def binarize(probs, th=0.5):
    return (probs >= th).astype(int)

def eval_on_loader(model, dl, device, threshold=0.5):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x,y in dl:
            x = {k:v.to(device) for k,v in x.items()}
            y = y.numpy()
            logits = model(**x).cpu().numpy()
            probs = 1/(1+np.exp(-logits))
            ys.append(y); ps.append(probs)
    Y = np.vstack(ys); P = np.vstack(ps)
    Yhat = binarize(P, threshold)
    micro = f1_score(Y, Yhat, average="micro", zero_division=0)
    macro = f1_score(Y, Yhat, average="macro", zero_division=0)
    p,r,f,_ = precision_recall_fscore_support(Y, Yhat, average="micro", zero_division=0)
    return {"micro_f1":micro,"macro_f1":macro,"micro_p":p,"micro_r":r}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--weak_path", type=str, default="data/weak_labels.jsonl")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ckpt", type=str, default="models/bert_intent.pt")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    ensure_splits(args.weak_path, args.data_dir)

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tr = QueryDataset(os.path.join(args.data_dir,"splits","train.jsonl"), tok, args.max_len)
    dv = QueryDataset(os.path.join(args.data_dir,"splits","dev.jsonl"), tok, args.max_len)

    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_dv = DataLoader(dv, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertMultiLabel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    best_micro = -1
    for ep in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"epoch {ep}")
        for x,y in pbar:
            x = {k:v.to(device) for k,v in x.items()}
            y = y.to(device)
            logits = model(**x)
            loss = bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # dev
        metrics = eval_on_loader(model, dl_dv, device, threshold=0.5)
        print(f"[dev] micro-F1={metrics['micro_f1']:.4f} | macro-F1={metrics['macro_f1']:.4f} | "
              f"P={metrics['micro_p']:.4f} R={metrics['micro_r']:.4f}")

        if metrics["micro_f1"] > best_micro:
            best_micro = metrics["micro_f1"]
            torch.save(model.state_dict(), args.ckpt)
            print(f"✅ saved best to {args.ckpt}")

    print("Done.")

if __name__ == "__main__":
    main()
