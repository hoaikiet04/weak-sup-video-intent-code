#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, random, re
from typing import Dict, List, Any, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizerFast

# ===== Project imports =====
# Lấy model & danh sách nhãn từ Day 3
from src.train_bert import BertMultiLabel, ENTITIES

# ===================== ENV CONFIG =====================
CKPT_PATH  = os.getenv("CKPT_PATH", "models/bert_intent.pt4")
TH_PATH    = os.getenv("THRESHOLDS_PATH", "report/metrics/thresholds.json")
WEAK_PATH  = os.getenv("WEAK_LABELS_PATH", "data/weak_labels.jsonl")

DEFAULT_MAXLEN = int(os.getenv("MAX_LEN", "128"))
MAX_REASONS_PER_LABEL = int(os.getenv("MAX_REASONS_PER_LABEL", "80"))  # giới hạn RAM
MAX_LINES_PER_LABEL_OUT = int(os.getenv("MAX_LINES_PER_LABEL_OUT", "3"))

# ===================== LOAD THRESHOLDS =====================
if os.path.exists(TH_PATH):
    with open(TH_PATH, "r", encoding="utf-8") as f:
        THRESHOLDS: Dict[str, float] = json.load(f)
else:
    THRESHOLDS = {e: 0.5 for e in ENTITIES}
    print(f"[api] WARNING: thresholds.json not found at {TH_PATH} -> using 0.5 for all")

# ===================== LOAD TOKENIZER & MODEL =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

model = BertMultiLabel().to(device)
state = torch.load(CKPT_PATH, map_location=device)
# state có thể là state_dict hoặc một wrapper có .state_dict()
if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
    model.load_state_dict(state)
else:
    model.load_state_dict(state.state_dict())
model.eval()
print(f"[api] Model loaded from {CKPT_PATH} on {device}, #labels={len(ENTITIES)}")

# ===================== REASONING INDEX (from weak labels) =====================
# Dạng: label -> list[str] (câu ngắn để hiện Explain)
_REASON_IDX: Dict[str, List[str]] = {e: [] for e in ENTITIES}

def _normalize_reasoning(reasoning_obj: Any) -> List[str]:
    """
    Chuẩn hóa mọi kiểu reasoning về list[str]:
      - str: tách dòng/bullet.
      - list[str]: giữ lại, strip từng câu.
      - dict[label->str/list]: gộp lại tất cả value về list[str].
    Cắt ngắn mỗi câu ~240 ký tự cho gọn UI.
    """
    out: List[str] = []
    if not reasoning_obj:
        return out
    if isinstance(reasoning_obj, str):
        s = reasoning_obj.replace("•", "\n")
        for line in s.split("\n"):
            t = line.strip(" -•\t\r ")
            if t:
                out.append(t)
    elif isinstance(reasoning_obj, list):
        for x in reasoning_obj:
            t = str(x).strip()
            if t:
                out.append(t)
    elif isinstance(reasoning_obj, dict):
        for v in reasoning_obj.values():
            out.extend(_normalize_reasoning(v))

    cleaned = []
    for t in out:
        if len(t) > 240:
            t = t[:237] + "..."
        if t:
            cleaned.append(t)
    return cleaned

# Detect “label-list” kiểu "IntentTvSeries; Genre; Recency"
_ENT_SET = {e.lower() for e in ENTITIES}
_LABEL_LIST_RE = re.compile(r"^[A-Za-z]+(?:\s*[;/,]\s*[A-Za-z]+)+$")

def _is_label_list_reason(s: str) -> bool:
    if not s:
        return False
    if not _LABEL_LIST_RE.match(s.strip()):
        return False
    tokens = re.split(r"[;/,]\s*", s.strip())
    return all(t.lower() in _ENT_SET for t in tokens)

def _build_reason_index():
    total_rows = 0
    if not os.path.exists(WEAK_PATH):
        print(f"[api] NOTE: weak labels not found at {WEAK_PATH} -> Explain (weak) will be empty")
        return
    try:
        with open(WEAK_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_rows += 1
                row = json.loads(line)
                labels = row.get("labels", {}) or {}
                reasons = _normalize_reasoning(row.get("reasoning"))
                if not reasons:
                    continue
                # bỏ các chuỗi chỉ là danh sách nhãn
                reasons = [r for r in reasons if not _is_label_list_reason(r)]
                if not reasons:
                    continue
                for e in ENTITIES:
                    if int(labels.get(e, 0)) == 1:
                        bucket = _REASON_IDX[e]
                        if len(bucket) < MAX_REASONS_PER_LABEL:
                            bucket.extend(reasons[:3])  # lấy vài câu đầu cho đa dạng
    except Exception as e:
        print("[api] ERROR loading weak labels:", e)
        return

    for e in ENTITIES:
        random.shuffle(_REASON_IDX[e])
        _REASON_IDX[e] = _REASON_IDX[e][:MAX_REASONS_PER_LABEL]

    # log thống kê
    filled = {e: len(v) for e, v in _REASON_IDX.items()}
    non_empty = sum(1 for v in filled.values() if v > 0)
    print(f"[api] Explain index built from {total_rows} rows. "
          f"Labels with reasons: {non_empty}/{len(ENTITIES)}")
    for e, c in filled.items():
        print(f"  - {e}: {c} reasons")

_build_reason_index()

# ===================== RULE-BASED EXPLAIN (from query) =====================
GENRES = {"thriller","horror","comedy","drama","romance","sci-fi","science fiction","action",
          "crime","fantasy","animation","documentary","adventure","mystery"}
SERVICES = {"netflix","disney+","disney plus","prime video","amazon prime","hbo","hbo max","max",
            "hulu","apple tv","apple tv+"}
TV_MARKERS = {"tv","series","show","season","episodes","ep"}
MOVIE_MARKERS = {"movie","movies","film","films","cinema"}
LANGS = {"english","korean","spanish","french","japanese","hindi","german","italian","thai","vietnamese"}
DECADE_PAT = re.compile(r"\b(?:[12]\d{3})s\b|\b(?:90s|80s|70s|60s|50s)\b", re.I)
YEAR_PAT = re.compile(r"\b(19|20)\d{2}\b")
FREE_MARKERS = {"free","for free","without subscription","no subscription"}
POPULARITY = {"top","best","popular","trending","most watched","hit"}
RECENCY = {"new","latest","recent","this year","this month","this week"}

def rule_explain(query: str, pos_labels: List[str], max_items_per_label: int = 3) -> Dict[str, List[str]]:
    q = query.lower()
    out: Dict[str, List[str]] = {}

    def add(lab, msg):
        out.setdefault(lab, [])
        if len(out[lab]) < max_items_per_label:
            out[lab].append(msg)

    # Intent (movie / tv)
    if "IntentTvSeries" in pos_labels and any(w in q for w in TV_MARKERS):
        add("IntentTvSeries", "Contains TV markers (e.g., 'TV', 'series', 'show').")
    if "IntentMovie" in pos_labels and any(w in q for w in MOVIE_MARKERS):
        add("IntentMovie", "Contains movie markers (e.g., 'movie', 'film').")

    # Genre
    if "Genre" in pos_labels:
        hits = [g for g in GENRES if re.search(rf"\b{re.escape(g)}\b", q)]
        if hits:
            add("Genre", f"Genre keyword(s) detected: {', '.join(hits)}.")

    # StreamingService
    if "StreamingService" in pos_labels:
        hits = [s for s in SERVICES if s in q]
        if hits:
            add("StreamingService", f"Streaming platform mentioned: {', '.join(hits)}.")

    # AudioLanguage
    if "AudioLanguage" in pos_labels:
        hits = [l for l in LANGS if re.search(rf"\b{re.escape(l)}\b", q)]
        if hits:
            add("AudioLanguage", f"Language mentioned: {', '.join(hits)}.")

    # ReleaseYear
    if "ReleaseYear" in pos_labels:
        y = YEAR_PAT.search(q)
        if y:
            add("ReleaseYear", f"Year detected: {y.group(0)}.")

    # Decade
    if "Decade" in pos_labels and DECADE_PAT.search(q):
        add("Decade", "Decade pattern detected (e.g., '90s', '2000s').")

    # Popularity / Recency / FreeContent
    if "Popularity" in pos_labels and any(w in q for w in POPULARITY):
        add("Popularity", "Popularity intent detected (e.g., 'best', 'top', 'popular').")
    if "Recency" in pos_labels and (any(w in q for w in RECENCY) or YEAR_PAT.search(q)):
        add("Recency", "Recency intent detected (e.g., 'new', 'latest', or a recent year).")
    if "FreeContent" in pos_labels and any(w in q for w in FREE_MARKERS):
        add("FreeContent", "Free-content intent detected (e.g., 'free').")

    return out

# ===================== SCHEMAS =====================
class PredictIn(BaseModel):
    query: str
    topk: Optional[int] = 5
    explain: Optional[bool] = False
    explain_source: Optional[str] = "rule"  # "rule" | "weak" | "both"
    max_len: Optional[int] = DEFAULT_MAXLEN

class PredictOut(BaseModel):
    query: str
    probs: Dict[str, float]
    labels: Dict[str, bool]
    over_threshold: List[str]
    thresholds: Dict[str, float]
    topk: List[Dict[str, Any]]
    explanations: Optional[Dict[str, List[str]]] = None

# ===================== HELPERS =====================
def predict_probs(text: str, max_len: int = DEFAULT_MAXLEN) -> Dict[str, float]:
    x = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    # một số bản model.forward chưa nhận token_type_ids -> fallback bỏ key
    try:
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in x.items()})
    except TypeError:
        x.pop("token_type_ids", None)
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in x.items()})
    probs = torch.sigmoid(logits).detach().cpu().squeeze(0).tolist()
    return {ENTITIES[i]: float(probs[i]) for i in range(len(ENTITIES))}

def apply_thresholds(probs: Dict[str, float], th: Dict[str, float]) -> Dict[str, bool]:
    return {e: probs[e] >= th.get(e, 0.5) for e in ENTITIES}

def get_explanations_from_index(pos_labels: List[str], limit_per_label: int = 3) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for e in pos_labels:
        cand = _REASON_IDX.get(e) or []
        # lọc bỏ chuỗi chỉ là danh sách nhãn
        cleaned = [c for c in cand if not _is_label_list_reason(c)]
        if cleaned:
            out[e] = cleaned[:limit_per_label]
    return out

# ===================== FASTAPI APP =====================
app = FastAPI(title="Video Query Intent API", version="1.3")

@app.get("/health")
def health():
    return {"ok": True, "model": "bert-base-uncased", "labels": len(ENTITIES)}

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    probs = predict_probs(body.query, max_len=body.max_len)
    labels = apply_thresholds(probs, THRESHOLDS)
    over = [e for e, v in labels.items() if v]
    topk = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[: max(1, body.topk or 5)]

    explanations = None
    if body.explain:
        src = (body.explain_source or "rule").lower()
        exp_rule = rule_explain(body.query, over, max_items_per_label=MAX_LINES_PER_LABEL_OUT) if src in ("rule","both") else {}
        exp_llm  = get_explanations_from_index(over, limit_per_label=MAX_LINES_PER_LABEL_OUT) if src in ("weak","both") else {}

        merged: Dict[str, List[str]] = {}
        for lab in over:
            lst: List[str] = []
            if lab in exp_rule:
                lst.extend(exp_rule[lab])
            if lab in exp_llm:
                for r in exp_llm[lab]:
                    if r not in lst:
                        lst.append(r)
            if lst:
                merged[lab] = lst[:MAX_LINES_PER_LABEL_OUT]
        explanations = merged or None

    return {
        "query": body.query,
        "probs": probs,
        "labels": labels,
        "over_threshold": over,
        "thresholds": THRESHOLDS,
        "topk": [{"label": k, "prob": float(v)} for k, v in topk],
        "explanations": explanations,
    }
