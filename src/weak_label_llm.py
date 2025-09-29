#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Day 2 — Weak labeling with LLMs (CoT + ICL + Confidence)
Providers:
  - groq   : (đề xuất) llama-3.1-8b-instant | mixtral-8x7b-32768 | llama3-8b-8192 ...
  - openai : nếu còn quota
  - dummy  : fallback rule-based (không cần API)

Tính năng chịu lỗi:
  - Retry + fallback khi model trả JSON lỗi (json_validate_failed)
  - Ghi từng dòng trong lúc chạy (stream) để không mất dữ liệu
  - Resume: bỏ qua các id đã có trong file out

Cách chạy ví dụ (GROQ):
  python src/weak_label_llm.py \
    --provider groq \
    --model llama-3.1-8b-instant \
    --in data/synthetic_queries.jsonl \
    --out data/weak_labels.jsonl \
    --personas "Merchandiser,Movie Critic,Horror Aficionado" \
    --rate 0.4 \
    --limit 100 \
    --stream 1 --resume 1
"""

import os, json, time, random, argparse, re
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# (tuỳ chọn) nạp .env nếu cài python-dotenv
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ---------------- Taxonomy (18 entity theo Appendix A — có thể mở rộng 22+) ---
TAXONOMY = [
    {"name":"IntentMovie","definition":"Intent targets movies.","examples":["Leonardo DiCaprio movies","movies like Interstellar"]},
    {"name":"IntentTvSeries","definition":"Intent targets TV series/shows.","examples":["drama TV series","shows like The Marvelous Mrs. Maisel"]},
    {"name":"Theme","definition":"Abstract topic/concept.","examples":["based on true story","alien abduction"]},
    {"name":"Genre","definition":"Genre label.","examples":["romantic comedy","action-comedy movies"]},
    {"name":"CastAndCrew","definition":"Actors/directors/crew.","examples":["Tom Hanks movies","films directed by Christopher Nolan"]},
    {"name":"TVSeriesName","definition":"Specific TV series.","examples":["The Marvelous Mrs. Maisel","Stranger Things"]},
    {"name":"MovieName","definition":"Specific movie.","examples":["Interstellar","Oppenheimer"]},
    {"name":"StreamingService","definition":"Streaming platform.","examples":["HBO Max","Netflix"]},
    {"name":"Recency","definition":"New/recent content.","examples":["new movies","recent shows"]},
    {"name":"Popularity","definition":"Popular/trending.","examples":["popular movies","trending shows"]},
    {"name":"ReleaseYear","definition":"Exact year.","examples":["2023 movies","2019 TV show"]},
    {"name":"Decade","definition":"Decade of release.","examples":["90s movies","80s shows"]},
    {"name":"FreeContent","definition":"Free to watch.","examples":["free movies","where to watch free movies"]},
    {"name":"AudioLanguage","definition":"Audio language.","examples":["Arabic movies","Bangla movies"]},
    {"name":"Franchise","definition":"Franchise/IP.","examples":["Avengers","Batman"]},
    {"name":"Holiday","definition":"Holiday theme.","examples":["Christmas movies","Halloween specials"]},
    {"name":"Sport","definition":"Sports/teams/athletes.","examples":["football","Manchester United","Serena Williams"]},
    {"name":"Character","definition":"Fictional character.","examples":["Batman","Charlie Brown"]},
]
ENTITIES = [e["name"] for e in TAXONOMY]

# ---------------- Prompt builder (CoT-lite + ICL + Confidence) ----------------
def build_messages(query_text: str, persona: str = None, with_xml_hint=False) -> List[Dict[str, str]]:
    system = "You are an expert at understanding user search queries for movies/TV."
    defs = []
    for e in TAXONOMY:
        ex = "; ".join(e["examples"][:2]) if e.get("examples") else ""
        defs.append(f"- {e['name']}: {e['definition']} Examples: {ex}")
    persona_line = f"Answer as a {persona} persona." if persona else ""
    extra = ""
    if with_xml_hint:
        extra = "\nReturn ONLY JSON wrapped inside <json>...</json> without any extra text."

    user = f"""
{persona_line}
You must detect relevant entities from this taxonomy. Provide per-entity binary labels and confidence.

Taxonomy (definition + 1-2 examples):
{chr(10).join(defs)}

Instructions:
1) For the given query, iterate entity-by-entity. If the query expresses that entity, set label=1 else 0.
2) Provide brief reasons (short cues) only for matched entities.
3) For each entity assign confidence: high / med / low.
4) Return a single JSON object ONLY in this schema:
{{
 "id": "<same id>",
 "text": "<the query>",
 "labels": {{ "<EntityName>": 0/1, ... for ALL entities }},
 "confidence": {{ "<EntityName>": "low|med|high", ... for ALL entities }},
 "reasoning": "bullet-like short cues for matched entities"
}}
{extra}

Query: "{query_text}"
""".strip()
    return [{"role":"system","content":system},{"role":"user","content":user}]

# ---------------- JSON helpers -------------------------------------------------
def safe_json_parse(s: str) -> Dict[str, Any]:
    s = s.strip()
    # nếu có thẻ <json>...</json> → lấy phần trong thẻ
    m_xml = re.search(r"<json>(.*?)</json>", s, flags=re.S|re.I)
    if m_xml:
        s = m_xml.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try: 
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}
def _coerce_label(v) -> int:
    # Chuẩn hoá mọi kiểu về 0/1
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if float(v) >= 0.5 else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "t", "on", "positive"):
            return 1
        if s in ("0", "false", "no", "n", "f", "off", "negative", "neg", "none", "null", ""):
            return 0
        # Một số model nhầm confidence vào labels:
        if s in ("high", "med", "medium"):
            return 1
        if s in ("low",):
            return 0
    # Mặc định thận trọng
    return 0

def _coerce_conf(v: object) -> str:
    s = str(v).strip().lower()
    if "high" in s:
        return "high"
    if "med" in s or "medium" in s:
        return "med"
    return "low"

def normalize_output(raw: Dict[str, Any], text: str, qid: str) -> Dict[str, Any]:
    # labels: đảm bảo đủ keys & ép 0/1 bằng _coerce_label
    raw_labels = raw.get("labels", {}) or {}
    labels = {}
    for e in ENTITIES:
        labels[e] = _coerce_label(raw_labels.get(e, 0))

    # confidence: chuẩn hoá về low/med/high bằng _coerce_conf
    raw_conf = raw.get("confidence", {}) or {}
    conf = {}
    for e in ENTITIES:
        conf[e] = _coerce_conf(raw_conf.get(e, "low"))

    # reasoning: ép về chuỗi an toàn
    r = raw.get("reasoning", "")
    if isinstance(r, list):
        r = " | ".join([json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else str(x) for x in r])
    elif isinstance(r, dict):
        r = json.dumps(r, ensure_ascii=False)
    else:
        r = str(r)

    return {"id": qid, "text": text, "labels": labels, "confidence": conf, "reasoning": r}

def aggregate_personas(outputs, qid, text):
    if not outputs:
        return {"id": qid, "text": text,
                "labels": {e: 0 for e in ENTITIES},
                "confidence": {e: "low" for e in ENTITIES},
                "reasoning": ""}
    votes = {e: 0 for e in ENTITIES}
    conf_sum = {e: 0 for e in ENTITIES}
    reason_list = []
    for o in outputs:
        for e in ENTITIES:
            votes[e] += int(o["labels"].get(e, 0))
            conf_sum[e] += {"low": 1, "med": 2, "high": 3}.get(o["confidence"].get(e, "low"), 1)
        r = o.get("reasoning", "")
        if isinstance(r, list):
            reason_list.extend([json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else str(x) for x in r])
        elif isinstance(r, dict):
            reason_list.append(json.dumps(r, ensure_ascii=False))
        elif r:
            reason_list.append(str(r))
    n = len(outputs)
    labels = {e: 1 if votes[e] > n/2 else 0 for e in ENTITIES}
    conf = {e: ("high" if round(conf_sum[e]/n) >= 3 else ("med" if round(conf_sum[e]/n) == 2 else "low"))
            for e in ENTITIES}
    reasoning = " | ".join(reason_list[:3])
    return {"id": qid, "text": text, "labels": labels, "confidence": conf, "reasoning": reasoning}

# ---------------- Providers ----------------------------------------------------
def call_groq_json(messages, model: str, temperature=0.0) -> str:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY chưa được thiết lập")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content

def call_groq_fallback(messages, model: str, temperature=0.0) -> str:
    # gọi lại KHÔNG ép JSON, và yêu cầu bọc trong <json>...</json>
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        # bỏ response_format để model tự do in JSON
    )
    return resp.choices[0].message.content

def label_with_groq(text: str, personas: List[str], model: str, rate: float) -> Dict[str, Any]:
    outs = []
    for persona in personas:
        msgs = build_messages(text, persona, with_xml_hint=False)
        try:
            raw = call_groq_json(msgs, model=model)
            data = safe_json_parse(raw)
        except Exception:
            # fallback 1: thêm XML hint
            msgs2 = build_messages(text, persona, with_xml_hint=True)
            try:
                raw = call_groq_fallback(msgs2, model=model)
                data = safe_json_parse(raw)
            except Exception:
                # fallback 2: bỏ persona + xml
                msgs3 = build_messages(text, None, with_xml_hint=True)
                raw = call_groq_fallback(msgs3, model=model)
                data = safe_json_parse(raw)
        outs.append(data)
        time.sleep(rate)
    return outs

# ---------------- Dummy rule-based fallback -----------------------------------
RULES = {
    "Genre": ["horror","comedy","romance","sci-fi","action","drama","thriller","fantasy","documentary","animation","crime"],
    "CastAndCrew": ["tom hanks","leonardo dicaprio","scarlett johansson","keanu reeves","nolan","tarantino","spielberg","scorsese"],
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
def dummy_label(query: str) -> Dict[str, Any]:
    ql = query.lower()
    labels = {e:0 for e in ENTITIES}; conf = {e:"low" for e in ENTITIES}
    if any(w in ql for w in ["movie","movies","film","films"]): labels["IntentMovie"]=1; conf["IntentMovie"]="med"
    if any(w in ql for w in ["tv","series","show","shows"]): labels["IntentTvSeries"]=1; conf["IntentTvSeries"]="med"
    if any(w in ql for w in ["popular","trending","top charts","most-watched"]): labels["Popularity"]=1
    if any(w in ql for w in ["new ","newly","recent","latest","released in "]): labels["Recency"]=1
    for e, rule in RULES.items():
        if isinstance(rule, str):
            if re.search(rule, ql): labels[e]=1
        else:
            if any(tok in ql for tok in rule): labels[e]=1
    reasoning = "; ".join([k for k,v in labels.items() if v==1])
    return {"labels":labels,"confidence":conf,"reasoning":reasoning}

# ---------------- IO helpers: resume + stream ---------------------------------
def load_done_ids(out_path: str) -> set:
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "id" in obj: done.add(obj["id"])
                except Exception:
                    continue
    return done

def writer(open_path: str):
    f = open(open_path, "a", encoding="utf-8")
    def write(rec: Dict[str, Any]):
        # Chuyển JSON sang UTF-8 an toàn, bỏ qua surrogate lỗi
        safe_line = json.dumps(rec, ensure_ascii=False).encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        f.write(safe_line + "\n")
        f.flush()
    return f, write

# ---------------- Split helper -------------------------------------------------
def write_splits(items, out_dir):
    random.seed(7); idx = list(range(len(items))); random.shuffle(idx)
    n = len(items); n_train = int(0.7*n); n_dev = int(0.1*n)
    parts = {"train":[items[i] for i in idx[:n_train]],
             "dev":[items[i] for i in idx[n_train:n_train+n_dev]],
             "test":[items[i] for i in idx[n_train+n_dev:]]}
    os.makedirs(os.path.join(out_dir,"splits"), exist_ok=True)
    for name, arr in parts.items():
        with open(os.path.join(out_dir,"splits",f"{name}.jsonl"),"w",encoding="utf-8") as f:
            for r in arr: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    return {k: len(v) for k,v in parts.items()}

# ---------------- Main ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="data/synthetic_queries.jsonl")
    ap.add_argument("--out", dest="out", type=str, default="data/weak_labels.jsonl")
    ap.add_argument("--provider", type=str, choices=["groq","openai","dummy"], default="groq")
    ap.add_argument("--model", type=str, default="llama-3.1-8b-instant",
                    help="groq: llama-3.1-8b-instant | mixtral-8x7b-32768 | llama3-8b-8192 ; openai: gpt-4o-mini ...")
    ap.add_argument("--personas", type=str, default="Merchandiser,Movie Critic,Horror Aficionado")
    ap.add_argument("--rate", type=float, default=0.4, help="sleep seconds between API calls (groq/openai)")
    ap.add_argument("--limit", type=int, default=0, help="label only first N (0 = all)")
    ap.add_argument("--stream", type=int, default=1, help="1 = ghi dần từng dòng khi chạy")
    ap.add_argument("--resume", type=int, default=1, help="1 = bỏ qua các id đã có trong file out")
    args = ap.parse_args()

    # Load queries
    all_queries = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    if args.limit > 0:
        all_queries = all_queries[:args.limit]

    # Resume: bỏ qua id đã làm
    done_ids = load_done_ids(args.out) if args.resume else set()
    queries = [q for q in all_queries if q["id"] not in done_ids]

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    personas = [p.strip() for p in args.personas.split(",") if p.strip()]

    # writer (stream)
    f_out, write_one = writer(args.out) if args.stream else (None, None)
    buffered = []  # dùng cho split sau cùng

    try:
        if args.provider == "groq":
            pbar = tqdm(queries, desc="GROQ labeling")
            for row in pbar:
                qid, text = row["id"], row["text"]
                try:
                    persona_jsons = label_with_groq(text, personas, model=args.model, rate=args.rate)
                except Exception:
                    # nếu tất cả hỏng → dummy
                    persona_jsons = []
                outs = []
                for data in persona_jsons:
                    if not isinstance(data, dict) or not data:
                        # nếu sample lỗi → dummy cho persona đó
                        data = dummy_label(text)
                        data.update({"id": qid, "text": text})
                    norm = normalize_output(data, text, qid)
                    outs.append(norm)
                agg = aggregate_personas(outs, qid, text) if len(outs) > 1 else (outs[0] if outs else normalize_output(dummy_label(text), text, qid))
                buffered.append(agg)
                if write_one: write_one(agg)

        elif args.provider == "openai":
            from openai import OpenAI
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY chưa được thiết lập")
            client = OpenAI()
            pbar = tqdm(queries, desc="OpenAI labeling")
            for row in pbar:
                qid, text = row["id"], row["text"]; outs=[]
                for persona in personas:
                    msgs = build_messages(text, persona)
                    try:
                        resp = client.chat.completions.create(
                            model=args.model, messages=msgs, temperature=0.0, response_format={"type":"json_object"}
                        )
                        data = safe_json_parse(resp.choices[0].message.content)
                    except Exception:
                        msgs2 = build_messages(text, persona, with_xml_hint=True)
                        resp = client.chat.completions.create(model=args.model, messages=msgs2, temperature=0.0)
                        data = safe_json_parse(resp.choices[0].message.content)
                    outs.append(normalize_output(data, text, qid)); time.sleep(args.rate)
                agg = aggregate_personas(outs, qid, text) if len(outs)>1 else outs[0]
                buffered.append(agg)
                if write_one: write_one(agg)

        else:  # dummy
            pbar = tqdm(queries, desc="Rule-based labeling")
            for row in pbar:
                qid, text = row["id"], row["text"]
                agg = dummy_label(text); agg.update({"id": qid, "text": text})
                buffered.append(agg)
                if write_one: write_one(agg)

    finally:
        if f_out: f_out.close()

    # Nếu không stream, ghi batch 1 lần
    if not args.stream:
        with open(args.out, "a", encoding="utf-8") as f:
            for r in buffered:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Chia tập từ toàn bộ file (đảm bảo resume cũng hoạt động)
    all_items = [json.loads(l) for l in open(args.out, encoding="utf-8")]
    stats = write_splits(all_items, out_dir)
    print(f"✅ Saved weak labels to {args.out}")
    print(f"✅ Splits -> train={stats['train']}, dev={stats['dev']}, test={stats['test']}")
    print(f"Entities: {', '.join(ENTITIES)}")

if __name__ == "__main__":
    main()
