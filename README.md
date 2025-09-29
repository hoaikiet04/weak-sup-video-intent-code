```markdown
# LLM-based Weak Supervision for Query Intent in Video Search

> Multi‑label NLU + LLM‑based weak supervision cho truy vấn tìm kiếm video.  
> Demo: FastAPI (API) + Flask (UI), có Explain ngắn theo từng nhãn.

## ✨ Điểm nổi bật

- **Weak supervision bằng LLM** (CoT + ICL + Confidence; hỗ trợ **GROQ**/**OpenAI**).
- **BERT-base** đa nhãn (BCEWithLogits + sigmoid), **tối ưu ngưỡng per‑label** trên dev.
- Xuất **PR‑curve** từng nhãn; **API FastAPI** + **Flask UI** dark mode, có **Explain**.
- Repo chạy được trên Windows/macOS/Linux (CPU/GPU).

---

## 📁 Cấu trúc thư mục
```

.
├── .env
├── .gitignore
├── README.md
├── data/
│ ├── queries_raw.jsonl
│ ├── synthetic_queries.jsonl
│ ├── weak_labels.jsonl
│ └── splits/
│ ├── train.jsonl
│ ├── dev.jsonl
│ └── test.jsonl
├── models/ # (trống trên repo)
├── report/
│ ├── figures/ Precision–Recall curve, biểu đồ so sánh ngưỡng và kết quả huấn luyện.  
│ │  
│ └── metrics/ Lưu các số liệu đánh giá chi tiết và các file JSON ngưỡng tối ưu per-label.
│  
└── src/
├── api.py # FastAPI: /predict
├── demo_flask.py # Flask UI
├── eval_bert.py # Đánh giá + PR-curve + tối ưu ngưỡng
├── gen_queries.py # Sinh truy vấn synthetic
├── train_bert.py # Huấn luyện BERT đa nhãn (+ ENTITIES)
└── weak_label_llm.py # Gán nhãn yếu bằng LLM (GROQ)
└── **pycache**/ # file biên dịch tạm của Python

```

**Chú thích nhanh**
- `data/queries_raw.jsonl`: truy vấn thô; `data/synthetic_queries.jsonl`: truy vấn sinh tự động.
- `data/weak_labels.jsonl`: nhãn yếu từ LLM (có thể kèm reasoning).
- `report/metrics/*.csv` + `thresholds.json`: số liệu và ngưỡng tối ưu per‑label.
- `models/`: nơi đặt checkpoint `.pt4` (không theo git).

---

## 📦 Checkpoint model (Google Drive)
Do file lớn, checkpoint **không commit lên repo**. Tải tại đây rồi **đặt vào `models/`**:

**➡️ [Google Drive – bert_intent.pt4](https://drive.google.com/file/d/1jeLlZy70Z0az1lF8uLxJclo0KotFCexx/view?usp=sharing)**

Sau khi tải xong:
```

models/
└── bert_intent.pt4

````
> Nếu tên file tải về khác, vui lòng **đổi tên về `bert_intent.pt4`** (hoặc cập nhật biến môi trường `CKPT_PATH` cho khớp).

---

## 🔧 Yêu cầu & Cài đặt

### 1) Python & thư viện
- Python **3.10+**
- PyTorch (CPU/GPU tuỳ máy), Transformers, FastAPI, Uvicorn, Flask
- scikit‑learn, matplotlib, numpy, tqdm, pydantic, requests
- **GROQ** và/hoặc **OpenAI** SDK; **python‑dotenv** (đọc `.env`)

Cài nhanh (gợi ý):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install torch transformers fastapi uvicorn flask requests \
            scikit-learn matplotlib numpy tqdm pydantic python-dotenv \
            groq openai
````

### 2) Khóa API (.env)

Tạo file `.env` ở thư mục gốc (nếu chưa có) để chạy gán nhãn yếu bằng LLM:

```
GROQ_API_KEY=xxxxx      # nếu dùng GROQ
OPENAI_API_KEY=xxxxx    # nếu dùng OpenAI
```

---

## 🚀 Chạy nhanh

### A) API (FastAPI)

Thiết lập biến môi trường và chạy server:

**Windows (cmd)**

```cmd
set CKPT_PATH=models\bert_intent.pt4
set THRESHOLDS_PATH=report\metrics\thresholds.json
set WEAK_LABELS_PATH=data\weak_labels.jsonl
uvicorn src.api:app --reload --port 8000
```

**macOS/Linux (bash)**

```bash
export CKPT_PATH=models/bert_intent.pt4
export THRESHOLDS_PATH=report/metrics/thresholds.json
export WEAK_LABELS_PATH=data/weak_labels.jsonl
uvicorn src.api:app --reload --port 8000
```

Gọi thử:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query":"best horror movies from the 90s on Netflix","explain":true,"explain_source":"rule"}'
```

### B) Demo Web (Flask)

```bash
# Windows
set API_URL=http://127.0.0.1:8000/predict
# macOS/Linux
export API_URL=http://127.0.0.1:8000/predict

python src/demo_flask.py
# Mở http://127.0.0.1:5000
```

---

## 🧪 Quy trình thao tác

1. **Sinh truy vấn**:

```bash
python src/gen_queries.py --out data/synthetic_queries.jsonl --n 2500
```

2. **Gán nhãn yếu bằng LLM** (GROQ/OpenAI):

```bash
python src/weak_label_llm.py \
  --in data/synthetic_queries.jsonl \
  --out data/weak_labels.jsonl \
  --provider groq \
  --personas "Movie Critic,Merchandiser,Horror Aficionado"
```

3. **Huấn luyện BERT đa nhãn**:

```bash
python src/train_bert.py \
  --epochs 4 --batch_size 32 --max_len 128 --lr 1e-5 \
  --ckpt models/bert_intent.pt4
```

4. **Đánh giá & vẽ PR‑curve** (tạo `thresholds.json`, `report/figures/*.png`, `report/metrics/*.csv`):

```bash
python src/eval_bert.py --ckpt models/bert_intent.pt4
# tuỳ chọn: --beta 0.5 (ưu precision) hoặc --mode precfloor --precision_floor 0.8
```

---

## 📊 Kết quả & file đầu ra

- PR‑curve: `report/figures/pr_<Label>.png`
- Ngưỡng tối ưu per‑label: `report/metrics/thresholds.json`
- Tổng hợp chỉ số: `report/metrics/model_agg.csv`, `model_per_label.csv`
- Baseline lexical (tham khảo): `report/metrics/baseline_*.csv`

> Lưu ý: baseline lexical có rủi ro **data leakage** nếu trùng cơ chế sinh/đánh nhãn; nên dùng **weak labels từ LLM** + **mini human‑gold** để đánh giá khách quan.

---

## 🧩 Troubleshooting

- `FileNotFoundError: models/bert_intent.pt4` → chưa tải/đặt checkpoint, xem mục **Checkpoint model** ở trên.
- Treo DataLoader trên Windows → đặt `num_workers=0`.
- `token_type_ids unexpected` → trong `BertMultiLabel.forward()` bỏ tham số này hoặc không truyền `token_type_ids`.
- Explain không hiện → kiểm tra `WEAK_LABELS_PATH` có trường `reasoning`; API có fallback **rule‑based**.

---

## 📚 Tham khảo

- F. Javadi et al., “LLM-based Weak Supervision Framework for Query Intent Classification in Video Search,” arXiv:2409.08931 (2024).

```

```
