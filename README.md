# 🎯 LLM-based Weak Supervision for Video Query Intent Classification

> Multi-label NLU + LLM-based weak supervision cho truy vấn tìm kiếm video.  
> **Demo**: FastAPI (API) + Flask (UI) + Explain reasoning theo từng nhãn.

---

## ✨ Điểm nổi bật
- 🧠 **Weak supervision bằng LLM** (CoT + ICL + Confidence; hỗ trợ **GROQ**/**OpenAI**)
- 🤖 **BERT-base** đa nhãn (BCEWithLogits + sigmoid), tối ưu ngưỡng *per-label* trên dev
- 📊 Xuất **PR-curve** từng nhãn + tổng hợp chỉ số vào `report/metrics/`
- 🌑 **Flask UI** dark mode + explain reasoning
- 💻 Chạy được trên Windows / macOS / Linux (CPU/GPU)

---

## 📁 Cấu trúc thư mục

```

.
├── .env
├── .gitignore
├── README.md
├── data/
│   ├── queries_raw.jsonl
│   ├── synthetic_queries.jsonl
│   ├── weak_labels.jsonl
│   └── splits/
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── test.jsonl
├── models/                     # (trống trên repo – đặt checkpoint .pt4 ở đây)
├── report/
│   ├── figures/                # PR-curve per-label (png)
│   └── metrics/
│       ├── baseline_agg.csv
│       ├── baseline_per_label.csv
│       ├── model_agg.csv
│       ├── model_per_label.csv
│       └── thresholds.json     # ngưỡng tối ưu per-label (DEV)
└── src/
├── api.py                  # FastAPI: /predict
├── demo_flask.py           # Flask UI
├── eval_bert.py            # Đánh giá + PR-curve + tối ưu ngưỡng
├── gen_queries.py          # Sinh truy vấn synthetic
├── train_bert.py           # Huấn luyện BERT đa nhãn
└── weak_label_llm.py       # Gán nhãn yếu bằng LLM (GROQ/OpenAI)

```

**Ghi chú**
- `models/` chứa checkpoint `.pt4` (không push lên repo)
- `report/metrics/*.json|csv` lưu kết quả đánh giá & ngưỡng tối ưu
- `data/` chứa dữ liệu thô, synthetic và weak labels

---

## 📦 Checkpoint model

File lớn nên **không commit** lên repo. Tải checkpoint tại:

**➡️ [Google Drive – bert_intent.pt4](https://drive.google.com/file/d/1jeLlZy70Z0az1lF8uLxJclo0KotFCexx/view?usp=sharing)**

Sau khi tải, đặt vào:

```

models/
└── bert_intent.pt4

````

> Nếu tên file khác, hãy đổi về `bert_intent.pt4` hoặc cập nhật biến môi trường `CKPT_PATH`.

---

## 🔧 Cài đặt

### Tạo môi trường & cài thư viện

```bash
python -m venv .venv
# Windows (cmd)
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
````

**Cách A – cài trực tiếp**

```bash
pip install torch transformers fastapi uvicorn flask \
            scikit-learn numpy pandas matplotlib tqdm \
            python-dotenv groq openai
```

**Cách B – dùng requirements.txt (tuỳ chọn)**
Tạo `requirements.txt` với nội dung:

```
torch
transformers
fastapi
uvicorn
flask
scikit-learn
numpy
pandas
matplotlib
tqdm
python-dotenv
groq
openai
```

Sau đó:

```bash
pip install -r requirements.txt
```

### Thiết lập API key (nếu gán nhãn yếu bằng LLM)

Tạo file `.env` ở thư mục gốc:

```
GROQ_API_KEY=xxxxx
OPENAI_API_KEY=xxxxx
```

---

## 🚀 Chạy nhanh

### 1) API (FastAPI)

**Windows (cmd)**

```cmd
set CKPT_PATH=models\bert_intent.pt4
set THRESHOLDS_PATH=report\metrics\thresholds.json
set WEAK_LABELS_PATH=data\weak_labels.jsonl
uvicorn src.api:app --reload --port 8000
```

**Windows (PowerShell)**

```powershell
$env:CKPT_PATH="models/bert_intent.pt4"
$env:THRESHOLDS_PATH="report/metrics/thresholds.json"
$env:WEAK_LABELS_PATH="data/weak_labels.jsonl"
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

### 2) Demo UI (Flask)

**Windows (cmd)**

```cmd
set API_URL=http://127.0.0.1:8000/predict
python src\demo_flask.py
```

**macOS/Linux (bash)**

```bash
export API_URL=http://127.0.0.1:8000/predict
python src/demo_flask.py
```

Mở trình duyệt: `http://127.0.0.1:5000`

---

## 🧪 Quy trình huấn luyện

**(1) Sinh synthetic queries**

```bash
python src/gen_queries.py --out data/synthetic_queries.jsonl --n 2500
```

**(2) Gán nhãn yếu bằng LLM**

```bash
python src/weak_label_llm.py \
  --in data/synthetic_queries.jsonl \
  --out data/weak_labels.jsonl \
  --provider groq \
  --personas "Movie Critic,Merchandiser,Horror Aficionado"
```

**(3) Huấn luyện BERT đa nhãn**

```bash
python src/train_bert.py \
  --epochs 4 --batch_size 32 --max_len 128 --lr 1e-5 \
  --ckpt models/bert_intent.pt4
```

**(4) Đánh giá + PR‑curve + tối ưu ngưỡng**

```bash
python src/eval_bert.py --ckpt models/bert_intent.pt4
# tuỳ chọn: --beta 0.5 (ưu precision) hoặc --mode precfloor --precision_floor 0.8
```

**Đầu ra**

* PR‑curve: `report/figures/pr_<Label>.png`
* Ngưỡng tối ưu: `report/metrics/thresholds.json`
* Chỉ số tổng hợp: `report/metrics/model_agg.csv`, `model_per_label.csv`
* (Tham khảo) Baseline lexical: `report/metrics/baseline_*.csv`

---

## ⚙️ Biến môi trường chính

* `CKPT_PATH` – đường dẫn checkpoint BERT (mặc định `models/bert_intent.pt4`)
* `THRESHOLDS_PATH` – file ngưỡng per‑label (mặc định `report/metrics/thresholds.json`)
* `WEAK_LABELS_PATH` – weak labels có trường `reasoning` (để Explain‑weak)
* `MAX_LEN` – max token length (mặc định 128)
* `MAX_REASONS_PER_LABEL` – số câu reasoning nạp per‑label (API)
* `MAX_LINES_PER_LABEL_OUT` – số dòng explain hiển thị mỗi nhãn (API)

---

## 🧩 Troubleshooting

* `FileNotFoundError: models/bert_intent.pt4` → chưa tải/đặt checkpoint (xem mục Checkpoint).
* `TypeError: forward() got an unexpected keyword argument 'token_type_ids'` → trong `api.py` đã có fallback; nếu tự gọi model, bỏ key `token_type_ids`.
* DataLoader treo trên Windows → đặt `num_workers=0`.
* Explain không hiện → kiểm tra `WEAK_LABELS_PATH` có trường `reasoning`; API có **fallback rule‑based**.
* Hugging Face cache/symlink warning trên Windows → có thể bỏ qua hoặc bật Developer Mode.

---

## 📚 Tham khảo

* F. Javadi et al., “LLM-based Weak Supervision Framework for Query Intent Classification in Video Search,” arXiv:2409.08931 (2024).
