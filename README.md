# LLM-based Weak Supervision for Query Intent in Video Search

## ✨ Điểm nổi bật

- **Weak supervision bằng LLM** (CoT + ICL + Confidence; optional multi-persona).
- **Multi-label NLU** với **BERT-base** (BCEWithLogits, sigmoid từng nhãn).
- **Tối ưu ngưỡng per-label** trên dev (không mặc định 0.5) + **PR-curve** từng nhãn.
- **API FastAPI** cho inference; **Flask mini-app** giao diện dark, có **Explain** .
- Chạy **được** trên Windows (CPU/GPU), phù hợp **sinh viên/đồ án** thời gian ngắn.

---

## 📁 Cấu trúc thư mục

```
.
├── data/
│   ├── queries.jsonl              # truy vấn synthetic
│   ├── weak_labels.jsonl          # nhãn yếu (labels/confidence/reasoning)
│   └── splits/
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── test.jsonl
├── models/
│   └── bert_intent.pt4            # checkpoint BERT
├── report/
│   ├── figures/                   # PR-curves per-label
│   └── metrics/
│       ├── model_agg.csv
│       ├── model_per_label.csv
│       ├── baseline_agg.csv
│       ├── baseline_per_label.csv
│       └── thresholds.json        # ngưỡng tối ưu per-label
├── src/
│   ├── day1_synth_queries.py
│   ├── day2_weak_labeling.py
│   ├── train_bert.py
│   ├── eval_bert.py
│   ├── api.py                     # FastAPI
│   └── demo_flask.py              # Flask UI (Day 5)
├── requirements.txt
└── README.md
```

---

## 🔧 Yêu cầu

- Python 3.10+
- PyTorch (CPU/GPU tuỳ máy)
- Transformers (HuggingFace), FastAPI, Uvicorn, Flask, matplotlib, scikit-learn

Cài đặt:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🚀 Chạy nhanh (Quickstart)

### 0) (Tuỳ chọn) Tải checkpoint & thresholds

Đặt các file vào đúng chỗ

```
models/bert_intent.pt4
report/metrics/thresholds.json
```

### 1) API (FastAPI)

```bash
set CKPT_PATH=models\bert_intent.pt4
set THRESHOLDS_PATH=report\metrics\thresholds.json
set WEAK_LABELS_PATH=data\weak_labels.jsonl
uvicorn src.api:app --reload --port 8000
```

Gọi thử:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query":"best horror movies from the 90s on Netflix","explain":true,"explain_source":"rule"}'
```

### 2) Demo Web (Flask)

```bash
set API_URL=http://127.0.0.1:8000/predict
python src/demo_flask.py
# Mở http://127.0.0.1:5000
```

---

## 📚 Quy trình chi tiết

### Sinh dữ liệu truy vấn

- **Mục tiêu**: tạo \~**2k–3k** truy vấn tiếng Anh theo **22 thực thể** (IntentMovie/TV, Genre, CastAndCrew, StreamingService, ReleaseYear/Decade, AudioLanguage, …).
- **Cách làm**: dùng **template**/lexicon + (tuỳ chọn) **persona** để đa dạng hoá câu.
- **Kết quả**: `data/queries.jsonl`.

Chạy:

```bash
python src/day1_synth_queries.py \
  --out data/queries.jsonl --n 2500
```

### Gán nhãn yếu bằng LLM

- **Prompt khung**: CoT + ICL + Confidence; xuất **JSON** theo schema:

  ```json
  {
    "id": "<qid>",
    "text": "<query>",
    "labels": { "Genre": 1, "CastAndCrew": 0, "...": 0 },
    "confidence": { "Genre": "high", "...": "low" },
    "reasoning": "... hoặc list các câu ngắn ..."
  }
  ```

- (Tuỳ chọn) chạy **2–3 persona** → **majority vote** per-label.
- Chia **train/dev/test = 70/10/20** vào `data/splits/`.

Chạy:

```bash
# ví dụ dùng provider GROQ / OpenAI (tuỳ script của bạn)
python src/day2_weak_labeling.py \
  --in data/queries.jsonl \
  --out data/weak_labels.jsonl \
  --personas "Movie Critic,Merchandiser,Horror Aficionado"
```

### Huấn luyện BERT đa nhãn

- **Backbone**: `bert-base-uncased`, max_len=128
- **Loss**: `BCEWithLogitsLoss`; **Optimizer**: `AdamW`, `lr=1e-5`
- **batch=32**, **epochs=3–5**
- **Checkpoint**: `models/bert_intent.pt4`

Chạy:

```bash
python src/train_bert.py \
  --epochs 4 --batch_size 32 --max_len 128 --lr 1e-5 \
  --ckpt models/bert_intent.pt4
```

> **Windows tip**: nếu gặp lỗi `token_type_ids`, trong `BertMultiLabel.forward()` bỏ qua `token_type_ids` hoặc set `token_type_ids=None`. `num_workers=0` để tránh treo DataLoader.

### Đánh giá & tối ưu ngưỡng

- **Mục tiêu**: tìm **threshold per-label** tối ưu **F1/Fβ** trên **dev**, vẽ **PR-curve** từng nhãn.
- Xuất:

  - Hình PR: `report/figures/pr_<Label>.png`
  - Bảng tổng hợp: `report/metrics/model_agg.csv`, `*per_label.csv`
  - Ngưỡng: `report/metrics/thresholds.json`

Chạy:

```bash
python src/eval_bert.py --ckpt models/bert_intent.pt4
# hoặc ưu tiên precision: --beta 0.5
# hoặc đặt "precision floor": --mode precfloor --precision_floor 0.8
```

### API & Frontend

- **FastAPI** (`src/api.py`): `/predict` trả `probs`, `labels`, `over_threshold`, `thresholds`, `topk`, `explanations`.
- **Explain**:

  - **rule**: giải thích theo từ khoá/regex trong câu nhập.
  - **weak**: ví dụ rút từ `weak_labels.jsonl` (đã lọc chuỗi chỉ liệt kê nhãn).
  - **both**: kết hợp cả hai.

- **Flask UI** (`src/demo_flask.py`): dark theme, progress bar, badge true/false, chip Top-K; chọn **Explain source**.

---

## 📊 Kết quả mẫu

> **Lưu ý**: Các con số thay đổi theo dữ liệu/seed. Vui lòng chạy `eval_bert.py` để xuất số liệu mới nhất rồi cập nhật bảng này.

| Mô hình                              | Micro-F1 | Macro-F1 | Precision | Recall |
| ------------------------------------ | -------: | -------: | --------: | -----: |
| **BERT (threshold per-label, test)** |    0.895 |    0.584 |     0.867 |  0.925 |
| **Baseline (lexical, test)**         |    0.995 |    0.830 |     0.989 |  1.000 |

> Baseline cao bất thường nếu test trùng cơ chế sinh luật (leakage). Khi dùng LLM gán nhãn thật sự, baseline sẽ giảm và mô hình có cơ hội vượt ở nhiều nhãn khó.

---

## ⚙️ Cấu hình & biến môi trường

- `CKPT_PATH` – đường dẫn checkpoint BERT (`models/bert_intent.pt4`)
- `THRESHOLDS_PATH` – file ngưỡng per-label (`report/metrics/thresholds.json`)
- `WEAK_LABELS_PATH` – weak labels có trường `reasoning` (để Explain-weak)
- `MAX_LEN` – max token length tokenizer (mặc định 128)
- `MAX_REASONS_PER_LABEL` – giới hạn số câu reasoning nạp vào RAM (API)
- `MAX_LINES_PER_LABEL_OUT` – tối đa số dòng explain hiển thị mỗi nhãn

---

## 🧪 API Spec

- `GET /health` → `{ok, model, labels}`
- `POST /predict`

  ```json
  {
    "query": "string",
    "explain": true,
    "explain_source": "rule|weak|both",
    "topk": 5,
    "max_len": 128
  }
  ```

  **Response rút gọn**:

  ```json
  {
    "query": "...",
    "probs": {"Genre": 0.81, "...": 0.12},
    "labels": {"Genre": true, "...": false},
    "over_threshold": ["Genre", "..."],
    "thresholds": {"Genre": 0.35, "...": 0.40},
    "topk": [{"label":"Genre","prob":0.81}, ...],
    "explanations": {"Genre": ["Genre keyword(s) detected: comedy."], ...}
  }
  ```

---

## 🧩 Troubleshooting (đặc biệt trên Windows)

- **`token_type_ids` unexpected** → trong `BertMultiLabel.forward()` bỏ tham số này, hoặc khi gọi model bỏ key `token_type_ids`.
- **`pin_memory` warning** → an toàn nếu không dùng GPU.
- **Dataloader treo** → `num_workers=0`.
- **Huggingface cache/symlink warning** → có thể bật Developer Mode Windows hoặc bỏ qua (không ảnh hưởng chức năng).
- **Explain không hiện** → kiểm tra `WEAK_LABELS_PATH` có trường `reasoning` hay không; trong API đã có fallback **rule-based**.

---

## 📜 Giấy phép

MIT (hoặc theo yêu cầu của bạn). Đổi nội dung file `LICENSE` nếu cần.

---

## 📚 Trích dẫn

Nếu dùng dự án này trong học thuật, vui lòng trích:

- F. Javadi et al., “LLM-based Weak Supervision Framework for Query Intent Classification in Video Search,” _arXiv preprint_ arXiv:2409.08931, Sep. 2024.

```bibtex
@article{javadi2024llmweak,
  title   = {LLM-based Weak Supervision Framework for Query Intent Classification in Video Search},
  author  = {Javadi, Farnoosh and Gampa, Phanideep and Woo, Alyssa and Geng, Xingxing and Zhang, Hang and Sepulveda, Jose and Bayar, Belhassen and Wang, Fei},
  journal = {arXiv preprint arXiv:2409.08931},
  year    = {2024},
  month   = {September}
}
```

---

## 🙏 Ghi nhận

- Cảm ơn tác giả bài báo tham khảo và cộng đồng mở nguồn (PyTorch, Transformers, FastAPI, Flask).
- Dự án hướng tới mục đích **học thuật/giáo dục**; không dùng dữ liệu người dùng thật.
