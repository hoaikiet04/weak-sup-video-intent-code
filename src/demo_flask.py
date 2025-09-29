#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, render_template_string
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

app = Flask(__name__)

TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Video Query Intent - Demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --bg: #0b1020;
  --card: #11172b;
  --muted: #9fb0d0;
  --text: #eaf0ff;
  --accent: #6ea8ff;
  --accent-2: #9b7bff;
  --ok-bg:#0e4d2a;   --ok:#b7ffd3;
  --no-bg:#56202a;   --no:#ffc7d1;
  --chip:#1c2440;
  --table-border:#243154;
  --progress:#28d08f;
  --progress-2:#9b7bff;
}
*{box-sizing:border-box}
html,body{background:var(--bg);color:var(--text);font-family:Inter,system-ui,Segoe UI,Arial,sans-serif;margin:0}
.wrap{max-width:1000px;margin:40px auto;padding:0 20px}
.header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:14px}
h1{font-size:28px;margin:0;background:linear-gradient(90deg,var(--accent),var(--accent-2));
  -webkit-background-clip:text;background-clip:text;color:transparent}
.desc{color:var(--muted);margin-bottom:18px}
.card{background:var(--card);border:1px solid #1a2340;border-radius:14px;padding:18px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
.form-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input[type=text]{flex:1;min-width:320px;border:1px solid #233055;background:#0e1428;color:var(--text);
  border-radius:10px;padding:12px 14px;font-size:16px;outline:none}
select{border:1px solid #233055;background:#0e1428;color:var(--text);border-radius:10px;padding:10px 12px}
.check{display:flex;align-items:center;gap:8px;color:var(--muted)}
.btn{background:linear-gradient(90deg,var(--accent),var(--accent-2));color:#0a0f21;border:none;border-radius:10px;
  padding:12px 16px;font-weight:700;cursor:pointer}
.btn:hover{filter:brightness(1.1)}
.kbd{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#0b132a;border:1px solid #1f2b4d;color:#cfe3ff;
  border-radius:6px;padding:2px 6px}
.mt16{margin-top:16px}
.mt24{margin-top:24px}
.table{width:100%;border-collapse:separate;border-spacing:0;border:1px solid var(--table-border);border-radius:12px;overflow:hidden}
.table th,.table td{padding:10px 12px;border-bottom:1px solid var(--table-border)}
.table thead th{background:#0e162e;color:#bcd0ff;font-weight:600;text-align:left}
.table tbody tr:last-child td{border-bottom:none}
.badge{display:inline-block;padding:4px 8px;border-radius:999px;font-size:12px;font-weight:700}
.badge.ok{background:var(--ok-bg);color:var(--ok)}
.badge.no{background:var(--no-bg);color:var(--no)}
.row-yes{background:rgba(40,208,143,0.06)}
.bar{position:relative;height:10px;background:#121a34;border-radius:999px;overflow:hidden;border:1px solid #243154}
.bar>span{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,var(--progress),var(--progress-2));
  width:0%}
.small{color:var(--muted);font-size:13px}
.section-title{font-size:18px;margin:18px 0 8px 0}
.explain-box{background:#0c1430;border:1px dashed #2b3a66;border-radius:12px;padding:12px;margin:10px 0 22px}
.explain-item{margin:6px 0}
.code{background:#0b132a;border:1px solid #1f2b4d;border-radius:8px;padding:10px}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{background:var(--chip);border:1px solid #26345e;color:#cfe3ff;border-radius:999px;padding:6px 10px;font-size:12px}
@media (max-width:720px){ .form-row{flex-direction:column;align-items:stretch} }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>Video Query Intent - Demo</h1>
  </div>
  <p class="desc">Nhập truy vấn → mô hình dự đoán xác suất cho từng thực thể; nhãn vượt ngưỡng sẽ được tô nổi.
    <br/> Bật <span class="kbd">Explain</span> để xem lời giải thích.</p>

  <div class="card">
    <form method="POST" class="form-row">
      <input type="text" name="q" placeholder="e.g., best horror movies from the 90s on Netflix" value="{{q or ''}}">
      <label class="check">
        <input type="checkbox" name="explain" {% if explain %}checked{% endif %}> Explain
      </label>
      <button class="btn" type="submit">Predict</button>
    </form>

    {% if resp %}
      <div class="mt16 small">Kết quả cho: <span class="kbd">{{ resp.query }}</span></div>

      {% if resp.topk %}
      <div class="mt16 chips">
        {% for item in resp.topk %}
          <span class="chip">{{ item.label }}: {{ '%.3f'|format(item.prob) }}</span>
        {% endfor %}
      </div>
      {% endif %}

      <table class="table mt16">
        <thead>
          <tr><th>Entity</th><th style="width:36%">Prob</th><th>Threshold</th><th>Above?</th></tr>
        </thead>
        <tbody>
        {% for row in rows %}
          <tr class="{% if row.over %}row-yes{% endif %}">
            <td style="font-weight:600">{{ row.entity }}</td>
            <td>
              <div class="bar"><span style="width: {{ (row.prob*100)|round(1) }}%"></span></div>
              <div class="small">{{ '%.3f'|format(row.prob) }}</div>
            </td>
            <td class="small">{{ '%.2f'|format(row.th) }}</td>
            <td>
              {% if row.over %}
                <span class="badge ok">✓ true</span>
              {% else %}
                <span class="badge no">false</span>
              {% endif %}
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>

      {% if explain and resp.explanations %}
        <div class="section-title">Explain</div>
        {% for e, reasons in resp.explanations.items() %}
          <div style="font-weight:700;margin-top:10px">{{ e }}</div>
          <div class="explain-box">
            {% for r in reasons %}
              <div class="explain-item">• {{ r }}</div>
            {% endfor %}
          </div>
        {% endfor %}
      {% elif explain %}
        <div class="section-title">Explain</div>
        <div class="explain-box small">No explanations available for this query / labels.</div>
      {% endif %}
    {% endif %}
  </div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    q = ""
    explain = False
    explain_source = "rule"
    resp = None
    rows = []

    if request.method == "POST":
        q = request.form.get("q","").strip()
        explain = bool(request.form.get("explain"))
        explain_source = (request.form.get("explain_source") or "rule").lower()
        if q:
            try:
                payload = {"query": q, "explain": explain, "topk": 5, "explain_source": explain_source}
                r = requests.post(API_URL, json=payload, timeout=20)
                r.raise_for_status()
                resp = r.json()
                # Chuẩn hóa dữ liệu bảng: sắp xếp nhãn over_threshold lên trên, prob giảm dần
                probs = resp.get("probs", {})
                labels = resp.get("labels", {})
                ths = resp.get("thresholds", {})
                tmp = []
                for e, p in probs.items():
                    tmp.append({
                        "entity": e,
                        "prob": float(p),
                        "th": float(ths.get(e, 0.5)),
                        "over": bool(labels.get(e, False))
                    })
                rows = sorted(tmp, key=lambda x: (not x["over"], -x["prob"], x["entity"]))
            except Exception:
                resp = {"query": q, "probs": {}, "labels": {}, "thresholds": {}, "explanations": None}
                rows = []

    return render_template_string(TEMPLATE, q=q, explain=explain, explain_source=explain_source, resp=resp, rows=rows)

if __name__ == "__main__":
    # chạy: python src/demo_flask.py
    app.run(host="127.0.0.1", port=5000, debug=True)
