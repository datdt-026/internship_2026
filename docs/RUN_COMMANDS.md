# Lệnh chạy project — Thesis ICT (Vaccine Misinfo Multimodal)

**Sinh viên:** Do Thanh Dat (M23.ICT.002)  
**Deadline:** 31/08/2026  
**Project code:** `internship_2026/thesis/`  

**Kế hoạch đồ án (bước / tuần / ngày / kết quả / show-off):** xem [`PROJECT_PLAN_REPORT.md`](./PROJECT_PLAN_REPORT.md)

---

## 0. Đường dẫn quan trọng

| Mục | Path |
|-----|------|
| Code project | `/Users/datdo/Desktop/USTH_Master/internship_2026/thesis` |
| Docs (file này) | `/Users/datdo/Desktop/USTH_Master/internship_2026/docs` |
| Data raw (CONSTRAINT) | `thesis/data/raw/constraint_covid_fake_news/` |
| ANTiVax (chỉ tweet ID, chưa hydrate) | `internship_2026/ANTiVax/` |
| Proposal / form | `internship_2026/DatDT.M23.ICT.002-Intership-2026/` |

Luôn `cd` vào thư mục `thesis` trước khi chạy lệnh bên dưới (trừ khi ghi chú khác).

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
```

---

## 1. Setup môi trường (làm 1 lần, hoặc khi thêm package)

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
bash scripts/bootstrap_env.sh
```

Script sẽ:
- tạo `.venv` nếu chưa có / bị hỏng
- cài `requirements.txt`

### Bật virtualenv mỗi session mới

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
```

Thấy prefix `(.venv)` là đã vào đúng môi trường.

### (Tuỳ chọn) Token Hugging Face — tăng rate limit

```bash
export HF_TOKEN=hf_xxxxxxxx
# hoặc copy .env.example -> .env rồi điền HF_TOKEN=
```

---

## 2. Smoke test scaffold (kiểm tra project chạy được)

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
bash scripts/smoke_test.sh
```

Kỳ vọng cuối output có dạng: `SMOKE OK` / `All smoke checks passed.`

### Entrypoint stub riêng

```bash
source .venv/bin/activate
export PYTHONPATH=.

python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
python -m src.topics --config configs/default.yaml
```

> Hiện tại Step 1: train/eval/topics vẫn là **scaffold** (chưa train model thật). Full training = Step 3.

### Dummy split (test layout data, không dùng cho experiment thật)

```bash
source .venv/bin/activate
python scripts/make_dummy_split.py
```

---

## 3. Tải dataset Hugging Face (CONSTRAINT COVID Fake News)

Dataset: [`nanyy1025/covid_fake_news`](https://huggingface.co/datasets/nanyy1025/covid_fake_news)  
(Patwa et al. — Fighting an Infodemic / CONSTRAINT 2021)

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
python scripts/download_constraint_hf.py
```

### Output sau khi tải

```
data/raw/constraint_covid_fake_news/
├── train.csv          # 6420
├── validation.csv     # 2140
├── test.csv           # 2140
└── all.csv            # 10700
```

Cột: `id`, `tweet`, `label` (`real` / `fake`)

### Kiểm tra nhanh bằng Python

```bash
source .venv/bin/activate
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("nanyy1025/covid_fake_news")
print(ds)
print(ds["train"][0])
PY
```

### Chuẩn hóa → splits (Step 2)

Map `real→credible`, `fake→misinfo`, clean text, ghi JSONL:

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
python scripts/prepare_constraint_splits.py
```

Output:

```
data/processed/constraint_all.jsonl
data/processed/constraint_stats.json
data/splits/train.jsonl   # 6420
data/splits/val.jsonl     # 2140
data/splits/test.jsonl    # 2140
```

---

## 4. Step 3 — Train text baseline (RoBERTa)

**Điều kiện:** đã có `data/splits/{train,val,test}.jsonl`.

### 4.1 Train đầy đủ (khoảng vài chục phút tùy máy)

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
export PYTHONPATH=.

python -m src.train --config configs/default.yaml --modality text
```

Lần đầu sẽ **download `roberta-base`** (~500MB) từ Hugging Face.

### 4.2 Smoke train nhanh (thử pipeline, ~vài phút)

```bash
python -m src.train --config configs/default.yaml --modality text --epochs 1 --max-steps 20
```

### 4.3 Evaluate trên test set

```bash
python -m src.evaluate --config configs/default.yaml \
  --checkpoint checkpoints/text_roberta_best.pt \
  --split test
```

> Checkpoint mới cũng lưu `checkpoints/text_best.pt` (alias Step 3: `text_roberta_best.pt`).

### Output Step 3

| File | Nội dung |
|------|----------|
| `checkpoints/text_roberta_best.pt` | Checkpoint tốt nhất theo val F1 |
| `results/text_baseline_results.json` | History + test metrics |
| `results/eval_test.json` / `eval_text_test.json` | Báo cáo evaluate |

---

## 5. Step 4 — MMCoVaR multimodal (text + image + fusion)

### 5.1 Tải ảnh từ URL

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
python scripts/download_mmcovar_images.py --workers 12
```

Ảnh → `data/raw/mmcovar/images/` · manifest → `data/raw/mmcovar/image_download_manifest.jsonl`

Smoke nhanh (100 ảnh):

```bash
python scripts/download_mmcovar_images.py --limit 100 --workers 8
```

### 5.2 Tạo multimodal splits

```bash
python scripts/prepare_mmcovar_splits.py
```

Output: `data/splits_mmcovar/{train,val,test}.jsonl` + `data/processed/mmcovar_stats.json`

### 5.3 Train từng modality / ablation

```bash
export PYTHONPATH=.

# từng cái
python -m src.train --config configs/mmcovar.yaml --modality text
python -m src.train --config configs/mmcovar.yaml --modality image
python -m src.train --config configs/mmcovar.yaml --modality fusion

# hoặc chạy cả ablation
bash scripts/run_mmcovar_ablation.sh
```

Smoke fusion:

```bash
python -m src.train --config configs/mmcovar.yaml --modality fusion --epochs 1 --max-steps 10
```

### 5.4 Evaluate

```bash
python -m src.evaluate --config configs/mmcovar.yaml \
  --checkpoint checkpoints/fusion_best.pt --modality fusion --split test
```

### Output Step 4

| File | Nội dung |
|------|----------|
| `checkpoints/{text,image,fusion}_best.pt` | Best ckpt theo modality |
| `results/{text,image,fusion}_results.json` | Metrics |
| `results/mmcovar_ablation_summary.json` | Bảng so sánh (sau ablation script) |

---

## 6. Step 5 — BERTopic narrative analysis

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
export PYTHONPATH=.

# Default: CONSTRAINT + MMCoVaR misinfo texts
python -m src.topics --config configs/default.yaml --output results/topics --top-n 10

# Chỉ CONSTRAINT
python -m src.topics --input data/processed/constraint_all.jsonl \
  --output results/topics_constraint --top-n 10
```

Output chính:

| File | Dùng để |
|------|---------|
| `results/topics/TAXONOMY_THESIS.md` | Bảng narrative cho Chapter 5 |
| `results/topics/taxonomy.md` | Auto themes + examples |
| `results/topics/taxonomy.json` | Machine-readable |
| `results/topics/topic_info.csv` | Full topic stats |
| `results/topics/doc_topics.jsonl` | Gán topic từng document |

---

## 7. Ghi chú ANTiVax (đã clone sẵn)

**Điều kiện:** đã có `data/splits/{train,val,test}.jsonl`.

### 4.1 Train đầy đủ (khoảng vài chục phút tùy máy)

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
export PYTHONPATH=.

python -m src.train --config configs/default.yaml --modality text
```

Lần đầu sẽ **download `roberta-base`** (~500MB) từ Hugging Face.

### 4.2 Smoke train nhanh (thử pipeline, ~vài phút)

```bash
python -m src.train --config configs/default.yaml --modality text --epochs 1 --max-steps 20
```

### 4.3 Evaluate trên test set

```bash
python -m src.evaluate --config configs/default.yaml \
  --checkpoint checkpoints/text_roberta_best.pt \
  --split test
```

### Output Step 3

| File | Nội dung |
|------|----------|
| `checkpoints/text_roberta_best.pt` | Checkpoint tốt nhất theo val F1 |
| `results/text_baseline_results.json` | History + test metrics |
| `results/eval_test.json` | Báo cáo evaluate |

Đọc kết quả:

```bash
cat results/text_baseline_results.json
```

---

## 5. Ghi chú ANTiVax (đã clone sẵn)

Path: `/Users/datdo/Desktop/USTH_Master/internship_2026/ANTiVax`

| File | Dùng? |
|------|--------|
| `Labeled/VaxMisinfoData.csv` | Có label (`id`, `is_misinfo`) nhưng **không có text** |
| `Labeled/ids.txt` | Không cần |
| `VaccineTweets/*.csv` | Chỉ tweet ID, không label → không train trực tiếp |

Muốn dùng ANTiVax phải **hydrate** tweet ID qua API X → chưa ưu tiên trước deadline.

Copy label file (tuỳ chọn, backup):

```bash
mkdir -p /Users/datdo/Desktop/USTH_Master/internship_2026/thesis/data/raw/antivax
cp /Users/datdo/Desktop/USTH_Master/internship_2026/ANTiVax/Labeled/VaxMisinfoData.csv \
   /Users/datdo/Desktop/USTH_Master/internship_2026/thesis/data/raw/antivax/
```

---

## 6. Checklist theo bước (timeline gấp)

| Bước | Việc | Lệnh / trạng thái |
|------|------|-------------------|
| 1 | Scaffold + env | `bash scripts/bootstrap_env.sh` + `bash scripts/smoke_test.sh` ✅ |
| 2 | Dataset | `download_constraint_hf.py` + `prepare_constraint_splits.py` ✅ |
| 3 | Text baseline | `python -m src.train --modality text` |
| 4 | Late fusion + ablation | train fusion / image |
| 5 | BERTopic narratives | `python -m src.topics` |
| 6 | Viết thesis + slide + nộp | `thesis_docs/` |

---

## 7. Troubleshooting nhanh

### `.venv/bin/activate: No such file or directory`

`.venv` bị tạo dở / rỗng. Chạy lại:

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
rm -rf .venv
bash scripts/bootstrap_env.sh
```

(`bootstrap_env.sh` đã tự xoá và tạo lại nếu thiếu `bin/activate`.)

### `ModuleNotFoundError`

Chưa activate venv, hoặc chưa install:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Lệnh Python không thấy package `src`

Chạy từ thư mục `thesis` và set `PYTHONPATH`:

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
export PYTHONPATH=.
python -m src.train --config configs/default.yaml
```

(`scripts/smoke_test.sh` đã set `PYTHONPATH` giúp bạn.)

---

## 8. Cheat-sheet copy/paste (session thường ngày)

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
export PYTHONPATH=.

# kiểm tra môi trường
bash scripts/smoke_test.sh

# tải lại data HF (nếu cần)
python scripts/download_constraint_hf.py

# chuẩn hóa splits (Step 2)
python scripts/prepare_constraint_splits.py

# Step 3 — text baseline
python -m src.train --config configs/default.yaml --modality text
python -m src.evaluate --config configs/default.yaml \
  --checkpoint checkpoints/text_roberta_best.pt --split test
```
