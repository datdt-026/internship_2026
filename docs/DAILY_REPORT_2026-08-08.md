# Daily Report — 08/08/2026

**Sinh viên:** Đỗ Thành Đạt (M23.ICT.002)  
**Đề tài:** Decoding the Post-Pandemic Vaccine Hesitancy (Multimodal Transformer / X–TikTok)  
**Deadline:** 31/08/2026  
**Workspace:** `/Users/datdo/Desktop/USTH_Master/internship_2026`

---

## 1. Tóm tắt ngày

Hôm nay dựng xong nền tảng đồ án từ gần như zero → **có model text baseline chạy thật với F1 test ≈ 0.972**, đồng thời chốt hướng data ảnh (MMCoVaR) cho Step 4.

| Hạng mục | Kết quả |
|----------|---------|
| Step 1 — Scaffold + env | **Done** |
| Step 2 — Dataset CONSTRAINT + splits | **Done** |
| Step 3 — RoBERTa text baseline | **Done** (train + evaluate) |
| Step 4 — Multimodal / data ảnh | **Đã khảo sát & chốt nguồn** (chưa train fusion) |
| Docs / kế hoạch | **Done** |

---

## 2. Việc đã làm chi tiết

### 2.1 Scaffold project (Step 1)

- Tạo repo code tại `thesis/` với cấu trúc:
  - `src/` (data, models, utils, train/evaluate/topics)
  - `configs/default.yaml`
  - `data/{raw,processed,splits}`, `results/`, `checkpoints/`
  - `scripts/`, `thesis_docs/`
- Sửa `scripts/bootstrap_env.sh` (xử lý `.venv` rỗng / tạo dở)
- Cài virtualenv + `requirements.txt` (torch, transformers, datasets, …)
- Smoke test scaffold chạy OK

### 2.2 Dataset text — CONSTRAINT (Step 2)

- Chốt nguồn chính: Hugging Face [`nanyy1025/covid_fake_news`](https://huggingface.co/datasets/nanyy1025/covid_fake_news) (Patwa / CONSTRAINT 2021)
- Viết & chạy:
  - `scripts/download_constraint_hf.py` → `data/raw/constraint_covid_fake_news/`
  - `scripts/prepare_constraint_splits.py` → map `real/fake` → `credible/misinfo`, clean text, JSONL splits
- Kiểm tra ANTiVax (`ANTiVax/Labeled/VaxMisinfoData.csv`): **chỉ có tweet ID**, không text → không dùng train trực tiếp (cần hydrate)

**Thống kê splits:**

| Split | n | credible | misinfo |
|-------|---|----------|---------|
| train | 6420 | 3360 | 3060 |
| val | 2140 | 1120 | 1020 |
| test | 2140 | 1120 | 1020 |
| **all** | **10700** | **5600** | **5100** |

- `with_image = 0` (CONSTRAINT text-only)

### 2.3 Text baseline RoBERTa (Step 3)

- Implement training loop thật:
  - `src/engine.py` — DataLoader, train/eval helpers, device (CUDA/MPS/CPU)
  - `src/train.py` — fine-tune, early stopping theo val F1, lưu best checkpoint
  - `src/evaluate.py` — load checkpoint, báo cáo metrics + classification report
- Smoke train (`--epochs 1 --max-steps 20`): F1 thấp ~0.34 (đúng kỳ vọng, chưa train đủ)
- **Full train** (`--modality text`, early stop epoch 4):

| Metric (TEST) | Giá trị |
|---------------|---------|
| Accuracy | **0.9720** |
| Precision | **0.9717** |
| Recall | **0.9723** |
| Macro-F1 | **0.9719** |

- Evaluate lại xác nhận cùng mức (~0.972), ghi `results/eval_test.json`
- Artifacts:
  - `checkpoints/text_roberta_best.pt`
  - `results/text_baseline_results.json`
  - `results/eval_test.json`

> Log `UNEXPECTED lm_head` / `MISSING pooler` khi load `roberta-base` là bình thường (không phải lỗi).

### 2.4 Khảo sát data ảnh (chuẩn bị Step 4)

- CONSTRAINT không có ảnh → cần nguồn multimodal khác
- Đã có sẵn `MMCoVaR/` trong workspace; inspect `MMCoVaR_News_Dataset.csv`:
  - ~**2593** news COVID vaccine
  - Cột `image` = URL ảnh (**2592/2593** có)
  - Text: `title`, `body_text`
  - Label: `reliability` (`1` reliable / `0` unreliable) → map `credible` / `misinfo`
- **Chốt:** MMCoVaR News = nguồn multimodal chính cho Step 4  
- Backup nếu URL ảnh chết nhiều: Fakeddit (có bundle ảnh, nhưng không riêng vaccine)

### 2.5 Tài liệu trong `docs/`

| File | Nội dung |
|------|----------|
| `docs/RUN_COMMANDS.md` | Lệnh setup, download data, prepare splits, train/eval |
| `docs/PROJECT_PLAN_REPORT.md` | Các bước, mục tiêu tuần/ngày, KPI, show-off, daily log |
| `docs/DAILY_REPORT_2026-08-08.md` | Báo cáo ngày hôm nay (file này) |

---

## 3. File / script quan trọng tạo hoặc cập nhật hôm nay

```
thesis/
  configs/default.yaml
  scripts/bootstrap_env.sh
  scripts/download_constraint_hf.py
  scripts/prepare_constraint_splits.py
  scripts/smoke_test.sh
  scripts/make_dummy_split.py
  src/engine.py
  src/train.py
  src/evaluate.py
  src/data/*  src/models/*  src/utils/*
  data/raw/constraint_covid_fake_news/
  data/processed/constraint_*.json*
  data/splits/{train,val,test}.jsonl
  checkpoints/text_roberta_best.pt
  results/text_baseline_results.json
  results/eval_test.json

docs/
  RUN_COMMANDS.md
  PROJECT_PLAN_REPORT.md
  DAILY_REPORT_2026-08-08.md
```

---

## 4. Quyết định kỹ thuật đã chốt

1. Survival scope trước deadline: text baseline vững trước, multimodal late fusion sau  
2. Dataset text chính = CONSTRAINT (HF), không phụ thuộc API X  
3. ANTiVax chỉ giữ làm nguồn phụ / cite (chưa hydrate)  
4. Multimodal tiếp theo = MMCoVaR News (URL ảnh)  
5. Metrics chuẩn: Accuracy / Precision / Recall / Macro-F1  
6. Không paste secret (HF token) vào chat/docs; dùng `thesis/.env` (gitignored)

---

## 5. Việc chưa làm / tồn đọng

- [ ] Download ảnh MMCoVaR từ URL + prepare multimodal splits  
- [ ] Implement / train late fusion (text + CLIP image) + ablation  
- [ ] BERTopic narrative analysis (Step 5)  
- [ ] Viết các chương thesis (Intro → Conclusion)  
- [ ] Slide bảo vệ  
- [ ] Sync chính thức survival scope với GVHD  

---

## 6. Kế hoạch ngày mai (09/08/2026)

1. Copy/prepare MMCoVaR news vào `thesis/data/raw/mmcovar/`  
2. Script tải ảnh + lọc mẫu có ảnh hợp lệ  
3. Tạo `data/splits` multimodal (hoặc tách `mmcovar_splits/`)  
4. Bắt đầu skeleton train fusion (nếu ảnh tải đủ)  
5. Cập nhật daily log trong `PROJECT_PLAN_REPORT.md`  

---

## 7. Lệnh reproduce kết quả hôm nay (cheat-sheet)

```bash
cd /Users/datdo/Desktop/USTH_Master/internship_2026/thesis
source .venv/bin/activate
export PYTHONPATH=.

# Data (đã chạy; chạy lại nếu cần)
python scripts/download_constraint_hf.py
python scripts/prepare_constraint_splits.py

# Train text baseline
python -m src.train --config configs/default.yaml --modality text

# Evaluate
python -m src.evaluate --config configs/default.yaml \
  --checkpoint checkpoints/text_roberta_best.pt --split test
```

Chi tiết đầy đủ: `docs/RUN_COMMANDS.md`.

---

## 8. Trạng thái tổng thể tới cuối ngày 08/08

```
Step 1 Scaffold     ██████████ Done
Step 2 Dataset      ██████████ Done (text)
Step 3 Text baseline ██████████ Done (F1≈0.972)
Step 4 Multimodal   ██░░░░░░░░ Nguồn ảnh đã chốt (MMCoVaR)
Step 5 Narrative    ░░░░░░░░░░ Pending
Step 6 Submit       ░░░░░░░░░░ Pending (deadline 31/08)
```

**Verdict ngày:** tiến độ tốt hơn kế hoạch tuần 1 (đã có số liệu test mạnh trong ngày đầu). Ưu tiên tiếp theo là **đóng Step 4 đủ để có bảng ablation**, rồi song song viết thesis.
