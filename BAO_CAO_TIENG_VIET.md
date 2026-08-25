# Báo cáo luận văn (bản tiếng Việt để đọc hiểu)

**Sinh viên:** Đỗ Thành Đạt — M23.ICT.002  
**Email:** datdt.m23ict@usth.edu.vn  
**Giảng viên hướng dẫn:** Giang Anh Tuan  
**Năm học:** 2025–2026  

**Đề tài (EN):** Decoding the Post-Pandemic Vaccine Hesitancy: A Multimodal Transformer Approach to Misinformation Dynamics on Social Media  

> Đây là bản **tiếng Việt để bạn đọc hiểu**. Bản nộp chính thức vẫn là LaTeX tiếng Anh trong folder `Template/`.

---

## Tóm tắt (Abstract)

Sau COVID-19, **do dự tiêm vaccine** (vaccine hesitancy) vẫn bị khuếch đại bởi thông tin sai lệch trên mạng xã hội và báo chí số. Nội dung hiện nay thường **đa phương thức** (text + ảnh), nên chỉ dùng text có thể thiếu tín hiệu thị giác.

Luận văn này xây pipeline **phát hiện misinformation** (tin đáng tin / tin sai) ở mức nội dung, dùng Transformer, và phân tích **các chủ đề tường thuật** liên quan do dự tiêm.

**Kết quả chính:**
1. Trên CONSTRAINT (text): RoBERTa đạt **macro-F1 = 0.9719** (accuracy 0.9720). Baseline cổ điển TF-IDF + Logistic Regression chỉ **0.9224**.
2. Trên MMCoVaR (text + ảnh): text mạnh (~**0.963**), ảnh yếu (~**0.554**), late fusion gần bằng text (~**0.963**) — **ảnh gần như không giúp thêm**.
3. BERTopic trên 5.752 văn bản misinfo cho taxonomy chủ đề: số ca/tử vong giật gân, chính trị hóa, khẩu trang, lab/China, thuốc thần kỳ, Bill Gates, bệnh viện, trẻ em/trường học, bài thuốc dân gian, tin đồn cấm đi lại.

**Đóng góp:** pipeline tái lập được, bằng chứng đóng góp từng modality, bản đồ narrative phục vụ truyền thông y tế. TikTok/Whisper/crawl live = hạn chế / hướng tương lai.

---

## 1. Bối cảnh & câu hỏi nghiên cứu

### Vì sao làm đề tài này?
- WHO gọi giai đoạn COVID là **infodemic**: tin thật/giả/gây hiểu nhầm lan rất nhanh.
- Vaccine hesitancy = trì hoãn hoặc từ chối tiêm dù vaccine sẵn có.
- Post hiện đại: caption “trung tính” + ảnh gây sợ, hoặc bài báo dài + thumbnail cảm xúc.

### Bài toán
Cho nội dung \(x = (t, v)\) (text + ảnh tùy chọn) → dự đoán:
- `credible` (0) = đáng tin  
- `misinfo` (1) = sai lệch / không đáng tin  

### 3 câu hỏi (RQ)
1. **RQ1:** Late fusion (text+ảnh) có tốt hơn text-only trên bộ tin tức đa phương thức không?
2. **RQ2:** Ablation: modality nào đóng góp nhiều hơn?
3. **RQ3:** Chủ đề misinfo / hesitancy nổi bật là gì (topic modeling)?

### Phạm vi thực tế (survival scope)
Proposal gốc muốn X + TikTok + Whisper. Do thời gian + API, luận văn làm:
- CONSTRAINT (text benchmark)
- MMCoVaR news + ảnh tải về (ablation)
- BERTopic (narrative)
- **Chưa làm:** crawl live, TikTok audio (Whisper), cross-attention phức tạp

→ Scope nhỏ hơn proposal nhưng **đủ vòng khoa học**: data → model → experiment → limitation.

---

## 2. Công trình liên quan (Related work — ý chính)

| Hướng | Ý chính | Liên hệ luận văn |
|-------|---------|------------------|
| Fake news trên MXH | Content-based vs social-context (Shu et al.) | Ta làm **content-based** |
| Dataset COVID | CONSTRAINT, CoAID, ANTi-Vax, MMCoVaR | CONSTRAINT + MMCoVaR (có text/ảnh) |
| Multimodal | Fakeddit, CLIP, late vs cross-attention | Late fusion đơn giản, dễ thiếu modality |
| Topic | BERTopic + Sentence-BERT | Taxonomy narrative |

**ANTi-Vax** chỉ có tweet ID → hydrate khó (API) → không dùng để train chính.

---

## 3. Phương pháp (Contributions / Methodology)

### Pipeline tổng quát
1. Chuẩn bị data + map nhãn  
2. Encoder: RoBERTa (text), CLIP (ảnh)  
3. Classifier: text / image / fusion  
4. Đánh giá: Acc, P, R, macro-F1  
5. BERTopic trên tập misinfo  

### Nhãn chuẩn
- CONSTRAINT: `real`→credible, `fake`→misinfo  
- MMCoVaR: `reliability=1`→credible, `0`→misinfo  

### Dataset

**CONSTRAINT** (text, split chính thức):

| Split | N | Credible | Misinfo |
|-------|---|----------|---------|
| Train | 6420 | 3360 | 3060 |
| Val | 2140 | 1120 | 1020 |
| Test | 2140 | 1120 | 1020 |
| **All** | **10700** | **5600** | **5100** |

**MMCoVaR news** (có ảnh tải thành công):

| Split | N | Credible | Misinfo |
|-------|---|----------|---------|
| Train | 1491 | 1034 | 457 |
| Val | 320 | 222 | 98 |
| Test | 320 | 222 | 98 |
| **All** | **2131** | **1478** | **653** |

- 2592 URL ảnh → tải OK **2131**, fail **461** (~18%).

### Mô hình
- **Text:** RoBERTa-base → CLS → linear head  
- **Image:** CLIP ViT-B/32 **đóng băng** encoder → linear head  
- **Late fusion:** nối \(h_t\) và \(h_v\) → MLP  

### Vì sao fusion có thể “không hơn text”?
Nếu ảnh nhiễu / ít liên quan nhãn, MLP học **bỏ qua** khối ảnh → fusion ≈ text. Đó là quan sát thực nghiệm quan trọng, không phải thất bại kỹ thuật thuần túy.

### BERTopic
- Gộp ~**5752** văn bản misinfo (CONSTRAINT + MMCoVaR)  
- Embed: `all-MiniLM-L6-v2`  
- `min_topic_size=15`  
- Đặt tên chủ đề thủ công từ keyword  

---

## 4. Thực nghiệm & kết quả

### A. CONSTRAINT — text

| Model | Acc | P | R | Macro-F1 |
|-------|-----|---|---|----------|
| TF-IDF + Logistic Regression | 0.9224 | 0.9222 | 0.9230 | **0.9224** |
| RoBERTa-base | 0.9720 | 0.9717 | 0.9723 | **0.9719** |

→ Transformer hơn baseline cổ điển ~**5 điểm F1**. RoBERTa là “trần” text mạnh.

### B. Error analysis (RoBERTa trên CONSTRAINT test)

- Tổng test: **2140**  
- Đúng: **2080** | Sai: **60** (~2.8%)  
- False Positive (credible→misinfo): **38**  
- False Negative (misinfo→credible): **22**  

| Nhóm lỗi (heuristic) | Số lượng | Ý nghĩa |
|----------------------|----------|---------|
| Subtle / hedged misinfo (FN) | 20 | Tin giả “giống tin thật”, giọng dè dặt |
| Stylistic false alarm (FP) | 16 | Văn phong giật gân nhưng nhãn vẫn credible |
| Short / ambiguous | 16 | Câu ngắn, thiếu ngữ cảnh |
| Hedged credible (FP) | 3 | Có từ may/might… |
| Alarmist but credible (FP) | 3 | Báo động nhưng đúng |
| Politicized credible | 1 | Mang màu chính trị |
| Miracle-cure missed (FN) | 1 | Lỡ “thuốc thần” |

**Bài học:** lỗi còn lại thường là biên (ngắn, bóng gió, giọng báo chí), không phải model “hỏng”.

### C. MMCoVaR — ablation multimodal

| Modality | Acc | P | R | Macro-F1 |
|----------|-----|---|---|----------|
| Text-only | 0.9688 | 0.9658 | 0.9604 | **0.9630** |
| Image-only | 0.5969 | 0.5550 | 0.5613 | **0.5542** |
| Late fusion | 0.9688 | 0.9632 | 0.9632 | **0.9632** |

**Trả lời RQ:**
- **RQ1:** Fusion **không rõ hơn** text.  
- **RQ2:** **Text >> Image**.  

**Vì sao?** Thân bài báo dài, ảnh hay chỉ minh họa; nhãn theo độ tin cậy nhà xuất bản; CLIP đóng băng; ~18% ảnh tải lỗi.

Baseline TF-IDF trên MMCoVaR text: macro-F1 **0.9180** (vẫn dưới RoBERTa 0.963).

### D. BERTopic — taxonomy (top 10)

| # | Chủ đề | Size | Ý nghĩa ngắn |
|---|--------|------|--------------|
| 1 | Số ca/tử vong giật gân | 520 | Phóng đại mức độ nghiêm trọng |
| 2 | Chính trị hóa đại dịch | 500 | Đổ lỗi đảng phái |
| 3 | Biện pháp (khẩu trang…) | 214 | Tranh cãi NPI + tin đồn |
| 4 | China / lab-origin | 188 | Âm mưu nguồn gốc |
| 5 | Thuốc thần / HCQ… | 116 | “Chữa khỏi” chưa kiểm chứng |
| 6 | Bill Gates / elite | 112 | Âm mưu kiểm soát |
| 7 | Hình ảnh quá tải BV | 97 | Visual gây sợ |
| 8 | Trường / trẻ em | 82 | Lo lắng phụ huynh |
| 9 | Bài thuốc dân gian | 79 | Muối/hơi nước/tỏi… |
| 10 | Tin đồn cấm đi lại | 76 | Policy giả |

~2000 doc nằm outlier (quá nhiễu/unique) — bình thường với post ngắn.

**RQ3:** Hesitancy không chỉ là “tác dụng phụ vaccine” mà là **hệ sinh thái narrative** (thiếu tin tưởng, thuốc thay thế, sợ hãi, chính trị).

---

## 5. Thảo luận, hạn chế, hướng sau

### Ý chính khi bảo vệ
1. Text baseline phải mạnh trước khi khoe multimodal.  
2. Trên MMCoVaR news, multimodal **không magic** — kết quả âm cũng là đóng góp khoa học.  
3. Detection + BERTopic = vòng giám sát nội dung cho truyền thông y tế.  

### Hạn chế (nói thật)
- Chưa X/TikTok live, chưa Whisper  
- Chủ yếu tiếng Anh  
- ~18% ảnh fail  
- Nhãn publisher ≠ fact-check từng câu  
- BERTopic phụ thuộc seed/tham số  

### Future work
TikTok + Whisper; cross-attention / unfreeze CLIP; nhãn claim-level; bộ tiếng Việt; dashboard narrative realtime.

### Đạo đức
Model hỗ trợ quyết định, **không** thay fact-checker; tránh over-block tranh luận hợp pháp/satire.

---

## 6. Code & file cần nhớ

```
thesis/
  scripts/run_sklearn_baseline.py      # baseline TF-IDF
  scripts/error_analysis.py            # phân tích lỗi RoBERTa
  scripts/run_mmcovar_ablation.sh
  src/train.py, evaluate.py, topics.py
  results/text_baseline_results.json
  results/sklearn_baseline_*.json
  results/mmcovar_ablation_summary.json
  results/error_analysis/roberta_test_errors.md
  results/topics/TAXONOMY_THESIS.md

Template/                              # BẢN NỘP CHÍNH THỨC (EN LaTeX)
  student-number.tex / M23.ICT.002.tex
  biblio.bib
  USTH-logo.png
```

### Lệnh chạy lại (nếu cần)
```bash
cd thesis && source .venv/bin/activate && export PYTHONPATH=.
python scripts/run_sklearn_baseline.py --tag constraint
python scripts/error_analysis.py --checkpoint checkpoints/text_roberta_best.pt
```

---

## 7. Checklist đọc tối nay (15–20 phút)

1. Nhớ **3 số vàng**: CONSTRAINT 0.9719 | fusion≈text 0.963 | image 0.55  
2. Nhớ **1 câu**: “Trên news MMCoVaR, text thống trị; fusion không thắng text.”  
3. Nhớ **taxonomy**: 2–3 chủ đề (Gates, HCQ, politicized)  
4. Nhớ **vì sao scope cắt TikTok**: API + thời gian; ghi future work  
5. Bản nộp = `Template/` tiếng Anh; file này chỉ để **hiểu**  

---

## 8. Câu hỏi bảo vệ có thể gặp (gợi ý trả lời ngắn)

**Q: Sao không làm TikTok như proposal?**  
A: API/thời gian hạn chế; giữ lõi khoa học (baseline + ablation + narrative); TikTok/Whisper = future work.

**Q: Fusion không hơn text thì multimodal vô ích?**  
A: Không — trên *dataset này* ảnh ít tín hiệu veracity; multimodal vẫn có ích ở meme/video ngắn; cần claim-level label và fusion mạnh hơn.

**Q: Vì sao không dùng ANTi-Vax?**  
A: Chỉ có ID, hydrate phụ thuộc API và tweet bị xóa → không reproducible trong internship.

**Q: Macro-F1 là gì?**  
A: Trung bình F1 hai lớp; công bằng hơn accuracy khi lệch lớp (MMCoVaR credible nhiều hơn).

---

*Chúc bạn đọc hiểu suôn sẻ. Mai cần rút slide 10–12 trang hoặc luyện Q&A thì nhắn tiếp.*
