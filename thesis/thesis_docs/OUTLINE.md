# Thesis outline (survival scope)

**Title (EN):** Decoding the Post-Pandemic Vaccine Hesitancy: A Multimodal Transformer Approach to Misinformation Dynamics on X and TikTok  

**Student:** Do Thanh Dat (M23.ICT.002)  
**Deadline:** 31/08/2026

## Research questions (draft)

1. Does multimodal (text + image) late fusion improve vaccine misinformation detection over text-only baselines on our dataset?  
2. Which modality contributes more to detection performance in an ablation study?  
3. What dominant post-pandemic vaccine hesitancy narratives appear in classified misinformation content?

## Chapter plan

### 1. Introduction
- Post-pandemic hesitancy + social media multimodality
- Problem statement & gap (unimodal vs multimodal content-level detection)
- Contributions (dataset curation subset, fusion model, narrative taxonomy)
- Thesis structure

### 2. Related Work
- Vaccine hesitancy / misinformation on social media
- Multimodal misinformation detection (MMCOVID, transformer fusion)
- Encoders: BERT/RoBERTa, CLIP/ViT, (Whisper as future work)
- Topic modeling for narrative analysis (BERTopic)

### 3. Methodology
- Problem formulation (binary / 3-class classification)
- Data pipeline (collect → clean → label map → splits)
- Models: text-only, image-only, late fusion
- Training setup & metrics
- Narrative analysis pipeline

### 4. Experiments & Results
- Dataset statistics
- Implementation details
- Main results table
- Ablation (text / image / fusion)
- Error analysis

### 5. Narrative Analysis
- BERTopic themes on misinfo class
- Qualitative taxonomy (named themes + examples)
- Implications for public-health communication

### 6. Discussion & Limitations
- Scope cuts (API limits, TikTok audio, dataset size)
- Ethics / dual-use of misinfo detectors
- Threats to validity

### 7. Conclusion & Future Work
- Summary of findings
- Extensions: Whisper audio, cross-platform TikTok scale, cross-attention fusion

## Writing rule

Only describe methods that are implemented and evaluated. Mark deferred items as Future Work / Limitations.
