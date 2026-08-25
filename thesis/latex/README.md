# LaTeX Master Thesis

**Title:** Decoding the Post-Pandemic Vaccine Hesitancy: A Multimodal Transformer Approach to Misinformation Dynamics on Social Media  
**Student:** Do Thanh Dat (M23.ICT.002)

## Compile

### Option A — latexmk (recommended)

```bash
cd thesis/latex
latexmk -pdf -interaction=nonstopmode main.tex
```

### Option B — manual

```bash
cd thesis/latex
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf`

### Overleaf

1. Upload the entire `latex/` folder to Overleaf  
2. Set main document to `main.tex`  
3. Compiler: pdfLaTeX; enable biber/biblatex  

## Structure

```
latex/
  main.tex
  refs.bib
  chapters/
    01_introduction.tex
    02_related_work.tex
    03_methodology.tex
    04_experiments.tex
    05_narratives.tex
    06_discussion.tex
    07_conclusion.tex
    A_appendix.tex
```

## Notes

- Content matches implemented experiments (CONSTRAINT, MMCoVaR ablation, BERTopic).
- TikTok/Whisper/X live crawl are discussed as limitations / future work (honest scope).
- After editing numbers, recompile; keep `results/*.json` as source of truth.
# internship_2026
