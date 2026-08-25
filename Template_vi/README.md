# Bản LaTeX tiếng Việt (đọc hiểu / luyện bảo vệ)

**Không dùng để nộp chính thức.** Bản nộp = `Template/` (tiếng Anh).

## File
- `main_vi.tex` — báo cáo 5 chương tiếng Việt
- `biblio.bib` — tham chiếu (giống bản Anh)
- `USTH-logo.png` — logo bìa

## Compile trên Overleaf
1. Upload cả folder `Template_vi/`
2. Main document: `main_vi.tex`
3. Compiler: **pdfLaTeX**, bibliography: **biber**
4. Nếu font tiếng Việt lỗi: Menu → Compiler thử **XeLaTeX**, hoặc thêm package `vntex` theo hướng dẫn Overleaf Vietnamese

## Local (nếu có TeX)
```bash
cd Template_vi
pdflatex main_vi
biber main_vi
pdflatex main_vi
pdflatex main_vi
```
