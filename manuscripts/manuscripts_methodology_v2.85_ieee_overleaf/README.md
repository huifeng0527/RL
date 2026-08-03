# IEEE LaTeX conversion of manuscript v2.85

This project was converted from `manuscripts_methodology_v2.85_conclusion_polished.docx`. The Word document remains the authoritative source for the manuscript text and numerical results.

## Overleaf

1. Upload `manuscripts_methodology_v2.85_ieee_overleaf.zip` as a new Overleaf project.
2. Set `main.tex` as the main document.
3. Use the pdfLaTeX compiler.
4. Overleaf will run BibTeX automatically when `references.bib` is detected.

## Local build

Run either:

```text
latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error main.tex
```

or:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The project uses automatic numbering and clickable links for sections, equations, figures, tables, the algorithm, and citations. The author field is intentionally empty because the source DOCX does not contain a byline.
