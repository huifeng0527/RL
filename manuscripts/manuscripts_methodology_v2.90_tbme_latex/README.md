# TBME LaTeX project — manuscript v2.90

This project converts `manuscripts_methodology_v2.89_figures_standardized.docx` to the full-paper LaTeX template from:

`G:\Download\Preparation_Papers_for_TBME_December_2025 (1)\generic-color.tex`

The DOCX remains the authoritative source for manuscript prose and numerical results.

## Front matter

The title is:

> Adaptive Difficulty Regulation for Noncontact Upper-Limb Rehabilitation Using Reinforcement Learning

The first-page author names, affiliations, funding, and corresponding-author details are visible placeholders and should be replaced before submission. No end-of-paper author biography blocks or author photographs are included.

## Build

Use pdfLaTeX and BibTeX. In Overleaf, set `main.tex` as the main file and select the pdfLaTeX compiler.

Local build sequence:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Alternatively:

```text
latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error main.tex
```

The current environment does not provide a LaTeX compiler, so the project was checked statically and packaged for Overleaf compilation.

## Project contents

- `main.tex` — TBME full-paper source using `ieeecolor2`
- `references.bib` — 51 BibTeX entries
- `citation_manifest.tsv` — reference-number/key mapping
- `ieeecolor2.cls`, `generic.sty`, `LOGO-generic-web.eps` — template support files
- `figures/` — the five standardized figures extracted from v2.89
