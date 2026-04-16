# LaTeX projekt: particle_nn_report

Tento archiv obsahuje kompletní zdrojový LaTeX projekt k reportu o klasifikaci detekovaných částic ve řídkých energetických maticích.

## Obsah

- `particle_nn_report.tex` - hlavní LaTeX dokument
- `particle_nn_report.bib` - BibTeX bibliografie
- `particle_nn_report.bbl` - vygenerovaná bibliografie pro snazší reprodukci
- `particle_nn_report.pdf` - hotové zkompilované PDF
- `assets/` - obrázky použité v dokumentu

## Doporučená kompilace

### Varianta 1: latexmk

```bash
latexmk -pdf particle_nn_report.tex
```

### Varianta 2: ručně

```bash
pdflatex particle_nn_report.tex
bibtex particle_nn_report
pdflatex particle_nn_report.tex
pdflatex particle_nn_report.tex
```

## Poznámky

- Projekt používá běžné balíčky TeX Live: `babel`, `graphicx`, `booktabs`, `tabularx`, `longtable`, `tcolorbox`, `natbib`, `siunitx`, `hyperref` a další.
- Dokument očekává relativní cestu `assets/` ve stejné složce jako `.tex` soubor.
- Soubor `particle_nn_report.bbl` je přiložen i pro případ, že chceš PDF znovu přeložit bez spouštění BibTeXu.
