PDF=particle_nn_report.pdf
TEX=particle_nn_report.tex
BASENAME=particle_nn_report

all:
	latexmk -pdf $(TEX)

clean:
	latexmk -c $(TEX)

distclean:
	latexmk -C $(TEX)

manual:
	pdflatex $(TEX)
	bibtex $(BASENAME)
	pdflatex $(TEX)
	pdflatex $(TEX)
