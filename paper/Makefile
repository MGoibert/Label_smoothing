all: clean
		pdflatex main
		bibtex main.aux
		pdflatex main.tex
		pdflatex main.tex

clean:
		rm -f *.aux *.log *.bbl *.blg *.run.xml *.bcf *.out
		rm -rf auto main.pdf
	  rm -f main.nlo main.nls main.toc main.ilg

