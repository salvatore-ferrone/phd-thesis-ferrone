PDF_DIR=output/pdf
HTML_DIR=output/html
CHAPTERS=chapters/introduction.tex chapters/numerics.tex chapters/results.tex chapters/theory.tex
BIBLIOGRAPHY=mybib.bib
CSL=chicago-author-date.csl
# PANDOC_FLAGS=--standalone --mathjax --bibliography=$(BIBLIOGRAPHY) --csl=$(CSL) --citeproc
PANDOC_FLAGS=--standalone --mathjax --bibliography=$(BIBLIOGRAPHY) --csl=$(CSL) --citeproc --filter pandoc-citeproc



all: pdf html

pdf:
	mkdir -p $(PDF_DIR)
	pdflatex -output-directory=$(PDF_DIR) main.tex
	bibtex $(PDF_DIR)/main
	pdflatex -output-directory=$(PDF_DIR) main.tex
	pdflatex -output-directory=$(PDF_DIR) main.tex

html: copy_images
	mkdir -p $(HTML_DIR)
	pandoc -s -o $(HTML_DIR)/index.html main.tex $(PANDOC_FLAGS) --metadata title="Main"
	pandoc -s -o $(HTML_DIR)/introduction.html chapters/introduction.tex $(PANDOC_FLAGS) --metadata title="Introduction"
	pandoc -s -o $(HTML_DIR)/numerics.html chapters/numerics.tex $(PANDOC_FLAGS) --metadata title="Numerics"
	pandoc -s -o $(HTML_DIR)/results.html chapters/results.tex $(PANDOC_FLAGS) --metadata title="Results"
	pandoc -s -o $(HTML_DIR)/theory.html chapters/theory.tex $(PANDOC_FLAGS) --metadata title="Theory"

copy_images:
	mkdir -p $(HTML_DIR)/images
	cp -r images/* $(HTML_DIR)/images/

clean:
	rm -rf $(PDF_DIR) $(HTML_DIR) *.aux *.bbl *.blg *.log *.out *.toc