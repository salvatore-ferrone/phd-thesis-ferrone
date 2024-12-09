PDF_DIR=pdf
HTML_DIR=docs
CHAPTERS=chapters/introduction.tex chapters/numerics.tex chapters/results.tex chapters/theory.tex
BIBLIOGRAPHY=mybib.bib
CSL=chicago-author-date.csl
PANDOC_FLAGS=--standalone --mathjax --bibliography=$(BIBLIOGRAPHY) --csl=$(CSL) --citeproc

all: pdf html

pdf:
	mkdir -p $(PDF_DIR)
	pdflatex -output-directory=$(PDF_DIR) main.tex
	bibtex $(PDF_DIR)/main
	pdflatex -output-directory=$(PDF_DIR) main.tex
	pdflatex -output-directory=$(PDF_DIR) main.tex

html: copy_images generate_homepage
	mkdir -p $(HTML_DIR)
	# Generate HTML from chapters
	pandoc -s -o $(HTML_DIR)/introduction.html chapters/introduction.tex $(PANDOC_FLAGS) --metadata title="Introduction"
	pandoc -s -o $(HTML_DIR)/numerics.html chapters/numerics.tex $(PANDOC_FLAGS) --metadata title="Numerics"
	pandoc -s -o $(HTML_DIR)/results.html chapters/results.tex $(PANDOC_FLAGS) --metadata title="Results"
	pandoc -s -o $(HTML_DIR)/theory.html chapters/theory.tex $(PANDOC_FLAGS) --metadata title="Theory"

	# Generate HTML homepage
	pandoc -s -o $(HTML_DIR)/index.html homepage_preprocessed.md $(PANDOC_FLAGS) --metadata title="Table of Contents"

copy_images:
	mkdir -p $(HTML_DIR)/images
	cp -r images/* $(HTML_DIR)/images/

generate_homepage:
	# Convert abstract.tex to markdown and insert into homepage.md
	pandoc -s -t markdown chapters/abstract.tex > abstract.md
	# Concatenate abstract.md into homepage.md to create homepage_preprocessed.md
	cat homepage.md abstract.md > homepage_preprocessed.md

clean:
	rm -rf $(PDF_DIR) $(HTML_DIR) *.aux *.bbl *.blg *.log *.out *.toc abstract.md homepage_preprocessed.md
