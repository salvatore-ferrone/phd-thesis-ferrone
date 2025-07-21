PDF_DIR=pdf
HTML_DIR=docs
CHAPTERS=chapters/introduction.tex chapters/numerics.tex chapters/results.tex chapters/theory.tex
BIBLIOGRAPHY=mybib.bib
CSL=chicago-author-date.csl
PANDOC_FLAGS=--standalone --mathjax --bibliography=$(BIBLIOGRAPHY) --csl=$(CSL) --citeproc 

all: pdf html


# Draft mode - builds PDF without rendering images
draft:
	mkdir -p $(PDF_DIR)
	# Copy bibliography file to output directory
	cp $(BIBLIOGRAPHY) $(PDF_DIR)/
	mkdir -p $(PDF_DIR)/psl-cover
	cp config/psl-cover/* $(PDF_DIR)/psl-cover/
	# First LaTeX pass - uses TEXINPUTS to find input files
	TEXINPUTS=.:./chapters: pdflatex -output-directory=$(PDF_DIR) main.tex
	# Run BibTeX in the output directory
	cd $(PDF_DIR) && biber main
	# Run subsequent LaTeX passes
	TEXINPUTS=.:./chapters:./config/psl-cover: pdflatex -output-directory=$(PDF_DIR) main.tex
	TEXINPUTS=.:./chapters:./config/psl-cover: pdflatex -output-directory=$(PDF_DIR) main.tex

pdf:
	mkdir -p $(PDF_DIR)
	# Copy bibliography file to output directory
	cp $(BIBLIOGRAPHY) $(PDF_DIR)/
	# First LaTeX pass - uses TEXINPUTS to find input files
	TEXINPUTS=.:./chapters: pdflatex -output-directory=$(PDF_DIR) main.tex
	# Run BibTeX in the output directory
	cd $(PDF_DIR) && bibtex main
	# Run subsequent LaTeX passes
	TEXINPUTS=.:./chapters: pdflatex -output-directory=$(PDF_DIR) main.tex
	TEXINPUTS=.:./chapters: pdflatex -output-directory=$(PDF_DIR) main.tex

html: copy_images copy_videos generate_homepage
	mkdir -p $(HTML_DIR)
	pandoc -s -o $(HTML_DIR)/introduction.html chapters/introduction.tex $(PANDOC_FLAGS) --lua-filter=video-filter.lua --metadata title="Introduction"
	pandoc -s -o $(HTML_DIR)/numerics.html chapters/numerics.tex $(PANDOC_FLAGS) --lua-filter=video-filter.lua --metadata title="Numerics"
	pandoc -s -o $(HTML_DIR)/results.html chapters/results.tex $(PANDOC_FLAGS) --lua-filter=video-filter.lua --metadata title="Results"
	pandoc -s -o $(HTML_DIR)/theory.html chapters/theory.tex $(PANDOC_FLAGS) --lua-filter=video-filter.lua --metadata title="Theory"
	pandoc -s -o $(HTML_DIR)/gapology.html chapters/gapology.tex $(PANDOC_FLAGS) --lua-filter=video-filter.lua --metadata title="Gapology"

copy_videos:
	mkdir -p $(HTML_DIR)/videos
	cp -r videos/* $(HTML_DIR)/videos/

copy_images:
	mkdir -p $(HTML_DIR)/images
	cp -r images/* $(HTML_DIR)/images/

preprocess_homepage:
	sed -f include_abstract.sed homepage_base.md > homepage.md

generate_homepage: preprocess_homepage
	pandoc -s -o $(HTML_DIR)/index.html homepage.md $(PANDOC_FLAGS) --metadata title="Salvatore Ferrone"

clean:
	rm -rf $(PDF_DIR) $(HTML_DIR) *.aux *.bbl *.blg *.log *.out *.toc homepage.md