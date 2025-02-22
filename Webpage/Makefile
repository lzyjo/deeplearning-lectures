


OUTPUT_DIR=../Site
SRC = $(wildcard *.md)
HTML=$(addprefix $(OUTPUT_DIR)/, $(SRC:.md=.html))

SRC_LATEX=$(wildcard latex/*.tex)
DST_LATEX=$(addprefix $(OUTPUT_DIR)/, $(SRC_LATEX:.tex=.png)) 

.PHONY: copyextra
copyextra: 
	@cp Makefile $(OUTPUT_DIR)
	@cp biblio.bib $(OUTPUT_DIR)
	@cp -r templates $(OUTPUT_DIR)
	@cp -r data $(OUTPUT_DIR)

easylab:
	rm -rf $(OUTPUT_DIR)/labs/01-pytorch-segmentation
	easylabwork ../LabsSolutions/01-pytorch-segmentation  $(OUTPUT_DIR)/labs/01-pytorch-segmentation
	# For the LabWork pytorch-ASR
	easylabwork ../LabsSolutions/02-pytorch-asr  $(OUTPUT_DIR)/labs/02-pytorch-asr
	cp ../LabsSolutions/02-pytorch-asr/pytorch-asr-py3.9.yaml $(OUTPUT_DIR)/labs/02-pytorch-asr/
	# For the labwork on GANs
	easylabwork ../LabsSolutions/03-pytorch-gan  $(OUTPUT_DIR)/labs/03-pytorch-gan


all: outputdirs $(HTML) $(DST_LATEX) easylab copyextra

outputdirs:
	mkdir -p $(OUTPUT_DIR)/md-sources $(OUTPUT_DIR)/labs

clean:
	rm -rf $(OUTPUT_DIR)

$(HTML): | $(OUTPUT_DIR)

css:
	pygmentize -f html -S friendly -a .highlight > ./templates/pygmentize.css

$(OUTPUT_DIR)/%.html: %.md biblio.bib
	pandoc -r markdown+tex_math_dollars+simple_tables+table_captions+yaml_metadata_block+fenced_code_attributes+implicit_figures -w html --mathjax --standalone --table-of-contents --toc-depth=3 --template=./templates/html.template --filter ./templates/pandocfilter-pygments.py --filter pandoc-citeproc --csl=./templates/apsa.csl --bibliography=biblio.bib  -M date="`LC_TIME=en_US.UTF-8 date "+%B %e, %Y %H:%M"`" -M link-citations=true -M source=$< -o $@ $<
	cp $< $(OUTPUT_DIR)/md-sources/

$(OUTPUT_DIR)/latex/%.png: latex/%.tex
	latexmk -output-directory=$(OUTPUT_DIR)/latex/ -pdf $< 
	convert -density 300 $(@:.png=.pdf) $@

install-deps:
	sudo apt install pandoc pandoc-data pandoc-citeproc python3-pygments python3-pandocfilters
	mkdir -p latex/layers
	wget -P latex/layers https://raw.githubusercontent.com/HarisIqbal88/PlotNeuralNet/master/layers/Ball.sty
	wget -P latex/layers https://raw.githubusercontent.com/HarisIqbal88/PlotNeuralNet/master/layers/Box.sty
	wget -P latex/layers https://raw.githubusercontent.com/HarisIqbal88/PlotNeuralNet/master/layers/RightBandedBox.sty
	wget -P latex/layers https://raw.githubusercontent.com/HarisIqbal88/PlotNeuralNet/master/layers/init.tex
	pip install  git+https://github.com/jeremyfix/easylabwork.git

