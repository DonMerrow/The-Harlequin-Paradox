.PHONY: all figures paper clean

all: figures paper

figures:
	python simulate_harlequin.py
	python plot_phase_space.py
	python time_integrated_sim.py
	python plot_least_time_refraction.py

paper:
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz