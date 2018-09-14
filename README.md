# Model-agnostic explanations of black box classifiers for image recognition


This repository contains the source code of my master thesis 
(Model-agnostic explanations of black box classifiers for image recognition) and the annotated data
set used in the experiments.

This work is an extension of [LIME](https://github.com/marcotcr/lime) (M. T. Ribeiro, S. Singh, and C. Guestrin).

## Relevant folders:

- `chosen_1000_images` contains the 1000 images used in the experiments
- `immagini_ritagliate` contains the reference areas for each image in `chosen_1000_images` (i.e. the annotated dataset)
- `master thesis` includes all the LaTeX files and pdf version of the thesis

## Relevant files:

- `predizioni_bb.txt` contains the predictions of the black box on each of the 1000 images in the test set
- `chosen_classes_for_validation.txt` specifies the chosen labels from the ILSVRC dataset
- `requirements.txt` specifies the required Python libraries

## Experiments 

- Source code to run the experiments can be found under `lime/lime/`

## Jupyter Notebooks

- Under `lime/doc/notebooks/` you can find various example notebooks used to generate plots and figures
