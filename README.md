# Prediction Rigidities for Data-Driven Chemistry

This repository contains the inputs and scripts that were used to obtain the results shared in [our contribution](https://pubs.rsc.org/en/content/articlelanding/2024/fd/d4fd00101j) to the Faraday Discussions: "Data-driven discovery in the chemical sciences". Datasets and trained models are available on Materials Cloud, [here](https://doi.org/10.24435/materialscloud:6x-gs). 

The files are organized into the following directories:

### `section_3`: PRs of NN models

This directory contains the NN model training inputs and example analysis scripts in the form of Jupyter notebooks for the results shared in Section 3 of our paper. Scripts for the acquisition and generation of training datasets are also made available.

For MACE, [native implementation](https://github.com/ACEsuit/mace) was used for training, but the analysis was performed with a version available [here](https://github.com/SanggyuChong/mace/tree/LLPR_farad), which allows for the last-layer prediction rigidity (LLPR) analysis (see [Bigi et al.](https://arxiv.org/abs/2403.02251)).

For PaiNN, again, [native implementation](https://github.com/atomistic-machine-learning/schnetpack) was used for training, but the analysis was performed with a version available [here](https://github.com/SanggyuChong/schnetpack/tree/LLPR) that contains the LLPR implementation.

For SOAP-BPNN, [`metatrain`](https://github.com/lab-cosmo/metatrain) was used for training. Features for the LLPR analysis is actively being incorporated into `main` branch, but should all be available in the `llpr` branch.


### `section_4a`: PR-guided dataset construction -- dataset augmentation

This directory contains the three Jupyter notebooks that were used in training the models for the analysis shown in Figure 4 of the manuscript. Models are being trained within the notebook using [LE-ACE](https://github.com/frostedoyster/LE-ACE).

### `section_4b`: PR-guided dataset construction -- active learning

This directory contains the Python files and shell scripts used to perform the analysis shown in Figure 5 of the manuscript. `dia.xyz` and `MD_structure.xyz` are each single structure files that were used during the analysis. The extraction and embedding strategies are implemented in `extract.py`, `embed_FG.py`, and `embed_SC.py`. Main analysis script is written in `lpr_md_refined.py`.

### `section_5a`: Component-wise prediction rigidity -- body-ordered model

This directory contains the Julia script used to compute the ACE feature vectors (using [`ACE.jl`](https://github.com/ACEsuit/ACE.jl)) before and after "purification", implemented by [Ho et al.](https://www.sciencedirect.com/science/article/pii/S0021999124005199), as well as Jupyter notebooks used for the analysis in Section 5, Figure 6 of the manuscript.

Note that silicon cluster generation and reference energy calculations for the analysis were done with the same inputs and scripts provided for `section_3`.

### `section_5b`: Component-wise prediction rigidity -- multi-range model

This directory contains the Jupyter notebook used to perform the analysis in Section 5, Figure 7 of the manuscript. [`rascaline`](https://github.com/Luthaf/rascaline) was used to compute the SOAP and LODE features. Descriptor management was done with [`metatensor`](https://github.com/lab-cosmo/metatensor).

### `section_6`: Application to coarse-grained water model

This directory contains (1) inputs for MACE model training, and the subsequent MD simulations of coarse-grained water with the trained models; (2) Jupyter notebooks used for the analysis results shown in Section 6 of the manuscript.

It is very important to note that the MACE model training was done using a custom MACE implementation that enforces all the nonlinear activation functions to be `tanh`, even the ones that cannot be controlled from the native implementation. This implementation can be accessed [here](https://github.com/SanggyuChong/mace/tree/cg_fd)). LLPR analysis can be done with the same implementation provided above for `section_3`.
