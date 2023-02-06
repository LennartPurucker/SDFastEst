# SDFastEst - Estimating the Pruned Search Space Size of Subgroup Discovery

## Contents of this Repository

This repository contains the code for SDFastEst and its evaluation. Hence, it contains the sampling framework, code for
processing the original datasets, code to collect algorithms' data, and code for application examples.

* `applications`: Contains code for three different use-case of SDFastEst (runtime meta-learning, warm starting,
  spectrum plots).
* `data`: Contains result data of SDFastEst; Plots from the paper (and any other relevant plot); Code to collect the
  search space size for different Pysubgroup algorithms, the resulting data of the collection (i.e., the ground truth
  used for evaluation), all SD task settings (combinations of depth, dataset, target value); Moreover, it contains the
  code used to process datasets, an overview of all used datasets.
* `general_utils`: Contains general utility code like file management. Also contains the code used to plot all plots in
  the paper and produce the tables' data.
* `SDFastEst`: Contains code to run the sampling framework SDFastEst, the framework itself, and auxiliary code like
  algorithm or sampling profiles.
* `sdfastest_comparer.py`: Main file used to produce SDFastEst's estimations and results. Setup to produce the results
  used in the paper plus some additional results with other sampling profiles.

## Installation and Usage

1. Download and preprocess the datasets
2. Set up the config.py accordingly to your own environment
3. Set up the Python environment with all requirements as listed in the requirements.txt

Having done the above, you can test some applications, run the fastest_comparer.py or directly use the SDFastEst
framework.

## Relevant publication

If you use our code or data in scientific publications, we would appreciate citations.

**Estimating the Pruned Search Space Size of Subgroup Discovery**, _Purucker et al.,_
_International Conference on Data Mining, 2022_

[Link](https://ieeexplore.ieee.org/abstract/document/10027642) to publication.

```
@inproceedings{purucker2022sdfastest,
  author    = {Lennart Purucker and
               Felix I. Stamm and
               Florian Lemmerich and
               Joeran Beel},
  title     = {Estimating the Pruned Search Space Size of Subgroup Discovery},
  booktitle = {{IEEE} International Conference on Data Mining, {ICDM} 2022, Orlando,
               FL, USA, November 28 - Dec. 1, 2022},
  year      = {2022},
}
```
