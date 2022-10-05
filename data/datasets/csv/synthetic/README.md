To obtain the exact same synthetic datasets that were used by us, follow this link: https://doi.org/10.7910/DVN/GCREVU

Please note, the original format is CSV. The data source, however, changes this to .tab sometimes. Use the available
option in the drop-down-menu to download CSV files.

Alternatively, use the script provided to generate new synthetic datasets.

# Generators for Synthetic Datasets

Four different generators for synthetic datasets exist. The first two generators create random datasets in which (1) the
values of attributes are uniformly distributed, and (2) the values of attributes are normally distributed. Additionally,
two generators were created, which create non-random synthetic data. These generators first compute all possible
attribute combinations of maximal length and then distribute the target value such that the resulting SD dataset is
either interesting or uninteresting. The uninteresting generator (3) creates data by generating multiple instances of
the same combination such that each has a target share of 50%, which is equal to the overall target share. The
interesting generator (4) creates a dataset with an overall target share different (higher or lower) to the target share
of specific subsets of the combinations. 