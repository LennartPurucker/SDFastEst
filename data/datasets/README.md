# Datasets

`csv/real_world` and `csv/syntethic` contains information on how to get the same datasets that we used.

## General Preprocessing Remarks

All datasets were preprocessed by dropping rows if a row contains a missing value. Moreover, quantile-based
discretization with 4 bins and dropping duplicates was used to encode numeric data as categorical data. For all
hand-selected datasets, this was applied to all found numeric attributes. For datasets from the literature, this was
applied to numeric attributes with more than $5$ unique values. Finally, all values of descriptive attributes were
replaced by categorical codes, i.e., ascending numbers. All datasets were prepossessed and then exported to CSV and ARFF
files. In any file, the last column corresponds to the target attribute. 