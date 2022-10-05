# Script to load all CSV files and convert them to ARFF
import arff
from general_utils.data_loader import all_data_paths, load_preprocessed_dataset_csv
from config import arff_files_path
import os


def save_csv_as_arff(df, name):
    # Following usage as describe by https://stackoverflow.com/a/57109015
    attributes = [(c, df[c].unique().astype(str).tolist()) for c in df.columns.values[:-1]]
    attributes += [("class", df[df.columns[-1]].unique().astype(str).tolist())]
    t = df.columns[-1]
    arff_data = [df.loc[i].values[:-1].tolist() + [df[t].loc[i]] for i in range(df.shape[0])]

    arff_dic = {
        "attributes": attributes,
        "data": arff_data,
        "relation": "data",
        "description": ''
    }

    with open(os.path.join(arff_files_path, name + ".arff"), "w", encoding="utf8") as f:
        arff.dump(arff_dic, f)


# Get path for all datasets
data_paths = all_data_paths()

# Go over each path
nr_paths = len(data_paths)
for i, path_to_data in enumerate(data_paths):
    data, data_name = load_preprocessed_dataset_csv(path_to_data)
    save_csv_as_arff(data, data_name)
