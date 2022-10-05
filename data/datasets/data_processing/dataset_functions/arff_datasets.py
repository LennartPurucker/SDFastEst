import glob
from scipy.io import arff
import pandas as pd
import os
from data.datasets.data_processing.dataset_functions import preprocessing_utils

# ---- CHANGED REQUIRED
ARFF_DIR = ""  # Add the path to the downloaded ARFF datasets here

# ---- CHANGED REQUIRED

DROP_COL = ["thyroid_hypothyroid", "secom", "internet", "audiology_std", "heart_hungarian", "heart_va"]


def get_arff_datasets():
    # Get all arff files
    check_path = "{}/*.arff".format(ARFF_DIR)
    all_arff_datasets_paths = glob.glob(check_path)

    # Get datasets
    arff_datasets = []
    for arff_dataset_path in all_arff_datasets_paths:
        data_name = os.path.basename(arff_dataset_path[:-5])

        # Transform to pandas
        arff_data = arff.loadarff(arff_dataset_path)
        df = pd.DataFrame(arff_data[0])

        # have to do this here to make sure all columns are parsed to string for handling missing values
        for arff_type, col_name in zip(arff_data[1].types(), df.columns):
            if arff_type != "numeric":
                df[col_name] = df[col_name].str.decode("utf-8")

        # handle missing values - if existing, drop all rows with missing values
        # Define which axis to drop
        axis = 0  # drop rows
        if data_name in DROP_COL:
            axis = 1  # drop columns

        # Handle missing values
        preprocessing_utils.handle_missing_values(df, "?", axis=axis)
        df.dropna(inplace=True, axis=axis)

        # Handle numeric data now since missing values are dropped
        for arff_type, col_name in zip(arff_data[1].types(), df.columns):
            if arff_type == "numeric":
                df[col_name] = preprocessing_utils.discretize(df[col_name], force_string_for_small=True)

        # Add to list
        arff_datasets.append((df, data_name))

    return arff_datasets
