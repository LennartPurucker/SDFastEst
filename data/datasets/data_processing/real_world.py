# A script used to preprocess the real-world dataset as described in the paper.
# You have to download the original datasets and give the files "uci_datasets", "other_datasets", and "arff_datasets"
#  in the directory "./dataset_functions" the path to the downloaded file directories.
from config import real_world_csv_files_path
import re
import os
import csv
from data.datasets.data_processing.dataset_functions import uci_datasets, other_datasets, arff_datasets


# --- Save Utils
def save_collection_of_datasets(datasets):
    for data, name in datasets:
        save_dataset(data, name)


def save_dataset(data, name):
    # Clean Name for Saving
    name_without_space = name.replace(" ", "")
    name = re.sub('[^\w\-_\. ]', '_', name_without_space)

    data.to_csv(os.path.join(real_world_csv_files_path, name + ".csv"), index=False, quoting=csv.QUOTE_ALL)


def transform_to_numbers_dataset(data):
    """Function to transform preprocessed categorical real-world datasets into only numbers / categorical codes"""
    columns = data.columns

    # Transform all string values to numeric codes
    for col_name in columns:
        if col_name == columns[-1]:
            continue
        if data[col_name].dtype == 'object':
            data[col_name] = data[col_name].astype('category')
            data[col_name] = data[col_name].cat.codes
            continue

        data[col_name] = data[col_name].astype(int)

    # Build dict to replace names
    name_replace = {}
    for replace_index, col in enumerate(columns):
        if col == columns[-1]:
            replace_index = "class"  # mark last column as class as it is not used anyways
        name_replace[col] = str(replace_index)

    # Replace columns
    data = data.rename(columns=name_replace)

    # Make all values to string
    data = data.applymap(str)

    return data


def load_datasets():
    datasets = [
        uci_datasets.get_uci_iris(),
        uci_datasets.get_uci_adult(),
        uci_datasets.get_uci_bcw(),
        uci_datasets.get_uci_german_credit(),
        uci_datasets.get_uci_heart(),
        uci_datasets.get_uci_krkp(),
        uci_datasets.get_uci_mushroom(),
        uci_datasets.get_uci_tictactoe(),
        uci_datasets.get_uci_vote(),
        uci_datasets.get_uci_seismic_bumps(),
        uci_datasets.get_uci_balance(),
        uci_datasets.get_uci_census(),
        other_datasets.get_eosl_income_data(),
        other_datasets.get_kaggle_stroke(),
        uci_datasets.get_uci_diabetic(),
        uci_datasets.get_uci_divorce(),
        uci_datasets.get_uci_covtype(),
        other_datasets.get_dry_beans(),
        uci_datasets.get_uci_online_shoppers_intention(),
        uci_datasets.get_uci_in_vehicle_coupon_recommendation(),
        uci_datasets.get_uci_default_of_credit_card_clients()
    ]
    datasets += arff_datasets.get_arff_datasets()

    # Cast to list instead of tuple pairs for processing
    datasets = [list(data_pair) for data_pair in datasets]

    # Idea: Transform to number dataset to save some memory when numbers instead of string selectors
    for data_pair in datasets:
        data_pair[0] = transform_to_numbers_dataset(data_pair[0])

    return datasets


if __name__ == "__main__":
    real_world_datasets = load_datasets()

    # Force all to be columns to be string
    for data_tuple in real_world_datasets:
        data_tuple[0] = data_tuple[0].applymap(str)

    save_collection_of_datasets(real_world_datasets)
