import os
import pandas as pd
import glob
from scipy.io import arff
from config import real_world_csv_files_path, synthetic_csv_files_path, arff_files_path, \
    sd_experiment_settings_path


# ------ Data Loaders
def get_sd_exp_data():
    return pd.read_csv(sd_experiment_settings_path)


def load_preprocessed_dataset_csv(path):
    # Read data
    df = pd.read_csv(path)

    # Get basename with .csv
    name = os.path.basename(path)[:-4]

    df = df.applymap(str)

    return df, name


def load_preprocessed_dataset_arff(path):
    # Using arff data for now across all implementations thus this function instead of the above
    data_name = os.path.basename(path)[:-5]
    df = pd.DataFrame(arff.loadarff(path)[0])

    for col in df.columns:
        df[col] = df[col].str.decode("utf-8")
    return df, data_name


def load_dataset(path, use_arff):
    if use_arff:
        return load_preprocessed_dataset_arff(path)
    else:
        return load_preprocessed_dataset_csv(path)


# ------ Collect Data Paths
def all_data_paths():
    check_path = "{}/*".format(real_world_csv_files_path)
    real_world_data_paths = glob.glob(check_path)
    check_path = "{}/*.csv".format(synthetic_csv_files_path)
    synthetic_data_paths = glob.glob(check_path)

    all_paths = real_world_data_paths + synthetic_data_paths
    return all_paths


def all_data_paths_arff():
    check_path = "{}/*.arff".format(arff_files_path)
    all_paths = glob.glob(check_path)

    return all_paths


def get_name_to_path_dict(use_arff):
    if use_arff:
        return {os.path.basename(path)[:-5]: path for path in all_data_paths_arff()}
    else:
        return {os.path.basename(path)[:-4]: path for path in all_data_paths()}
