import os
import pandas as pd
from data.datasets.data_processing.dataset_functions import preprocessing_utils

# ---- CHANGED REQUIRED
OTHER = ""  # Add the path to the downloaded OTHER datasets here


# ---- CHANGED REQUIRED


def get_eosl_income_data():
    path = os.path.join(OTHER, "marketing.data")
    df = pd.read_csv(path,
                     names=["ANNUAL INCOME OF HOUSEHOLD", "SEX", "MARITAL STATUS", "AGE", "EDUCATION", "OCCUPATION",
                            "HOW LONG LIVED", "DUAL INCOMES (IF MARRIED)", "PERSONS IN YOUR HOUSEHOLD",
                            "PERSONS IN HOUSEHOLD UNDER 18", "HOUSEHOLDER STATUS", "TYPE OF HOME",
                            "ETHNIC CLASSIFICATION", "WHAT LANGUAGE"], delim_whitespace=True)
    # Reorder columns s.t. class is at the end
    df = df[["SEX", "MARITAL STATUS", "AGE", "EDUCATION", "OCCUPATION", "HOW LONG LIVED", "DUAL INCOMES (IF MARRIED)",
             "PERSONS IN YOUR HOUSEHOLD", "PERSONS IN HOUSEHOLD UNDER 18", "HOUSEHOLDER STATUS", "TYPE OF HOME",
             "ETHNIC CLASSIFICATION", "WHAT LANGUAGE", "ANNUAL INCOME OF HOUSEHOLD"]]

    # Handle missing values
    preprocessing_utils.handle_missing_values(df, "nan")

    # Meta Info
    dataset_name = "Marketing (EoSL)"

    return df, dataset_name


def get_kaggle_stroke():
    path = os.path.join(OTHER, "healthcare-dataset-stroke-data.csv")
    df = pd.read_csv(path)

    df = df.drop(columns=["id"])

    # Handle missing values
    preprocessing_utils.handle_missing_values(df, "nan")

    numerical_col = ["age", "avg_glucose_level", "bmi"]

    for attribute in numerical_col:
        df[attribute] = preprocessing_utils.discretize(df[attribute])

    # Meta Info
    dataset_name = "Stroke (Kaggle)"

    return df, dataset_name


def get_dry_beans():
    path = os.path.join(OTHER, "dry_beans.csv")
    df = pd.read_csv(path, delimiter=";", index_col=0, decimal=",")

    for attribute in df.columns[:-1]:
        df[attribute] = preprocessing_utils.discretize(df[attribute])

    # Meta Info
    dataset_name = "Dry Beans"
    return df, dataset_name
