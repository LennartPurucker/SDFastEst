import os
import pandas as pd
from data.datasets.data_processing.dataset_functions import preprocessing_utils

# ---- CHANGED REQUIRED
UCI = ""  # Add the path to the downloaded UCI datasets here


# ---- CHANGED REQUIRED

def get_uci_iris():
    path = os.path.join(UCI, "iris.data")
    df = pd.read_csv(path, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "Iris Class"])

    # Preprocess Attributes: Numeric to Categorical
    for attribute in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        df[attribute] = preprocessing_utils.discretize(df[attribute])

    # Meta Info
    dataset_name = "Iris (UCI)"

    return df, dataset_name


def get_uci_adult():
    # Read datasets
    path_data = os.path.join(UCI, "adult.data")
    path_train = os.path.join(UCI, "adult.test")

    # Merge and make dataframe
    adult_col = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                 "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                 "income"]
    # Need to skip space after delimiter and first row of train data
    df_data = pd.read_csv(path_data, index_col=None, names=adult_col, skipinitialspace=True)
    df_train = pd.read_csv(path_train, index_col=None, names=adult_col, skiprows=1, skipinitialspace=True)
    df_train["income"] = df_train["income"].replace(["<=50K.", ">50K."],
                                                    ["<=50K", ">50K"])  # Remove period at end of data
    df = pd.concat([df_data, df_train], axis=0, ignore_index=True)

    # Preprocess
    # Drop numeric attributes
    df = df.drop(columns=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"])
    preprocessing_utils.handle_missing_values(df, "?")

    # Meta Info
    dataset_name = "Adult (UCI)"

    return df, dataset_name


def get_uci_bcw():
    path_data = os.path.join(UCI, "breast-cancer-wisconsin.data")

    # Merge and make dataframe
    col = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
           "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli",
           "Mitoses", "Class 2-benign/4-malignant"]
    df = pd.read_csv(path_data, names=col)
    df = df.drop(columns=["Sample code number"])

    preprocessing_utils.handle_missing_values(df, "?")

    # Meta Info
    dataset_name = "Breast Cancer Wisconsin (UCI)"

    return df, dataset_name


def get_uci_census():
    # Read datasets
    path_data = os.path.join(UCI, "census-income.data")
    path_train = os.path.join(UCI, "census-income.test")

    # Merge and make dataframe
    pre_col = ["age", "class of worker", "detailed industry recode", "detailed occupation recode", "education",
               "wage per hour", "enroll in edu inst last wk", "marital stat", "major industry code",
               "major occupation code", "race", "hispanic origin", "sex", "member of a labor union",
               "reason for unemployment", "full or part time employment stat", "capital gains", "capital losses",
               "dividends from stocks", "tax filer stat", "region of previous residence", "state of previous residence",
               "detailed household and family stat", "detailed household summary in household", "X",
               "migration code-change in msa", "migration code-change in reg", "migration code-move within reg",
               "live in this house 1 year ago", "migration prev res in sunbelt", "num persons worked for employer",
               "family members under 18", "country of birth father", "country of birth mother", "country of birth self",
               "citizenship", "own business or self employed", "fill inc questionnaire for veteran's admin",
               "veterans benefits", "weeks worked in year", "year", "income"]

    df_data = pd.read_csv(path_data, index_col=False, names=pre_col, skipinitialspace=True)
    df_train = pd.read_csv(path_train, index_col=False, names=pre_col, skipinitialspace=True)
    df = pd.concat([df_data, df_train], axis=0, ignore_index=True)
    df["income"] = df["income"].replace(["- 50000.", "50000+."], ["- 50000", "50000+"])

    # Drop numeric attributes
    df = df.drop(columns=["age", "wage per hour", "capital gains", "capital losses", "dividends from stocks",
                          "num persons worked for employer", "weeks worked in year", "year", "X",
                          "migration code-change in msa", "migration code-change in reg",
                          "migration code-move within reg", "migration prev res in sunbelt"])

    preprocessing_utils.handle_missing_values(df, "?")

    # Meta Info
    dataset_name = "Census-Income (KDD) (UCI)"

    return df, dataset_name


def get_uci_german_credit():
    path = os.path.join(UCI, "german.data")
    df = pd.read_csv(path, delim_whitespace=True,
                     names=["Status of existing checking account", "Duration in month", "Credit history", "Purpose",
                            "Credit amount", "Savings account/bonds", "Present employment since",
                            "Installment rate in percentage of disposable income", "Personal status and sex",
                            "Other debtors / guarantors", "Present residence since", "Property", "Age in years",
                            "Other installment plans", "Housing", "Number of existing credits at this bank", "Job",
                            "Number of people being liable to provide maintenance for", "Telephone", "foreign worker",
                            "class"])

    numerical_col = ["Duration in month", "Credit amount", "Installment rate in percentage of disposable income",
                     "Present residence since", "Age in years", "Number of existing credits at this bank",
                     "Number of people being liable to provide maintenance for"]
    for attribute in numerical_col:
        df[attribute] = preprocessing_utils.discretize(df[attribute])

    # Meta Info
    dataset_name = "German Credit (UCI)"

    return df, dataset_name


def get_uci_heart():
    path = os.path.join(UCI, "heart.dat")
    df = pd.read_csv(path, delim_whitespace=True,
                     names=["age", "sex", "chest pain type", "resting blood pressure", "serum cholesterol in mg/dl",
                            "fasting blood sugar > 120 mg/dl", "resting electrocardiographic results",
                            "maximum heart rate achieved", "exercise induced angina", "oldpeak",
                            "the slope of the peak", "number of major vessels", "thal", "Absence/Presence"])

    numerical_col = ["age", "resting blood pressure", "serum cholesterol in mg/dl", "maximum heart rate achieved",
                     "oldpeak", "number of major vessels"]

    for attribute in numerical_col:
        df[attribute] = preprocessing_utils.discretize(df[attribute])

    # Meta Info
    dataset_name = "Heart (Statlog) (UCI)"

    return df, dataset_name


def get_uci_krkp():
    path = os.path.join(UCI, "kr-vs-kp.data")
    df = pd.read_csv(path,
                     names=["bkblk", "bknwy", "bkon8", "bkona", "bkspr", "bkxbq", "bkxcr", "bkxwp", "blxwp", "bxqsq",
                            "cntxt", "dsopp", "dwipd",
                            "hdchk", "katri", "mulch", "qxmsq", "r2ar8", "reskd", "reskr", "rimmx", "rkxwp", "rxmsq",
                            "simpl", "skach", "skewr",
                            "skrxp", "spcop", "stlmt", "thrsk", "wkcti", "wkna8", "wknck", "wkovl", "wkpos", "wtoeg",
                            "class"])

    # Meta Info
    dataset_name = "Chess (King-Rook vs. King-Pawn) (UCI)"

    return df, dataset_name


def get_uci_mushroom():
    path = os.path.join(UCI, "agaricus-lepiota.data")
    df = pd.read_csv(path,
                     names=["class", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment",
                            "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
                            "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
                            "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type",
                            "spore-print-color", "population", "habitat"])
    # Reorder columns s.t. class is at the end
    df = df[["cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment",
             "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
             "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
             "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type",
             "spore-print-color", "population", "habitat", "class"]]

    # Meta Info
    dataset_name = "Mushroom (UCI)"

    return df, dataset_name


def get_uci_tictactoe():
    path = os.path.join(UCI, "tic-tac-toe.data")
    df = pd.read_csv(path,
                     names=["top-left", "top-middle", "top-right", "middle-left", "middle-middle", "middle-right",
                            "bottom-left", "bottom-middle", "bottom-right", "Class"])

    # Meta Info
    dataset_name = "Tic-Tac-Toe Endgame (UCI)"

    return df, dataset_name


def get_uci_vote():
    path = os.path.join(UCI, "house-votes-84.data")
    df = pd.read_csv(path,
                     names=["class", "handicapped-infants", "water-project-cost-sharing",
                            "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid",
                            "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras",
                            "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending",
                            "superfund-right-to-sue", "crime", "duty-free-exports",
                            "export-administration-act-south-africa"])
    # Reorder columns s.t. class is at the end
    df = df[["handicapped-infants", "water-project-cost-sharing",
             "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid",
             "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras",
             "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending",
             "superfund-right-to-sue", "crime", "duty-free-exports",
             "export-administration-act-south-africa", "class"]]

    # Meta Info
    dataset_name = "Congressional Voting Records (UCI)"

    return df, dataset_name


def get_uci_seismic_bumps():
    path = os.path.join(UCI, "seismic-bumps.data")
    df = pd.read_csv(path,
                     names=["seismic", "seismoacoustic", "shift", "genergy", "gpuls", "gdenergy", "gdpuls", "ghazard",
                            "nbumps", "nbumps2", "nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89",
                            "energy", "maxenergy", "class"])

    numerical_col = ["genergy", "gpuls", "gdenergy", "gdpuls", "nbumps", "nbumps2", "nbumps3", "nbumps4", "nbumps5",
                     "nbumps6", "nbumps7", "nbumps89", "energy", "maxenergy"]

    for attribute in numerical_col:
        df[attribute] = preprocessing_utils.discretize(df[attribute])

    dataset_name = "Seismic-Bumps (UCI)"

    return df, dataset_name


def get_uci_balance():
    path = os.path.join(UCI, "balance-scale.data")
    df = pd.read_csv(path,
                     names=["class", "lw", "ld", "rw", "rd"])
    # Reorder columns s.t. class is at the end
    df = df[["lw", "ld", "rw", "rd", "class"]]

    # Meta Info
    dataset_name = "Balance (UCI)"

    return df, dataset_name


def get_uci_default_of_credit_card_clients():
    path = os.path.join(UCI, "default_of_credit_card_clients.csv")
    df = pd.read_csv(path, delimiter=";", skiprows=[1], index_col=0)

    for attribute in df.columns[:-1]:
        df[attribute] = preprocessing_utils.discretize(df[attribute],
                                                       force_string_for_small=True, str_threshold=11)

    dataset_name = "Default of Credit Card"
    return df, dataset_name


def get_uci_in_vehicle_coupon_recommendation():
    path = os.path.join(UCI, "in-vehicle-coupon-recommendation.csv")
    df = pd.read_csv(path)

    # Handle missing values
    axis = 1  # drop columns
    preprocessing_utils.handle_missing_values(df, "nan", axis=axis)

    dataset_name = "Vehicle Coupon Recommendation"
    return df, dataset_name


def get_uci_online_shoppers_intention():
    path = os.path.join(UCI, "online_shoppers_intention.csv")
    df = pd.read_csv(path)

    numerical_cols = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
                      "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
                      "SpecialDay"]

    for attribute in numerical_cols:
        df[attribute] = preprocessing_utils.discretize(df[attribute],
                                                       force_string_for_small=True)

    dataset_name = "Online Shoppers Intention"
    return df, dataset_name


def get_uci_covtype():
    path = os.path.join(UCI, "covtype.data")
    col_names = [str(i) for i in range(55)]
    df = pd.read_csv(path, names=col_names)

    for attribute in df.columns[:-1]:
        df[attribute] = preprocessing_utils.discretize(df[attribute],
                                                       force_string_for_small=True, str_threshold=2)

    dataset_name = "Covertype"
    return df, dataset_name


def get_uci_divorce():
    path = os.path.join(UCI, "divorce.csv")
    df = pd.read_csv(path, delimiter=";")

    dataset_name = "Divorce"
    return df, dataset_name


def get_uci_diabetic():
    path = os.path.join(UCI, "diabetic_data.csv")
    df = pd.read_csv(path)

    # Drop unique identifiers
    df = df.drop(columns=["encounter_id", "patient_nbr"])

    # Handle missing values
    preprocessing_utils.handle_missing_values(df, "?", axis=1)

    # Handle numeric attributes
    numeric_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                    "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    for attribute in numeric_cols:
        df[attribute] = preprocessing_utils.discretize(df[attribute])

    dataset_name = "Diabetic"
    return df, dataset_name
