# Example of using the results of our novel method for runtime prediction.
# This is the regression tasks of using the search space size estimate and the number of instance to predict the
#   real (wall-clock) runtime of the selected algorithm for an instance.
# Here, synthetic datasets are not used to speed up the process and enable it for computers with not enough memory
#   for the large synthetic datasets.
import pandas as pd
from general_utils.sd_utils import load_sd_dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def produce_data_for_rt_est(algorithm_name, remove_synthetic_data=True):
    # Get estimation data
    df = pd.read_csv(r"../data/results/res_{}_default_timeout.csv".format(algorithm_name))

    if remove_synthetic_data:
        # Remove Synthetic data
        df = df[~df["dataset_name"].str.contains("Synthetic")].reset_index(drop=True)

    # Get number of instances for all entries
    instance_counts = []
    print("Loading Datasets to get n_instances")
    for index, sd_exp in df.iterrows():
        print("{}/{}".format(index + 1, len(df)))
        dataset_name = sd_exp["dataset_name"]
        sd_data, sd_data_name = load_sd_dataset(dataset_name)
        instance_counts.append(len(sd_data))
    df["n_instances"] = pd.Series(instance_counts)

    # Select relevant subset of data and re-name
    df = df[[algorithm_name + "-e_est", "n_instances", algorithm_name + "-real_time_taken", "dataset_name"]]
    return df.rename(columns={algorithm_name + "-e_est": "sss_est", algorithm_name + "-real_time_taken": "runtime"})


def get_log_2_mae(y_true, y_predicted):
    y_true_log2 = np.nan_to_num(np.log2(y_true), nan=0, posinf=0, neginf=0)
    y_pred_log2 = np.nan_to_num(np.log2(y_predicted), nan=0, posinf=0, neginf=0)
    return mean_absolute_error(y_true_log2, y_pred_log2)


def plot_comparison(y_true, y_predicted):
    # Sort by ids
    data = pd.DataFrame()
    data["true_runtime"] = y_true
    data["predicted_runtime"] = y_predicted
    data = data.sort_values("true_runtime").reset_index(drop=True)
    rel_cols = ["true_runtime", "predicted_runtime"]

    # Plot
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    fig.set_figwidth(6)
    data[rel_cols].plot(ax=ax)
    plt.ylabel("Runtime")
    plt.xlabel("SD Task IDs (sorted) \n")
    ax.set_yscale("log")

    L = plt.legend()
    L.get_texts()[0].set_text("True Runtime")
    L.get_texts()[1].set_text("Predicted Runtime")

    fig.tight_layout()
    plt.savefig("../data/plots/applications/{}.pdf".format("Runtime_Meta_Learning_Apriori_fixed"))

    plt.show()


def scatter_error_plot(y_true, y_predicted):
    # Error Scatter plot
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_yscale('log')
    ax.set_xscale('log')
    max_predict = max(max(y_true), max(y_predicted)) * 1.2
    ax.scatter(y_true, y_predicted)
    plt.ylabel("Predicted Runtime")
    plt.xlabel("True Runtime")
    ax.axline([0, 0], [max_predict, max_predict], color="red")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Get Meta-Learning Data with 2 CMs (sss_estimate and and number of instances) and 1 Target (wall-clock runtime)
    # Loads all datasets and thus takes sometime...
    meta_learning_data = produce_data_for_rt_est("Apriori_DP (PS)", remove_synthetic_data=True)
    # Shuffle data
    meta_learning_data = meta_learning_data.sample(frac=1, random_state=1234).reset_index(drop=True)

    # Get Train / Test Split based on groups such that we do not "leak" datasets
    # In other words, we first "split" based on the dataset
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1234)
    gss.get_n_splits()
    feature_data = meta_learning_data[["sss_est", "n_instances"]]
    target_data = meta_learning_data["runtime"]
    train_idx, test_idx = list(gss.split(feature_data, target_data, meta_learning_data[["dataset_name"]]))[0]

    X_train = feature_data.iloc[train_idx]
    y_train = target_data.iloc[train_idx]
    X_test = feature_data.iloc[test_idx]
    y_test = target_data.iloc[test_idx]

    print(len(X_test))
    # Use basic RF to do meta-learning
    # Fit
    print("Predict")
    regr = RandomForestRegressor(random_state=1234)
    regr.fit(X_train, y_train)
    # Predict
    y_pred = regr.predict(X_test)

    # Score with log2 mae to make the results more comprehensible and comparable
    print("LOG2_MAE: ", get_log_2_mae(y_test, y_pred))
    # Show Plot
    print("Plot")
    scatter_error_plot(y_test, y_pred)
    plot_comparison(y_test, y_pred)
