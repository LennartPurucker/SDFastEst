import matplotlib.pyplot as plt
from config import build_full_path
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error

base_path_plot = build_full_path("data/plots")


def plot_comparison(ground_truth_data, algorithm_name, plot_ei_est=False, upper_limit=False, plot_sdfastest_esgs=False):
    # OLD PLOT VERSION, perhaps more easily human-readable but needs mor explaining, not used in paper

    # Sort by ids
    ground_truth_data = ground_truth_data.sort_values(algorithm_name + "-sgs_evaluated")
    ground_truth_data = ground_truth_data.reset_index(drop=True)

    rel_cols_postfix_list = ["-sgs_evaluated", "-e_est"]
    # relevant columns
    if plot_ei_est:
        rel_cols_postfix_list.append("-ei_est")
    if plot_sdfastest_esgs:
        rel_cols_postfix_list.append("-sdfastest-esgs")
    if upper_limit:
        rel_cols_postfix_list.append("-ub")

    rel_cols = [algorithm_name + postfix for postfix in rel_cols_postfix_list]

    # Plot
    fig, ax = plt.subplots()
    ground_truth_data[rel_cols].plot(ax=ax, style=["-", "-.", ":"], figsize=(8, 6))  # , ,"-", "-.", "--", "-.", ":", )

    plt.ylabel("Number of Subgroups")
    plt.xlabel("SD Task IDs (sorted) \n")
    ax.set_yscale("log")
    fig.tight_layout()

    plt.show()


def scatter_error_plot_multi(data, algorithm_name, experiment_type, plot_without_comp=False):
    # Rel cols
    col_y_true = algorithm_name + "-sgs_evaluated"
    col_x_1 = algorithm_name + "-ub"
    col_x_2 = algorithm_name + "-e_est"
    col_x_3 = algorithm_name + "-ei_est"

    fig, ax = plt.subplots()
    # Plot all data
    # added facecolor and edge colors here to make the circle (feel) empty inside
    plt.scatter(y=data[col_x_1], x=data[col_y_true], marker="o", facecolors='none', edgecolors='blue')  # , alpha=0.5)
    plt.scatter(y=data[col_x_2], x=data[col_y_true], marker="o", facecolors='none', edgecolors='orange')  # , alpha=0.5)
    if plot_without_comp:
        plt.scatter(y=data[col_x_3], x=data[col_y_true], marker="o", alpha=0.5)
    # Give color codes and names
    classes = ["Worst Case Upper Bound", "SDFastEst"]
    if plot_without_comp:
        classes.append("SDFastEst without Compensation")

    plt.ylabel("Predicted Number of Evaluated Subgroups")
    plt.xlabel("True Number of Evaluated Subgroups")
    # Log scale
    ax.set_yscale('log')
    ax.set_xscale('log')
    # Add perfect predict line
    max_predict_x = max(data[col_y_true]) * 1.2
    ax.axline([0, 0], [max_predict_x, max_predict_x], color="red")
    classes.append("Perfect Prediction")
    plt.legend(labels=classes)
    fig.tight_layout()
    plt.savefig(
        os.path.join(base_path_plot, "prediction_compare_scatter/{}_{}.pdf".format(algorithm_name, experiment_type)))
    # plt.show()


def get_log_mae(y_true, y_pred):
    y_true_log = np.nan_to_num(np.log2(y_true), nan=0, posinf=0, neginf=0)
    y_pred_log = np.nan_to_num(np.log2(y_pred), nan=0, posinf=0, neginf=0)
    return mean_absolute_error(y_true_log, y_pred_log)


def sgs_types_plot(algorithm_name):
    data = pd.read_csv(build_full_path("data/algorithm_data/run_data_{}.csv".format(algorithm_name)))
    data = data[data[algorithm_name + "-failed"] == 0].reset_index(drop=True)

    data = data.sort_values(algorithm_name + "-sgs_evaluated")
    data = data.reset_index(drop=True)

    # relevant columns
    rel_cols = [algorithm_name + "-sgs_evaluated", algorithm_name + "-sgs_eval_interesting",
                algorithm_name + "-sgs_quality_interesting", algorithm_name + "-sgs_top_k_interesting"]

    # Plot
    fig, ax = plt.subplots()
    data[rel_cols].plot(ax=ax)
    plt.ylabel("Number of Subgroups")  # Number of Subgroups
    plt.xlabel("SD Task IDs (sorted) \n")
    ax.set_yscale("log")
    L = plt.legend()
    L.get_texts()[0].set_text("Evaluated Subgroups")
    L.get_texts()[1].set_text("Evaluation Interesting Subgroups")
    L.get_texts()[2].set_text("Quality Interesting Subgroups")
    L.get_texts()[3].set_text("Top-k Interesting Subgroups")
    fig.tight_layout()
    plt.savefig(os.path.join(base_path_plot, "sgs_types/{}.pdf".format(algorithm_name)))
    # plt.show()


def get_all_results():
    algo_names = ["Apriori_DP (PS)", "DFS_BSD_DP (PS)", "Apriori_DP_fixed (PS)", "Apriori_DP_fixed_qf05 (PS)",
                  "DFS_BSD_DP_qf05 (PS)", "DFS_BSD_DP_FC (PS)", "DFS_BSD_DP_FC_qf05 (PS)"]
    experiment_types = ["default", "default_timeout", "dynamic", "dynamic_timeout"]

    # Init Result DFs
    error_table_headers = ["Algorithm", "MAE - SDFastEst (100)", "MAE - SDFastEst (100+20s)",
                           "MAE - SDFastEst (dynamic)",
                           "MAE - SDFastEst (dynamic+20s)", "MAE - Depth Upper Bound"]
    error_table_result = pd.DataFrame(columns=error_table_headers)
    eff_table_headers = ["Algorithm", "Minutes - SDFastEst (100)", "Minutes - SDFastEst (100+20s)",
                         "Minutes - SDFastEst (dynamic)",
                         "Minutes - SDFastEst (dynamic+20s)", "Minutes - Solving all SD Tasks"]
    eff_table_result = pd.DataFrame(columns=eff_table_headers)

    # Get Plots
    for algo_name in algo_names:
        # Get SG Types Plot
        sgs_types_plot(algo_name)

        name = "Error"
        log10_mae_ub = "Error"
        algo_time = "Error"
        exps_res = {}
        for experiment_type in experiment_types:
            # Get specific Experiment Data
            experiment_data = pd.read_csv(
                build_full_path("data/results/res_{}_{}.csv".format(algo_name, experiment_type)))

            # TO REMOVE SYNTH DATA if needed: experiment_data = experiment_data[~experiment_data["dataset_name"].str.contains("Synthetic")].reset_index(drop=True)
            # Old compare plot: plot_comparison(experiment_data, algo_name, upper_limit=True)

            scatter_error_plot_multi(experiment_data, algo_name, experiment_type)

            # Can be overwritten as it is the same for each element in this loop
            name = "{} (n={})".format(algo_name, len(experiment_data))
            log10_mae_ub = get_log_mae(experiment_data[algo_name + "-sgs_evaluated"],
                                       experiment_data[algo_name + "-ub"])
            algo_time = sum(experiment_data[algo_name + "-real_time_taken"]) / 60

            # Append
            exps_res[experiment_type] = get_log_mae(experiment_data[algo_name + "-sgs_evaluated"],
                                                    experiment_data[algo_name + "-e_est"])
            exps_res[experiment_type + "-eff"] = sum(experiment_data[algo_name + "-time_to_est"]) / 60

        # Build row for algorithm
        error_row_res = (name, exps_res["default"], exps_res["default_timeout"], exps_res["dynamic"],
                         exps_res["dynamic_timeout"], log10_mae_ub)
        eff_row_res = (name, exps_res["default-eff"], exps_res["default_timeout-eff"], exps_res["dynamic-eff"],
                       exps_res["dynamic_timeout-eff"], algo_time)

        # Add row to DF (workaround idea from https://stackoverflow.com/a/55541543)
        error_table_result.loc[len(error_table_result)] = error_row_res
        eff_table_result.loc[len(eff_table_result)] = eff_row_res

    return error_table_result, eff_table_result


if __name__ == "__main__":
    error, eff = get_all_results()

    # Some Stats - Speed Up
    eff["rel% - SDFastEst (100)"] = eff["Minutes - SDFastEst (100)"] / eff["Minutes - Solving all SD Tasks"]
    eff["rel% - SDFastEst (100+20s)"] = eff["Minutes - SDFastEst (100+20s)"] / eff["Minutes - Solving all SD Tasks"]

    # Printing values used in paper
    eff_1 = eff.iloc[[0, 1, 2, 5]]
    error_1 = error.iloc[[0, 1, 2, 5]]
    print("Mean Eff SDFastEst (100)", eff_1["rel% - SDFastEst (100)"].mean())
    print("Mean Eff SDFastEst (100+20s)", eff_1["rel% - SDFastEst (100+20s)"].mean())

    print("Mean MAE SDFastEst (100)", error_1["MAE - SDFastEst (100)"].mean())
    print("Mean MAE SDFastEst (100+20s)", error_1["MAE - SDFastEst (100+20s)"].mean())
    print("Mean MAE UB (100)", error_1["MAE - Depth Upper Bound"].mean())
