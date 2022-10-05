import pandas as pd
from general_utils.sd_utils import WRAcc, SimpleBinomialQF
from SDFastEst.sdfastest_runner import run_sdfastest
from general_utils.result_plotting import plot_comparison, get_log_mae
from SDFastEst.sampling_profiles import SPECIFIC_SAMPLING_PROFILES


def load_new_esg_data(algorithm_name):
    df = pd.read_csv("./data/algorithm_data/run_data_{}.csv".format(algorithm_name))
    df = df[df[algorithm_name + "-failed"] == 0].reset_index(drop=True)
    return df


def compare_for_algorithms(input_estimation_data, current_sample_settings, file_postfix):
    for to_eval_setting in input_estimation_data:
        compare_for_algorithm(*to_eval_setting, input_sample_settings=current_sample_settings,
                              file_postfix=file_postfix)


def compare_for_algorithm(algorithm_name, algorithm_profile_name, quality_function, input_sample_settings=None,
                          file_postfix=""):
    if input_sample_settings is None:
        input_sample_settings = {"timeout": None}

    algo_ground_truth_data = load_new_esg_data(algorithm_name)

    # ------ Get sampling settings for algorithm
    if algorithm_profile_name in ["PS-DFS"]:
        input_sample_settings.update(SPECIFIC_SAMPLING_PROFILES["DFS"])
    else:
        if input_sample_settings["timeout"]:
            input_sample_settings.update(SPECIFIC_SAMPLING_PROFILES["Apriori-Pathwise"])
        else:
            input_sample_settings.update(SPECIFIC_SAMPLING_PROFILES["Apriori-Levelwise"])

    # ------ Execute
    run_sdfastest(algo_ground_truth_data, quality_function, algorithm_profile_name, algorithm_name,
                  input_sample_settings)

    # ----- Plot results
    plot_comparison(algo_ground_truth_data, algorithm_name)
    # ----- Print Data
    algo_time = sum(algo_ground_truth_data[algorithm_name + "-real_time_taken"])
    sdfastest_time = sum(algo_ground_truth_data[algorithm_name + "-time_to_est"])
    data_path = "./data/results/res_{}_{}.csv".format(algorithm_name, file_postfix)
    log_mae = get_log_mae(algo_ground_truth_data[algorithm_name + "-sgs_evaluated"],
                          algo_ground_truth_data[algorithm_name + "-e_est"])

    print("SD Algorithm-Time", algo_time)
    print("SDFastEst-Time", sdfastest_time)
    print("LOG_MAE ESGS vs. SDFastEst", log_mae)
    print("LOG_MAE ESGS vs. UpperBound", get_log_mae(algo_ground_truth_data[algorithm_name + "-sgs_evaluated"],
                                                     algo_ground_truth_data[algorithm_name + "-ub"]))

    # Save SDFastEst data
    algo_ground_truth_data.to_csv(data_path, index=False)


if __name__ == "__main__":
    # Potential Algorithms and configurations
    eval_combos = [
        ("Apriori_DP_fixed (PS)", "PS-APRIORI-FIXED", WRAcc),
        ("DFS_BSD_DP (PS)", "PS-DFS", WRAcc),
        ("Apriori_DP (PS)", "PS-APRIORI", WRAcc),
        ("DFS_BSD_DP_FC (PS)", "PS-DFS", WRAcc),
        # QF05 (Simple Binomial Quality Function) tests
        ("Apriori_DP_fixed_qf05 (PS)", "PS-APRIORI-FIXED", SimpleBinomialQF),
        ("DFS_BSD_DP_qf05 (PS)", "PS-DFS", SimpleBinomialQF),
        ("DFS_BSD_DP_FC_qf05 (PS)", "PS-DFS", SimpleBinomialQF),
    ]

    # --- General Sampling Settings

    # Start Compare
    # Default 100 samples (no timeout, no dynamic sample count)
    compare_for_algorithms(eval_combos, {"n_sample": 100, "timeout": None, "adjust_samples_lower": False,
                                         "adjust_samples_higher": False}, "default")

    # Default 100 samples with timeout 20s to show difference to level-wise sampling
    compare_for_algorithms(eval_combos, {"n_sample": 100, "timeout": 20, "adjust_samples_lower": False,
                                         "adjust_samples_higher": False}, "default_timeout")

    # Dynamic High Count nr_sel
    compare_for_algorithms(eval_combos, {"n_sample": 100, "timeout": None, "adjust_samples_lower": True,
                                         "adjust_samples_higher": True}, "dynamic")

    # Dynamic High Count nr_sel with Timeout 20s to show difference to level-wise sampling and efficiency of timeout
    compare_for_algorithms(eval_combos, {"n_sample": 100, "timeout": 20, "adjust_samples_lower": True,
                                         "adjust_samples_higher": True}, "dynamic_timeout")
