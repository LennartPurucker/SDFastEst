import pandas as pd
import numpy as np
import random
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from general_utils.sd_utils import get_all_sd_task_info
from SDFastEst.sdfastest import SDFastEst
from SDFastEst.sampling_profiles import SPECIFIC_SAMPLING_PROFILES
from config import get_logger
from data.algorithm_data.ps_algorithms import SGCountingAprioriFixed, SGCountingDFS, SGCountingApriori, \
    SGCountingDFS_FC
import pysubgroup as ps

logger = get_logger("sss_spectrum")


# ---- Utils
def get_esgs_for_algorithm(algorithm_name, target_attribute, target_value, sd_data,
                           result_set_size, true_depth, min_quality, min_sg_size):
    logger.info("## Get Search Space Size for SD Task by Solving the SD Task ##")

    # Select Algorithm and associated Quality function
    if algorithm_name == "Apriori_DP_fixed (PS)":
        method = SGCountingAprioriFixed()
        qf = ps.WRAccQF()
    elif algorithm_name == "DFS_BSD_DP (PS)":
        method = SGCountingDFS(ps.BitSetRepresentation)
        qf = ps.WRAccQF()
    elif algorithm_name == "Apriori_DP (PS)":
        method = SGCountingApriori()
        qf = ps.WRAccQF()
    elif algorithm_name == "DFS_BSD_DP_FC (PS)":
        method = SGCountingDFS_FC(ps.BitSetRepresentation)
        qf = ps.WRAccQF()
    elif algorithm_name == "DFS_BSD_DP_FC_qf05 (PS)":
        method = SGCountingDFS_FC(ps.BitSetRepresentation)
        qf = ps.SimpleBinomialQF()
    elif algorithm_name == "Apriori_DP_fixed_qf05 (PS)":
        method = SGCountingAprioriFixed()
        qf = ps.SimpleBinomialQF()
    elif algorithm_name == "DFS_BSD_DP_qf05 (PS)":
        method = SGCountingDFS(ps.BitSetRepresentation)
        qf = ps.SimpleBinomialQF()
    else:
        raise ValueError("Unknown Algorithm Name: {}".format(algorithm_name))

    # Setup Target, SearchSpace
    target = ps.BinaryTarget(target_attribute=target_attribute, target_value=target_value)
    searchspace = ps.create_selectors(sd_data, ignore=[target_attribute])
    # Setup Task
    task = ps.SubgroupDiscoveryTask(sd_data, target, searchspace, result_set_size=result_set_size,
                                    depth=true_depth,
                                    qf=qf, min_quality=min_quality,
                                    constraints=[ps.MinSupportConstraint(min_support=min_sg_size)])
    # Run Task using algorithm selected above
    method.execute(task)
    # Get and return evaluated subgroups of the algorithm
    sgs_evaluated = method.sgs_evaluated
    return sgs_evaluated


def plot_spectrum(df, dataset_name, spectrum, plot_points=None):
    # Get plot
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    fig.set_figwidth(6)

    if plot_points is not None:
        # Plot grey dots in the background to show values of individual path estimates
        df.plot(ax=ax)
        plot_points.reset_index().plot.scatter(x="index", y="path_sample_estimate", c='lightgrey', ax=ax)

    else:
        # Only show final aggregated result
        df.plot(ax=ax)

    # log scale
    ax.set_yscale('log')
    # Title
    ax.set_title(("Estimated {} Spectrum {}" +
                  " \n All other SD Settings are default.").format(spectrum, dataset_name))
    # Add description with errors
    plt.ylabel("Search Space Size")
    plt.xlabel(spectrum)

    L = plt.legend()
    L.get_texts()[0].set_text("True Pruned Search Space Size")
    L.get_texts()[1].set_text("Estimated Pruned Search Space Size")
    fig.tight_layout()
    plt.savefig("../data/plots/applications/Spectrum_Plot_{}_{}.pdf".format(spectrum, dataset_name))
    plt.show()


# ---- spectra implementation
def sd_spectra_base(spectrum_name, input_tuple, sample_range, n_points, n_sample_per_point, get_true_val=False,
                    mon_increase=False, depth=None, min_quality=None, result_set_size=None, min_sg_size=None,
                    plot_points=False):
    df = pd.DataFrame()
    # Set Range correctly, for min quality also 0.0 is a valid sampling value.
    if min_quality is True:
        sorted_sample_range = sorted(random.sample(sample_range, n_points - 1))
        sorted_sample_range = [0.0] + sorted_sample_range
    else:
        sorted_sample_range = sorted(random.sample(sample_range, n_points))
    df["X"] = sorted_sample_range

    # Translate input tuple
    dataset_name, to_eval_algo_settings, sample_settings_name = input_tuple

    # Get SD Task settings
    algorithm_name, target_value, quality_function, sdfastest_algo_profile_name = to_eval_algo_settings
    # Get Sample Settings
    sample_settings = {"n_sample": n_sample_per_point, "return_list": True}
    sample_settings.update(SPECIFIC_SAMPLING_PROFILES[sample_settings_name])

    # Get data and SD settings
    i_depth, i_min_quality, i_result_set_size, i_min_sg_size, target_attribute, sd_data, \
    sd_data_name = get_all_sd_task_info(dataset_name, depth, min_quality, result_set_size, min_sg_size)

    # Get sample path estimate for these points
    y = []
    sample_point_list = []
    index_list = []
    true_value = []
    for i, x in enumerate(df["X"], 1):
        logger.info("{} - {} {}/{} | Sample Value: {}".format(sd_data_name, spectrum_name, i, n_points, x))

        # Set x to the correct spectrum, the implementation likes this also more flexibility
        if min_quality is True:
            i_min_quality = x
        elif min_sg_size is True:
            i_min_sg_size = x
        elif result_set_size is True:
            i_result_set_size = x
        elif depth is True:
            i_depth = x
        else:
            raise ValueError("Could not set spectrum value to SD setting parameter")

        # Get estimate for current config
        sdfastest = SDFastEst(sd_data, sd_data_name, target_attribute, target_value,
                              i_depth, i_min_quality, i_result_set_size, i_min_sg_size, sdfastest_algo_profile_name,
                              quality_function)
        _, _, e_est, estimate_list = sdfastest.sample_and_estimate(sample_settings)

        # Add sampled path estimates in correct format
        index_list.extend([x] * len(estimate_list))
        y.append(e_est)
        sample_point_list.extend(estimate_list)

        # Optional true esgs to collect
        if get_true_val:
            # Get true esgs Value
            true_value.append(get_esgs_for_algorithm(algorithm_name, target_attribute, target_value, sd_data,
                                                     i_result_set_size, i_depth, i_min_quality, i_min_sg_size))

    # Optional lines to plot
    if get_true_val:
        df["{} Search Space Size".format(algorithm_name)] = true_value

    # Get regression for line fitting to stay similar to spectra plots from previous work
    iso_reg = IsotonicRegression(increasing=mon_increase).fit(df["X"], y)
    df["Estimated Search Space Size"] = iso_reg.predict(df["X"])

    # Prepare for plot
    df = df.set_index("X")
    if plot_points:
        points_df = pd.DataFrame(sample_point_list, index=index_list, columns=["path_sample_estimate"])
    else:
        points_df = None
    plot_spectrum(df, dataset_name, spectrum_name, plot_points=points_df)


# ----- specific Usage of the spectra

def int_default_X(spectrum_bounds_inclusive):
    return list(range(spectrum_bounds_inclusive[0], spectrum_bounds_inclusive[1] + 1))


def sd_spectra_min_quality(spectrum_bounds_inclusive, input_tuple, n_points=10, n_sample_per_point=10,
                           get_true_val=False):
    # Setup Range for this spectrum
    min_step_for_quality = 0.0001
    X_full = np.arange(spectrum_bounds_inclusive[0] + min_step_for_quality,
                       spectrum_bounds_inclusive[1] + min_step_for_quality,
                       min_step_for_quality).tolist()

    sd_spectra_base("min_quality", input_tuple, X_full, n_points, n_sample_per_point, get_true_val=get_true_val,
                    mon_increase=False, min_quality=True)


def sd_spectra_min_sg_size(spectrum_bounds_inclusive, input_tuple, n_points=10, n_sample_per_point=10,
                           get_true_val=False):
    # Setup Range for this spectrum
    X_full = int_default_X(spectrum_bounds_inclusive)

    sd_spectra_base("min_sg_size", input_tuple, X_full, n_points, n_sample_per_point, get_true_val=get_true_val,
                    mon_increase=False, min_sg_size=True)


def sd_spectra_result_set_size(spectrum_bounds_inclusive, input_tuple, n_points=10, n_sample_per_point=10,
                               get_true_val=False):
    # Setup Range for this spectrum
    X_full = int_default_X(spectrum_bounds_inclusive)

    sd_spectra_base("result_set_size", input_tuple, X_full, n_points, n_sample_per_point,
                    get_true_val=get_true_val, mon_increase="auto", result_set_size=True)


def sd_spectra_depth(spectrum_bounds_inclusive, input_tuple, n_points=10, n_sample_per_point=10,
                     get_true_val=False):
    # Setup Range for this spectrum
    X_full = int_default_X(spectrum_bounds_inclusive)

    sd_spectra_base("depth", input_tuple, X_full, n_points, n_sample_per_point, get_true_val=get_true_val,
                    mon_increase=True, depth=True)
