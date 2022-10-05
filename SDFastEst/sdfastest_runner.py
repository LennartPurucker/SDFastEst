import pandas as pd
import time
from config import get_logger
from general_utils.sd_utils import get_all_sd_task_info
from SDFastEst.sdfastest import SDFastEst
import math

logger = get_logger("SDFastEst Runner")


def get_depth_ub(nr_sel, true_depth):
    # search space size (sss) Depth limited size of descriptions
    # +1 for including max depth as it would be in a sum
    return sum(math.comb(nr_sel, i) for i in range(true_depth + 1))


def run_sdfastest(ground_truth_data, quality_function, algorithm_profile_name, algorithm_name,
                  sample_settings):
    # Go over each path
    nr_paths = len(ground_truth_data)
    e_estimations = []
    ei_estimations = []
    time_taken_list = []
    sdfastest_esgs = []
    depth_ub_list = []

    for index, sd_exp in ground_truth_data.iterrows():
        # ------------------- Load data
        dataset_name = sd_exp["dataset_name"]
        target_value = sd_exp["target_value"]
        exp_depth = sd_exp["depth"]  # depth of the experiment not necessarily equal to true depth of data
        true_depth, min_quality, result_set_size, min_sg_size, target_attribute, \
        sd_data, sd_data_name = get_all_sd_task_info(dataset_name, exp_depth)
        logger.info("------Start processing: {} [{}/{}]------".format(sd_data_name, index + 1, nr_paths))

        st = time.time()

        sdfastest = SDFastEst(sd_data, sd_data_name, target_attribute, target_value,
                              true_depth, min_quality, result_set_size, min_sg_size, algorithm_profile_name,
                              quality_function)
        ei_est, e_est = sdfastest.sample_and_estimate(sample_settings)
        time_taken = time.time() - st
        esgs = sdfastest.evaluated_sgs

        # Get Depth Upper Bound and add to list
        nr_sel = sum(list(sd_data.nunique()[:-1]))
        depth_ub = get_depth_ub(nr_sel, true_depth)

        logger.info("------Finished Processing------")

        ei_estimations.append(ei_est)
        e_estimations.append(e_est)
        time_taken_list.append(time_taken)
        sdfastest_esgs.append(esgs)
        depth_ub_list.append(depth_ub)

    ground_truth_data[algorithm_name + "-e_est"] = pd.Series(e_estimations)
    ground_truth_data[algorithm_name + "-ei_est"] = pd.Series(ei_estimations)
    ground_truth_data[algorithm_name + "-time_to_est"] = pd.Series(time_taken_list)
    ground_truth_data[algorithm_name + "-sdfastest-esgs"] = pd.Series(sdfastest_esgs)
    ground_truth_data[algorithm_name + "-ub"] = pd.Series(depth_ub_list)
