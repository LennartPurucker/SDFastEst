# This is an example of using SDFastEst for warm starting normal SD algorithms.
# Using SDFastEst as preprocessing can reduce runtime (marginally); provide information on the complexity
#   before running the actual SD algorithm; and is guaranteed to return the same optimal results.
# To achieve this, SDFastEst is needed and a version of an SD algorithm with minor changes to allow passing an initial
#   top-k result set. See SGCountingAprioriFixed for an example.

from data.algorithm_data.ps_algorithms import SGCountingAprioriFixed
import pysubgroup as ps
import time
from general_utils.sd_utils import get_all_sd_task_info, WRAcc
from SDFastEst.sdfastest import SDFastEst


def sd_algorithm_example(target_attribute, target_value, sd_data, init_top_k,
                         result_set_size, true_depth, min_quality, min_sg_size):
    st = time.time()
    # Setup SD Algorithm
    method = SGCountingAprioriFixed()
    qf = ps.WRAccQF()
    target = ps.BinaryTarget(target_attribute=target_attribute, target_value=target_value)
    searchspace = ps.create_selectors(sd_data, ignore=[target_attribute])
    # Setup Task
    task = ps.SubgroupDiscoveryTask(sd_data, target, searchspace, result_set_size=result_set_size,
                                    depth=true_depth,
                                    qf=qf, min_quality=min_quality,
                                    constraints=[ps.MinSupportConstraint(min_support=min_sg_size)])
    # Run Task using algorithm selected above
    results = method.execute(task, initial_result=init_top_k)

    time_taken = time.time() - st
    esgs_taken = method.sgs_evaluated
    return time_taken, esgs_taken, results.to_descriptions()


def assert_results_are_the_same(results_1, results_2):
    # need to make sure they are sorted identically
    results_1.sort(key=lambda x: (x[0], x[1]), reverse=True)
    results_2.sort(key=lambda x: (x[0], x[1]), reverse=True)
    # assert that both lists have the same content
    for res_1_tuple, res_2_tuple in zip(results_1, results_2):
        for ele_1, ele_2 in zip(res_1_tuple, res_2_tuple):
            assert ele_1 == ele_2

    print("### The List of Results are identical ###")


def run_example_task_with_both_settings(dataset_name, depth, i_target_value, n_sample, timeout):
    i_depth, i_min_quality, i_result_set_size, i_min_sg_size, i_target_attribute, i_sd_data, \
    i_sd_data_name = get_all_sd_task_info(dataset_name, depth)

    # --- Run Task without Sampling before
    o_time_taken, o_esgs_taken, o_results = sd_algorithm_example(i_target_attribute, i_target_value, i_sd_data, [],
                                                                 i_result_set_size,
                                                                 i_depth,
                                                                 i_min_quality, i_min_sg_size)
    print("[Without Warm Starting] Time: {}s | ESGS: {}".format(o_time_taken, o_esgs_taken))

    # --- Run Task with Sampling before
    # Sample Settings
    i_quality_function = WRAcc
    sdfastest_algo_profile_name = "PS-APRIORI-FIXED"
    sample_settings = {"outlier_removal": True, "pre_sample": True, "n_sample": n_sample, "timeout": timeout}

    # Do Estimate
    sample_start_time = time.time()
    sdfastest = SDFastEst(i_sd_data, i_sd_data_name, i_target_attribute, i_target_value,
                          i_depth, i_min_quality, i_result_set_size, i_min_sg_size, sdfastest_algo_profile_name,
                          i_quality_function)
    _, e_est = sdfastest.sample_and_estimate(sample_settings)
    sample_time = time.time() - sample_start_time
    # Run Task with init top_k
    o_time_taken, o_esgs_taken, o_results_with_init = sd_algorithm_example(i_target_attribute, i_target_value,
                                                                           i_sd_data,
                                                                           sdfastest.top_k_result, i_result_set_size,
                                                                           i_depth, i_min_quality, i_min_sg_size)
    print("[With Warm Starting] Time: {}s | ESGS: {}".format(o_time_taken, o_esgs_taken))
    # print("[SDFastEst] Time: {}s | ESGS: {}".format(sample_time, sdfastest.evaluated_sgs))
    # print("[Additional SDFastEst Estimate] EST_ESGS for run without Sampling before the run: {}".format(e_est))
    print("[Combined Sampling + Solving the Task] Time: {}s | ESGS: {}".format(o_time_taken + sample_time,
                                                                               o_esgs_taken + sdfastest.evaluated_sgs))
    assert_results_are_the_same(o_results, o_results_with_init)


# Get a SD Task for a dataset
example_tasks = [("Adult_UCI_", 5, ">50K"), ("auto4", 5, "class1")]

for example_task in example_tasks:
    print("------ Test for example task", example_task)
    # One could fine tune n_sample and timeout to guarantee that the combine version is faster.
    run_example_task_with_both_settings(*example_task, n_sample=50, timeout=20)
