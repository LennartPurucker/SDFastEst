import time
import os
import sys

# ------------- Ensure that base path is found
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from general_utils import data_loader
import pysubgroup as ps
from data.algorithm_data.ps_algorithms import SGCountingAprioriFixed, SGCountingDFS, SGCountingApriori, SGCountingDFS_FC
from general_utils.sd_utils import get_all_sd_task_info
from config import get_logger

import pandas as pd
from multiprocessing import Process, Queue

logger = get_logger("PS Runner")

LAST_RESULT_SET = []


class PSRunBenchmark:

    def __init__(self, experiment_data, algorithm_name, time_in_min=20):
        self.experiment_data = experiment_data
        self.algorithm_name = algorithm_name
        self.time_in_min = time_in_min

        self.nr_paths = len(experiment_data)

        # Result Lists
        self.time_per_task = []
        self.failed_list = []
        # Sg counting lists
        self.sgs_evaluated_list = []
        self.sgs_eval_interesting_list = []
        self.sgs_quality_interesting_list = []
        self.sgs_top_k_interesting_list = []

    # ---- Actual Code
    def _run_q(self, queue, target_attribute, target_value, sd_data, result_set_size, true_depth,
               min_quality, min_sg_size):

        st = time.time()

        if self.algorithm_name == "Apriori_DP_fixed (PS)":
            method = SGCountingAprioriFixed()
            qf = ps.WRAccQF()
        elif self.algorithm_name == "DFS_BSD_DP (PS)":
            method = SGCountingDFS(ps.BitSetRepresentation)
            qf = ps.WRAccQF()
        elif self.algorithm_name == "Apriori_DP (PS)":
            method = SGCountingApriori()
            qf = ps.WRAccQF()
        elif self.algorithm_name == "DFS_BSD_DP_FC (PS)":
            method = SGCountingDFS_FC(ps.BitSetRepresentation)
            qf = ps.WRAccQF()
        elif self.algorithm_name == "DFS_BSD_DP_FC_qf05 (PS)":
            method = SGCountingDFS_FC(ps.BitSetRepresentation)
            qf = ps.SimpleBinomialQF()
        elif self.algorithm_name == "Apriori_DP_fixed_qf05 (PS)":
            method = SGCountingAprioriFixed()
            qf = ps.SimpleBinomialQF()
        elif self.algorithm_name == "DFS_BSD_DP_qf05 (PS)":
            method = SGCountingDFS(ps.BitSetRepresentation)
            qf = ps.SimpleBinomialQF()
        else:
            raise ValueError("Unknown Algorithm Name: {}".format(self.algorithm_name))

        target = ps.BinaryTarget(target_attribute=target_attribute, target_value=target_value)
        searchspace = ps.create_selectors(sd_data, ignore=[target_attribute])

        task = ps.SubgroupDiscoveryTask(sd_data, target, searchspace, result_set_size=result_set_size,
                                        depth=true_depth,
                                        qf=qf, min_quality=min_quality,
                                        constraints=[ps.MinSupportConstraint(min_support=min_sg_size)])

        method.execute(task)

        sgs_evaluated = method.sgs_evaluated
        sgs_eval_interesting = method.sgs_eval_interesting
        sgs_quality_interesting = method.sgs_quality_interesting
        sgs_top_k_interesting = method.sgs_top_k_interesting
        time_taken = time.time() - st

        queue.put((sgs_evaluated, sgs_eval_interesting, sgs_quality_interesting, sgs_top_k_interesting, time_taken))

    def _run_with_limits(self, target_attribute, target_value, sd_data, result_set_size, true_depth,
                         min_quality, min_sg_size):
        # Function to run relevant code with timeout and handle memory or other errors

        queue = Queue()

        p = Process(target=self._run_q,
                    args=(queue, target_attribute, target_value, sd_data, result_set_size, true_depth,
                          min_quality, min_sg_size))

        # Start Process and block for seconds equal to timeout if not returned earlier
        p.start()
        p.join(timeout=int(self.time_in_min * 60))

        # Vars
        failed = False

        if p.is_alive():
            # Handle correct termination
            p.terminate()
            p.join()

            # Timeout
            failed = True
        else:
            if p.exitcode == 1:
                # Code ran into a bug -> raise exit
                raise SystemExit(1)
            elif p.exitcode == -9:
                # Algorithm killed because uses too much memory
                failed = True

        # Catch none failed run to get results
        if not failed:
            sgs_evaluated, sgs_eval_interesting, sgs_quality_interesting, sgs_top_k_interesting, time_taken = queue.get()
            failed = 0
        else:
            sgs_evaluated = sgs_eval_interesting = sgs_quality_interesting = sgs_top_k_interesting = time_taken = -1
            failed = 1

        return sgs_evaluated, sgs_eval_interesting, sgs_quality_interesting, sgs_top_k_interesting, time_taken, failed

    def run_ps_algorithm_for_sd_tasks(self):
        for index, sd_exp in self.experiment_data.iterrows():
            # ------------------- Load data
            dataset_name = sd_exp["dataset_name"]
            target_value = sd_exp["target_value"]
            exp_depth = sd_exp["depth"]  # depth of the experiment not necessarily equal to true depth of data
            true_depth, min_quality, result_set_size, min_sg_size, target_attribute, \
            sd_data, sd_data_name = get_all_sd_task_info(dataset_name, exp_depth)
            logger.info("------Start {} for: {} [{}/{}]------".format(self.algorithm_name, sd_data_name,
                                                                      index + 1, self.nr_paths))

            sgs_evaluated, sgs_eval_interesting, sgs_quality_interesting, \
            sgs_top_k_interesting, time_taken, failed = self._run_with_limits(target_attribute, target_value, sd_data,
                                                                              result_set_size,
                                                                              true_depth, min_quality, min_sg_size)

            self.failed_list.append(failed)
            self.time_per_task.append(time_taken)
            self.sgs_evaluated_list.append(sgs_evaluated)
            self.sgs_eval_interesting_list.append(sgs_eval_interesting)
            self.sgs_quality_interesting_list.append(sgs_quality_interesting)
            self.sgs_top_k_interesting_list.append(sgs_top_k_interesting)

        self.fill_data_in_place()

    def fill_data_in_place(self):
        self.experiment_data[self.algorithm_name + "-real_time_taken"] = pd.Series(self.time_per_task)
        self.experiment_data[self.algorithm_name + "-failed"] = pd.Series(self.failed_list)

        self.experiment_data[self.algorithm_name + "-sgs_evaluated"] = pd.Series(self.sgs_evaluated_list)
        self.experiment_data[self.algorithm_name + "-sgs_eval_interesting"] = pd.Series(self.sgs_eval_interesting_list)
        self.experiment_data[self.algorithm_name + "-sgs_quality_interesting"] = pd.Series(
            self.sgs_quality_interesting_list)
        self.experiment_data[self.algorithm_name + "-sgs_top_k_interesting"] = pd.Series(
            self.sgs_top_k_interesting_list)


if __name__ == "__main__":
    # Run Algorithms to collect data
    algo_names = ["DFS_BSD_DP (PS)", "Apriori_DP_fixed (PS)", "Apriori_DP (PS)", "Apriori_DP_fixed_qf05 (PS)",
                  "DFS_BSD_DP_qf05 (PS)", "DFS_BSD_DP_FC (PS)", "DFS_BSD_DP_FC_qf05 (PS)"]

    # Settings for data collection
    max_min_per_algorithm = 20
    use_synthetic = True
    sub_sample = False
    write_to_file = True

    for name in algo_names:
        input_data = data_loader.get_sd_exp_data()[["dataset_name", "target_value", "depth"]]

        if not use_synthetic:
            # Remove synthetic data
            input_data = input_data[~input_data["dataset_name"].str.contains("Synthetic")]

        if sub_sample:
            # Sample a set amount of datasets to test
            input_data = input_data.sample(n=sub_sample, random_state=42).reset_index(drop=True)

        PSRunBenchmark(input_data, name, time_in_min=max_min_per_algorithm).run_ps_algorithm_for_sd_tasks()

        ## Removed failed data
        #input_data = input_data[input_data[name + "-failed"] == 0].reset_index(drop=True)

        if write_to_file:
            # Write to file
            input_data.to_csv("./run_data_{}.csv".format(name), index=False)