# ----- So far Found Algorithm Profiles from Source Code Analysis -----
# Algorithm Profiles contain settings for the sampling framework to model the original implementation during sampling
# as close as possible. The sampling framework uses the algorithm SDFastEst from the paper.
# To illustrate, the SDFastEst Algorithm, however, does not specify what type of pruning shall be employed.
# The Algorithm profiles do.
#       Algorithm Profiles should not contain sampling related adjustments, like number of samples,
#       level-wise or path-wise, or if to remove outliers. This is handled by the sample settings.
SPECIFIC_ALGORITHM_PROFILES = {
    "PS-APRIORI-FIXED": {
        "sg_size_pruning": True,
        "initial_mutually_exclusive_sg": True,
        "no_mutually_exclusive_sg": False,
        "eval_empty_pattern": False,
        "path_est_start_val": 0,
        "min_q_comparer": lambda x, y: x >= y,
        "add_if_required_min_q_comparer": lambda x, y: x > y,
        "dfs_refinements": False
    },
    "PS-APRIORI": {
        "sg_size_pruning": False,
        "initial_mutually_exclusive_sg": True,
        "no_mutually_exclusive_sg": False,
        "eval_empty_pattern": False,
        "path_est_start_val": 0,
        "min_q_comparer": lambda x, y: x >= y,
        "add_if_required_min_q_comparer": lambda x, y: x > y,
        "dfs_refinements": False
    },
    "PS-DFS": {
        "sg_size_pruning": True,
        "initial_mutually_exclusive_sg": False,
        "no_mutually_exclusive_sg": True,
        "eval_empty_pattern": True,
        "path_est_start_val": 1,
        "min_q_comparer": lambda x, y: x > y,
        "add_if_required_min_q_comparer": lambda x, y: x > y,
        "dfs_refinements": True
    },
    "VIKAMINE-DFS-or-SDMap": {
        "sg_size_pruning": True,
        "initial_mutually_exclusive_sg": False,
        "no_mutually_exclusive_sg": True,
        "eval_empty_pattern": False,
        "path_est_start_val": 0,
        "min_q_comparer": lambda x, y: x > y,
        "add_if_required_min_q_comparer": lambda x, y: x >= y,
        "dfs_refinements": True
    }

}
