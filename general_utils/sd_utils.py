import numpy as np
from config import SD_SETTINGS
from general_utils import data_loader

use_arff = False
name_to_path = data_loader.get_name_to_path_dict(use_arff)
default_target_column = -1


# ---- SD Settings
def load_sd_dataset(dataset_name):
    return data_loader.load_dataset(name_to_path[dataset_name], use_arff)


def get_all_sd_task_info(dataset_name, exp_depth, min_quality=None, result_set_size=None, min_sg_size=None):
    sd_data, sd_data_name = load_sd_dataset(dataset_name)

    # Collect Experiment settings
    target_attribute = sd_data.columns[default_target_column]
    true_depth, min_quality, result_set_size, min_sg_size = get_default_sd_settings(sd_data, exp_depth, min_quality,
                                                                                    result_set_size, min_sg_size)

    return true_depth, min_quality, result_set_size, min_sg_size, target_attribute, sd_data, sd_data_name


def get_default_sd_settings(data, depth, min_quality=None, result_set_size=None, min_sg_size=None):
    input_min_quality = min_quality if min_quality is not None else SD_SETTINGS["min_quality"]
    input_result_set_size = result_set_size if result_set_size is not None else SD_SETTINGS["result_set_size"]
    input_min_sg_size = min_sg_size if min_sg_size is not None else max(SD_SETTINGS["min_size_abs"],
                                                                        int(SD_SETTINGS["min_size_rel"]
                                                                            * len(data.index)))
    input_depth = depth if depth is not None else SD_SETTINGS["depth"][0]
    input_depth = min(input_depth, len(data.columns) - 1)

    return input_depth, input_min_quality, input_result_set_size, input_min_sg_size


# ---- Hardcoded Quality Functions for StandardQF
class StandardQF:
    def __init__(self, a, dataset_size, target_share):
        self.dataset_size = dataset_size
        self.target_share = target_share
        self.a = a

    def optimistic_estimate(self, positives_subgroup):
        # If positives_subgroup is 0, the optimistic estimate is 0 or by convention nan for later compare statement
        if positives_subgroup == 0:
            return np.nan
        # (p_p/i0) ** a * (1-t0)
        return (positives_subgroup / self.dataset_size) ** self.a * (1 - self.target_share)

    def quality(self, instances_subgroup, positives_subgroup):
        # If instances_subgroup is 0, the optimistic estimate is 0 or by convention nan for later compare statement
        if instances_subgroup == 0:
            return np.nan
        p_subgroup = np.divide(positives_subgroup, instances_subgroup)
        # (ip/i0) ** a * (t_p - t0)
        return (instances_subgroup / self.dataset_size) ** self.a * (p_subgroup - self.target_share)


class WRAcc(StandardQF):
    def __init__(self, dataset_size, target_share):
        super().__init__(1, dataset_size, target_share)


class SimpleBinomialQF(StandardQF):
    def __init__(self, dataset_size, target_share):
        super().__init__(0.5, dataset_size, target_share)
