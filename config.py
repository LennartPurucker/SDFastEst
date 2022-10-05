import logging
import os


def build_full_path(p):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, p)


# Data Paths
real_world_csv_files_path = build_full_path("data/datasets/csv/real_world")
synthetic_csv_files_path = build_full_path("data/datasets/csv/synthetic")
arff_files_path = build_full_path("data/datasets/arff")
# Relevant File Paths for results management
sd_experiment_settings_path = build_full_path("data/algorithm_data/sd_experiments_settings.csv")

# Default SD settings used in this project
SD_SETTINGS = {
    "result_set_size": 100,
    "depth": [5, 2, 3],
    "min_quality": 0.,
    "min_size_abs": 10,
    "min_size_rel": 0.001
}

# Logger
# Disable useless warnings
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


def get_logger(log_name):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
    )

    return logging.getLogger(log_name)
