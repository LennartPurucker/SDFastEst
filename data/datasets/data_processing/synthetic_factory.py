# Code to use the synthetic generators. Was used to create all different synthetic datasets.
from data.datasets.data_processing.dataset_functions import synthetic_generators
import re
import os
import csv
from config import synthetic_csv_files_path


# Utilities
def save_dataset(data, name):
    # Clean Name for Saving
    name_without_space = name.replace(" ", "")
    name = re.sub('[^\w\-_\. ]', '_', name_without_space)

    data.to_csv(os.path.join(synthetic_csv_files_path, name + ".csv"), index=False, quoting=csv.QUOTE_ALL)


def save_dataset_wrapper(prefix, data_tuple):
    # Force all to be columns to be string
    data = data_tuple[0].applymap(str)

    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    save_dataset(data, prefix + data_tuple[1])


def get_att_val_list(n):
    return [n for i in range(n)]


def generate_and_save_collections(prefix, lvl_att_val_list, generator, orders_of_magnitude=5):
    # Create non-random datasets multiple times with different order of magnitude
    for i in range(orders_of_magnitude):
        print("Save and Generate for {} and magnitude {}/{}".format(prefix, i + 1, orders_of_magnitude))
        save_dataset_wrapper(prefix + "{}_".format(i), generator(lvl_att_val_list, inst_multi=10 ** i))


def generate_and_save_random_collections(prefix, d, generator, orders_of_magnitude=5):
    # Create non-random datasets multiple times with different order of magnitude
    for i in range(orders_of_magnitude):
        print("Save and Generate for {} and magnitude {}/{}".format(prefix, i, orders_of_magnitude))
        n = 1000 * (10 ** i)  # Thus we get instance 1000, 10000, 100000, 1000000,...
        save_dataset_wrapper(prefix + "{}_".format(i),
                             generator(range_att=(d, d + 1), range_inst=(n, n + 1), range_vals=(d, d + 1)))


def generate_and_save_random_replications(replications, generator):
    datasets = []
    for i in range(replications):
        data, name = generator()
        datasets.append((data, name))

    for data, name in datasets:
        # Save data
        save_dataset_wrapper(data, name)


def generate_and_save_manual_replications(prefix, generator):
    """Create manual configurations of dataset structural properties for random distributed data"""
    # Lvl 1
    generate_and_save_random_collections(prefix + "_LVL_1_", 5, generator)
    # Lvl 2
    generate_and_save_random_collections(prefix + "_LVL_2_", 10, generator)
    # Lvl 3
    generate_and_save_random_collections(prefix + "_LVL_3_", 20, generator)


# Factories
def non_random_dataset_factory(initials, generator):
    # Default
    prefix = "{}_LVL0_".format(initials)
    save_dataset_wrapper(prefix, generator())

    # Lvl 1
    prefix = "{}_LVL1_".format(initials)
    lvl1_att_val_list = get_att_val_list(2)
    generate_and_save_collections(prefix, lvl1_att_val_list, generator)

    # Lvl 2
    prefix = "{}_LVL2_".format(initials)
    lvl2_att_val_list = get_att_val_list(5)
    generate_and_save_collections(prefix, lvl2_att_val_list, generator)

    # Lvl 3
    prefix = "{}_LVL3_".format(initials)
    lvl3_att_val_list = get_att_val_list(7)
    generate_and_save_collections(prefix, lvl3_att_val_list, generator, orders_of_magnitude=2)


def random_dataset_factory(prefix, replications, generator):
    """ Replicate datasets"""

    # Create new datasets
    generate_and_save_random_replications(replications, generator)

    # Create datasets with specific structural properties
    generate_and_save_manual_replications(prefix, generator)


# Main setup
def create_and_save_synthetic_data(replications=5):
    """
        Collect synthetic datasets
    """
    # Random Default datasets
    print("Starting Random Factory")
    random_dataset_factory("ED", replications, synthetic_generators.create_random_equal_dataset)
    random_dataset_factory("ND", replications, synthetic_generators.create_random_normal_dataset)

    # Non-random Datasets
    print("Starting Non-Random Factory")
    non_random_dataset_factory("I", synthetic_generators.create_small_scale_interesting_dataset)
    non_random_dataset_factory("U", synthetic_generators.create_small_scale_uninteresting_dataset)
