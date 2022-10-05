# Generators for Basic Synthetic Data usable for Subgroup Discovery to Test orders of magnitude
from random import randrange
import pandas as pd
import numpy as np
import itertools


def get_possible_combinations_numeric(att_val_list, dp_limit=None):
    """
        Create all possible combinations of attributes for the ranges given in the att_val_list
        where each each index represents one attribute
    :param att_val_list: [a,b,c,d,...] for attributes A,B,C,D with a,b,c,d being the corresponding range
    :return: List of all max length combinations i.e. [(0,0,0,0), (0,0,0,1), (0,0,0,2)....]
    """
    # Create iterable att_val_list
    iterable_att_val_list = []
    for att_val in att_val_list:
        iterable_att_val_list.append(range(att_val))

    iterator = itertools.product(*iterable_att_val_list)

    # If only a maximal number of combinations shall be created
    if dp_limit:
        dp_list = []
        # Use iterator until no more combinations exist (except case) or until limit is reached (for case)
        # IDea from https://stackoverflow.com/a/2058973
        for i in range(dp_limit):
            try:
                elem = next(iterator)
            except StopIteration:
                break

            dp_list.append(elem)

        return dp_list

    # Default case
    return list(iterator)


def roll_attribute_ranges(nr_attributes, range_vals):
    """
        Get the ranges for attributes randomly
    """
    att_ranges = []
    for i in range(nr_attributes):
        random_range = randrange(range_vals[0], range_vals[1])
        att_ranges.append(random_range)

    return att_ranges


def build_categorical_df(data, nr_attributes):
    # Create categorical dataframe s.t. all values are strings

    # Get column names which are just numbers
    column_names = [str(i) for i in range(nr_attributes)]
    column_names.append("class")

    # Build df and make string
    df = pd.DataFrame(data, columns=column_names)
    df = df.applymap(str)

    return df


def create_random_equal_dataset(range_att=(5, 15), range_inst=(5000, 10000), range_vals=(2, 8), nr_classes=2):
    """Randomly create a dataset such that the values of attributes are (statistically) equally distributed"""

    # Get overall ranges
    nr_attributes = randrange(range_att[0], range_att[1])
    nr_instances = randrange(range_inst[0], range_inst[1])
    att_ranges = roll_attribute_ranges(nr_attributes, range_vals)

    # Fill data
    data = []
    for i in range(nr_instances):
        instance = []

        # Fill values
        for j in range(nr_attributes):
            # This will create equally distributed values with enough instances
            instance.append(randrange(0, att_ranges[j]))

        # Add class value
        class_val = randrange(0, nr_classes)  # class will be evenly distirubted

        # Add to dataset
        instance.append(class_val)
        data.append(instance)

    name = "Equal Dist (Random-Synthetic) [{}|{}|{}]".format(
        nr_classes, nr_attributes, nr_instances)
    df = build_categorical_df(data, nr_attributes)
    return df, name


def create_random_normal_dataset(range_att=(5, 15), range_inst=(5000, 10000), range_vals=(2, 8), nr_classes=2):
    """Randomly create a dataset such that the values of attributes are (statistically) normally distributed"""

    # Get overall ranges
    nr_attributes = randrange(range_att[0], range_att[1])
    nr_instances = randrange(range_inst[0], range_inst[1])
    att_ranges = roll_attribute_ranges(nr_attributes, range_vals)

    # Get instance values normally distributed
    att_val_list = []
    for j in range(nr_attributes):
        # Idea: this will creat integer values that are normally distributed due to the central limit theorem
        random_values = np.random.binomial(att_ranges[j] - 1, .5, nr_instances)
        att_val_list.append(random_values)

    # Fill data
    data = []
    for i in range(nr_instances):
        instance = []
        # Fill values
        for j in range(nr_attributes):
            instance.append(att_val_list[j][i])

        # Add class value
        random_class = randrange(0, nr_classes)
        instance.append(random_class)

        # Add to dataset
        data.append(instance)

    name = "Normal Dist (Random-Synthetic) [{}|{}|{}]".format(
        nr_classes, nr_attributes, nr_instances)
    df = build_categorical_df(data, nr_attributes)
    return df, name


def adjust_values_for_instances(possible_combinations, instances_nr):
    # Adjust value s.t. the number of instance in the end is around the instance_nr

    nr_combis = len(possible_combinations)
    if nr_combis < instances_nr:
        inst_multi = max(int(instances_nr / nr_combis), 1)
    else:
        inst_multi = 1

    return inst_multi, possible_combinations


def create_small_scale_uninteresting_dataset(att_val_list=(5, 5, 5, 5), inst_multi=100, nr_classes=2,
                                             instances_nr=None):
    """
        Create an create_uninteresting_dataset datasets, settings are fixed for comparison to interesting dataset
        Goal: each combination will have a 50% target share
    """

    possible_combinations = get_possible_combinations_numeric(att_val_list, dp_limit=instances_nr)

    if instances_nr:
        inst_multi, possible_combinations = adjust_values_for_instances(possible_combinations, instances_nr)

    # Select instance values. Get random distributions over indexes of possible combinations
    data = []

    # Create dataset - this makes each combination have 50% target share
    for combi in possible_combinations:
        # Get randoms distribution for class
        # Multiply specific combination
        tmp_combs = [list(combi)] * inst_multi
        distribution_class = list(range(nr_classes)) * inst_multi
        # distribution_class = list(np.random.randint(0, nr_classes, inst_multi))

        for j, x in enumerate(tmp_combs):
            # Add instance to data
            y = x + [distribution_class[j]]
            data.append(y)

    # Return
    nr_attributes = len(att_val_list)
    name = "Uninteresting (Non-Random-Synthetic) [{}|{}|{}]".format(
        nr_classes, nr_attributes, len(possible_combinations) * inst_multi)
    df = build_categorical_df(data, nr_attributes)
    return df, name


def create_small_scale_interesting_dataset(att_val_list=(5, 5, 5, 5), inst_multi=100, nr_classes=2, instances_nr=None):
    """Create an (somewhat) interesting datasets, settings are fixed for comparison to uninteresting dataset"""
    possible_combinations = get_possible_combinations_numeric(att_val_list, dp_limit=instances_nr)

    if instances_nr:
        inst_multi, possible_combinations = adjust_values_for_instances(possible_combinations, instances_nr)

    # Select instance values. Get random distributions over indexes of possible combinations
    data = []

    # Define intervals
    mid_switch = int(len(possible_combinations) * 0.66)
    step = mid_switch // nr_classes
    subsplits = [step * i for i in range(1, nr_classes)]

    # Create dataset
    section = 0
    for running_nr, combi in enumerate(possible_combinations):

        # Run splits for lower 50% of the combinations
        if running_nr <= mid_switch:

            # each section represents one class value
            if (section + 1 < nr_classes) and (running_nr > subsplits[0]):
                section += 1
                subsplits.pop(0)

            # set section class for all combinations
            tmp_combs = [list(combi)] * inst_multi
            distribution_class = [section] * inst_multi
            # Build instance
            for j, x in enumerate(tmp_combs):
                # Combine combination list and class to create full instance and append
                y = x + [distribution_class[j]]
                data.append(y)

        # Run splits for upper half
        else:
            tmp_combs = [list(combi)] * inst_multi
            distribution_class = list(range(nr_classes)) * inst_multi
            for j, x in enumerate(tmp_combs):
                # Add instance to data
                y = x + [distribution_class[j]]
                data.append(y)
    # Return
    nr_attributes = len(att_val_list)
    name = "Interesting (Non-Random-Synthetic) [{}|{}|{}]".format(
        nr_classes, nr_attributes, len(possible_combinations) * inst_multi)
    df = build_categorical_df(data, nr_attributes)
    return df, name
