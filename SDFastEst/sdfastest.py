import pysubgroup as ps
import numpy as np
from config import get_logger
from heapq import heappush, heappop
import random
import copy
import math
import time
from SDFastEst.algorithm_profiles import SPECIFIC_ALGORITHM_PROFILES
import statistics
from itertools import chain
from collections import defaultdict

logger = get_logger("SDSampleTester")


class SDFastEst:
    def __init__(self, data, data_name, target_name, target_value, depth, min_quality, result_set_size, min_sg_size,
                 algorithm_to_estimate, qf, representation_type=None,
                 sg_size_pruning=True, no_mutually_exclusive_sg_generated=False, strictly_greater_min_q=False,
                 only_initial_mutually_exclusive_sg_generated=False, eval_empty_pattern=False,
                 strictly_greater_top_k_min_q=True, dfs_refinements=False):
        """
        Parameters
        ----------
        #### Settings for type of SD tasks that shall be estimated
        :param data: SD dataset
        :param data_name: Name for the SD dataset
        :param algorithm_to_estimate: Name of the algorithm that shall be estimated (algorithm profile name)
        :param depth: SD Task Setting
        :param min_quality: SD Task Setting
        :param result_set_size: SD Task Setting
        :param min_sg_size: SD Task Setting
        :param target_value: SD Task Setting
        :param qf: SD Task Setting

        ----------
        #### Other Settings
        :param representation_type: pysubgroup representation type (defaults to bitsets).

        ----------
        #### Settings for the Sampling Procedure that are specific to the algorithm's implementation
        :param sg_size_pruning: whether or not the algorithm prunes based on the subgroup size
                                (For Custom Profile Setting).
        :param no_mutually_exclusive_sg_generated: True if the algorithm does not generate mutually exclusive subgroups,
                                                   False otherwise (For Custom Profile Setting).
        :param strictly_greater_min_q: True if the algorithm uses strictly greater than for minimal quality comparison,
                                       False if it uses only greater (For Custom Profile Setting).
        :param only_initial_mutually_exclusive_sg_generated: True if an algorithm (like Apriori) generates mutually
                                                             exclusive subgroups initially (for d=2) but never again.
                                                             (For Custom Profile Setting).
        :param eval_empty_pattern: True if algorithm evaluates the empty pattern,
                                   False otherwise (For Custom Profile Setting).
        :param strictly_greater_top_k_min_q: True if the algorithm uses strictly greater than for minimal quality
                                             comparison in the top-k set, False if it uses only greater
                                             (For Custom Profile Setting).
        :param dfs_refinements: if True, the estimation compensation for DFS-based algorithm is used.
                                Set to True for DFS-like algorithms using a static refinement operator
        """

        # SD Task settings
        self.min_quality = min_quality
        self.result_set_size = result_set_size
        self.min_sg_size = min_sg_size
        # set to correct depth, wont change result if depth is too high but make it faster
        self.depth = min(depth, len(data.columns) - 1)
        self.algorithm_to_estimate = algorithm_to_estimate

        # SD Algorithm settings
        if algorithm_to_estimate == "CUSTOM":
            self.sg_size_pruning = sg_size_pruning
            self.initial_mutually_exclusive_sg = only_initial_mutually_exclusive_sg_generated
            self.no_mutually_exclusive_sg = no_mutually_exclusive_sg_generated
            self.eval_empty_pattern = eval_empty_pattern
            self.path_est_start_val = 1 if eval_empty_pattern else 0
            self.dfs_refinements = dfs_refinements
            if strictly_greater_min_q:
                self.min_q_comparer = lambda x, y: x > y
            else:
                self.min_q_comparer = lambda x, y: x >= y
            if strictly_greater_top_k_min_q:
                self.add_if_required_min_q_comparer = lambda x, y: x > y
            else:
                self.add_if_required_min_q_comparer = lambda x, y: x >= y
        else:
            profile_values = SPECIFIC_ALGORITHM_PROFILES.get(algorithm_to_estimate, False)

            if not profile_values:
                raise ValueError("Unknown algorithm estimation settings {}".format(algorithm_to_estimate))

            # Set based on loaded profile
            self.sg_size_pruning = profile_values["sg_size_pruning"]
            self.initial_mutually_exclusive_sg = profile_values["initial_mutually_exclusive_sg"]
            self.no_mutually_exclusive_sg = profile_values["no_mutually_exclusive_sg"]
            self.eval_empty_pattern = profile_values["eval_empty_pattern"]
            self.path_est_start_val = profile_values["path_est_start_val"]
            self.min_q_comparer = profile_values["min_q_comparer"]
            self.add_if_required_min_q_comparer = profile_values["add_if_required_min_q_comparer"]
            self.dfs_refinements = profile_values["dfs_refinements"]

        # Evaluation Method Settings
        self.representation_type = representation_type if representation_type is not None else ps.BitSetRepresentation

        # Dataset
        self.data = data
        self.data_name = data_name
        self.dataset_size = self.data.shape[0]

        # Result specifics
        self.top_k_result = []
        self.evaluated_sgs = 0
        self.not_pruned_sgs = 0

        # -------- Target specifics
        self.target_name = target_name
        target = ps.BinaryTarget(target_attribute=self.target_name, target_value=target_value)
        self.target_cover = target.covers(data)
        self.target_positives_count = np.count_nonzero(self.target_cover)

        # QF specific computation of target share
        self.target_share = self.target_positives_count / self.dataset_size

        # Get QF
        self.qf = qf(self.dataset_size, self.target_share)

        # Additional stats
        self.pruned_by = [0, 0]  # min sgs size, op est

        # -------- Selector specifics
        # Gather Selector representation - adapted from pysubgroup
        est_result = self._default_checks_pre_selector()
        # Only go further if default checks made clear that estimation is not done yet
        if est_result is None:
            # Selector evaluation
            self.selectors = ps.create_selectors(self.data, ignore=[self.target_name])
            self.interesting_selectors, self.empty_path, t_selectors = self._selector_evaluation()
            self.nr_selectors = len(self.selectors)
            self.nr_non_interesting_sel = self.nr_selectors - len(self.interesting_selectors)

            # set default values for possible reset
            self.default_eval_sgs = self.evaluated_sgs
            self.default_not_pruned_sgs = self.not_pruned_sgs
            self.default_pruned_by = self.pruned_by[:]  # explicitly copy the list
            self.default_top_k = self.top_k_result[:]  # explicitly copy the list

            est_result = self._default_checks_post_selector()
            # Check again to avoid expensive speed up computations that would not be needed
            if est_result is None:
                # Post processing - getting data for speed up and check later
                # Data needed to quickly check for mutual exclusivity
                self.selector_to_attribute = {sel_rep: sel_rep.selectors[0].attribute_name for sel_rep in
                                              self.interesting_selectors}
                if self.dfs_refinements:
                    self.ref_operator = StaticSpecializationOperator(t_selectors)
        self.pre_sample_est_result = est_result

    # -------- Intractable functions 
    def sample_and_estimate(self, sample_settings=None):
        """
            Estimate the search space size for an SD Problem by sampling
            By default, no improvements are used and 100 samples without timeout are taken.

            Pass parameters through a settings dictionary
            ----------
            :param sample_settings: The dictionary that is used to pass parameters.

            Possible Parameters in sample_settings:
            ----------
            :param n_sample: number of samples (int), if None 100 is used, Default=None.
            :param timeout: timeout in seconds, if None no timeout, Default=None.
            :param return_list: if a list of all estimates should be returned, Default=False.
            :param level_wise: if level-wise sampling shall be used.
                               If false path-wise sampling is used, Default=False.
            :param outlier_removal: if outlier removal is used for final aggregation, Default=False.
                                    Note: this only applies to the estimate of evaluated subgroups (e.g. SDFastEst)
            :param adjust_samples_lower: if Ture, the sample count is lower if appropriate, Default=False.
                                         if True, the sample count is set to min(n_samples, nr_selectors).
                                            If both adjust_samples_lower and adjust_samples_higher is True,
                                            n_samples is set to the number of selectors.
            :param adjust_samples_higher: if Ture, the sample count is higher if appropriate, Default=False.
                                          if True, the sample count is set to max(n_samples, nr_selectors).
                                              If both adjust_samples_lower and adjust_samples_higher is True,
                                              n_samples is set to the number of selectors.
            :param pre_sample: if pre_sampling should be used during path-wise sampling, Default=False.
            :param pre_sample_split: For Path-wise sampling and if pre_sampling is True, this defines the percentage of
                                     samples to be used for pre-sampling, i.e., to build an initial top-k during.
                                     Percentage input as a float between 0 and 1, Default=0.2.

            Returns
            -------
            :return: The result depends on the parameter return_list.
                         If return_list is False, returns (ei_est, e_est). Whereby ei_est represents the estimate for
                            evaluated subgroups and ei_est the estimate for evaluation interesting subgroups.
                         If return_list is True, returns (ei_est, (n_samples), e_est, (n_samples)).
                            Whereby (n_samples) is a list of estimates with length n_samples. The two given
                            estimate lists are different. One contains the estimates for ei_est samples and the other
                            for e_est samples.
        """
        # Catch None Case
        sample_settings = {} if sample_settings is None else sample_settings

        # Get Parameters or default
        n_sample = sample_settings.get("n_sample", 100)
        n_sample = n_sample if n_sample is not None else 100
        timeout = sample_settings.get("timeout", None)
        return_list = sample_settings.get("return_list", False)
        level_wise = sample_settings.get("level_wise", False)
        outlier_removal = sample_settings.get("outlier_removal", False)
        adjust_samples_lower = sample_settings.get("adjust_samples_lower", False)
        adjust_samples_higher = sample_settings.get("adjust_samples_higher", False)
        pre_sample = sample_settings.get("pre_sample", False)
        pre_sample_split = sample_settings.get("pre_sample_split", 0.2)
        if (not isinstance(pre_sample_split, float)) or (pre_sample_split < 0) or (pre_sample_split > 1):
            raise ValueError("The sampling parameter pre_sample_split is either not a float or not between 0 and 1")

        # Adjust Sample Count
        if adjust_samples_higher and adjust_samples_lower:
            n_sample = self.nr_selectors
        elif adjust_samples_lower:
            n_sample = min(n_sample, self.nr_selectors)
        elif adjust_samples_higher:
            n_sample = max(n_sample, self.nr_selectors)

        # Default check
        if self.pre_sample_est_result and self.pre_sample_est_result.is_final_result:
            logger.info("Stop sampling immediately since no-sampling needed for estimation.")
            return self.pre_sample_est_result.get_return_value(return_list)
        else:
            sample_method = "Level-wise" if level_wise else "Path-wise"
            logger.info("Start Sampling for {} with n_sample = {} | {} | {}".format(self.data_name, n_sample,
                                                                                    sample_method,
                                                                                    self.algorithm_to_estimate))

        # Start Sampling Method
        if level_wise:
            if timeout is not None:
                raise ValueError("Timeout not supported for level wise!")
            return self.level_wise_sampling_base(n_sample, return_list=return_list, outlier_removal=outlier_removal)

        return self.sample_base(n_sample, timeout=timeout, return_list=return_list, outlier_removal=outlier_removal,
                                pre_sample=pre_sample, pre_sample_split=pre_sample_split)

    def reset_for_new_sampling(self):
        """
            If one wants to use the Sampling Class for multiple sampling runs, the  class object can be re-used 
            after calling this function.
        """
        self.top_k_result = []
        self.evaluated_sgs = self.default_eval_sgs
        self.not_pruned_sgs = self.default_not_pruned_sgs
        self.pruned_by = self.default_pruned_by

    # -------- Estimation Methods
    def evaluated_path_estimate_for_out_degree_list(self, out_degree_list, patterns_checked_for_pruning_per_lvl):
        """ Path Estimate for Number of evaluated Subgroups (from SDFastEst Paper)"""
        path_est_value = self.path_est_start_val
        len_path = len(out_degree_list)

        for i in range(1, len_path + 1):
            to_prod_list = out_degree_list[:i]

            # Adapt level estimate with compensation value
            to_prod_list[i - 1] += patterns_checked_for_pruning_per_lvl[i - 1]

            path_est_value += math.prod(to_prod_list) / math.factorial(i)

        return path_est_value

    def evaluation_interesting_path_estimate_for_out_degree_list(self, out_degree_list):
        """ Path Estimate for number of evaluation interesting subgroups """
        path_est_value = self.path_est_start_val
        len_path = len(out_degree_list)

        for i in range(1, len_path + 1):
            # sum i in [1,l] ( 1/i! * product over all degrees from level 0 to current level-1)
            # [:i] gives all list items from 0 to i-1, therefore correct here
            path_est_value += math.prod(out_degree_list[:i]) / math.factorial(i)

        return path_est_value

    # -------- Pre-sampling checks
    def _default_checks_pre_selector(self):
        # Depth check 0
        if self.depth == 0:
            return EstimationResult(ei_est=0, e_est=0)

    def _default_checks_post_selector(self):
        # Empty pattern check
        if self.eval_empty_pattern and (not self.is_interesting(self.empty_path)):
            return EstimationResult(ei_est=0, e_est=1)

        # Depth check 1
        elif self.depth == 1:
            true_res = len(self.interesting_selectors)
            real_res = self.nr_selectors

            if self.eval_empty_pattern:
                # +1 in both case as empty pattern must be interesting if depth is 1
                real_res += 1
                true_res += 1

            return EstimationResult(ei_est=true_res, e_est=real_res)

    def _selector_evaluation(self):
        # Get the set of initially interesting selectors and the empty path representation

        stats_and_sel = []
        # Set the representation
        with self.representation_type(self.data, self.selectors) as representation:
            # Get function to build representation of a given selector
            combine_selectors = getattr(representation.__class__, "Conjunction")

            empty_path = combine_selectors([])
            # Break early if empty pattern evaluation is activated and we detect that it is uninteresting
            if self.eval_empty_pattern and (not self.is_interesting(empty_path)):
                return [], empty_path, []
            t_selectors = []
            # For each selector, build representation and store
            for sel_base in representation.selectors_to_patch:
                sel_rep = combine_selectors([sel_base])

                # Evaluate all selectors
                op_es, quality, size_sel = self.evaluate_sg(sel_rep)
                self.add_if_required(sel_rep, quality)
                stats_and_sel.append((op_es, quality, size_sel, sel_rep))
                t_selectors.append(sel_rep)

            # prune selectors
            interesting_selectors = [sel_rep for op_es, quality, size_sel, sel_rep in stats_and_sel
                                     if self.is_not_pruned(op_es, quality, size_sel, sel_rep)]

        return interesting_selectors, empty_path, t_selectors

    # -------- Path-sample-handler fpr the Path object
    def _extend_and_evaluate_path(self, path):
        # Check if path is already at a leaf node
        if not path:
            return

        # Check if path is at a leaf node and stop if yes
        if (path.path_sg.depth + 1 == self.depth) or (len(path.possible_extensions) == 0):
            path.set_at_leaf()
            return

        # Extend the path once
        path.extend()

        # Evaluate all possible extensions
        stats_and_extensions = []
        mx_count = 0
        for extension in path.possible_extensions:
            extended_path = copy.copy(extension)
            extended_path.append_and(path.path_sg)

            optimistic_estimate, quality, description_size_count = self.evaluate_sg(extended_path)

            # Add (if required) to Top-k and return true
            self.add_if_required(extended_path, quality)

            stats_and_extensions.append((optimistic_estimate, quality, description_size_count, extension))

            # fixme could make this more efficient with else and using attribute properties of selectors
            # Count mutually exclusive evaluations
            if self.selector_to_attribute[extension] == self.selector_to_attribute[path.last_sampled_extension]:
                mx_count += 1

        # Store results in path
        path.stats_and_extensions = stats_and_extensions
        path.mx_count = mx_count

    def _prune_path(self, path):
        # Check if path is already at a leaf node
        if not path:
            return

        # Prune each extension
        still_interesting_extensions = [sel for op_es, quality, sg_size, sel in path.stats_and_extensions if
                                        self.is_not_pruned(op_es, quality, sg_size, sel)]

        # Get adjustment counts
        uninteresting_patterns_checked_for_pruning_current_lvl = len(path.stats_and_extensions) - len(
            still_interesting_extensions)
        if (self.initial_mutually_exclusive_sg and ((path.length + 1) != 2)) or self.no_mutually_exclusive_sg:
            uninteresting_patterns_checked_for_pruning_current_lvl -= path.mx_count

        # Compensation Value Adjustment for DFS-based algorithms
        if self.dfs_refinements:
            # Get count of non-interesting extensions that are checked at current level due to DFS using a
            # refinement operator

            # Get set of checked selectors for cross referencing
            checked_sels = set([sel for op_es, quality, sg_size, sel in path.stats_and_extensions])
            # Get list of sels that are checked by dfs but were skipped due to not being an interesting extension
            checked_by_dfs = [sel for sel in self.ref_operator.refinements(path.path_sg) if sel not in checked_sels]
            # Add to the compensate value
            additional_dfs_extensions = len(checked_by_dfs)
            val1 = path.evaluated_but_not_interesting_counts[
                       -1] + uninteresting_patterns_checked_for_pruning_current_lvl
            val2 = uninteresting_patterns_checked_for_pruning_current_lvl + additional_dfs_extensions
            uninteresting_patterns_checked_for_pruning_current_lvl = (val1 + val2) / 2

        # Store values in path object
        path.update_compensation(uninteresting_patterns_checked_for_pruning_current_lvl)
        path.out_degrees.append(len(still_interesting_extensions))
        path.possible_extensions = still_interesting_extensions

    # -------- Pathwise sampling
    def both_single_sample(self):
        out_degree_list, patterns_checked_for_pruning_per_lvl = self.sample_path_wise()
        ei_est = self.evaluation_interesting_path_estimate_for_out_degree_list(out_degree_list)
        e_est = self.evaluated_path_estimate_for_out_degree_list(out_degree_list, patterns_checked_for_pruning_per_lvl)
        return ei_est, e_est

    def sample_path_wise(self):
        # Initialize a sample path
        nr_interesting_selectors = len(self.interesting_selectors)
        sample_path = PathSample(copy.copy(self.empty_path), copy.copy(self.interesting_selectors),
                                 nr_interesting_selectors, self.nr_non_interesting_sel)

        # -------- Path sample processing
        # Iterate as long as the path is not yet at a leaf node
        while bool(sample_path):
            self._extend_and_evaluate_path(sample_path)
            self._prune_path(sample_path)

        # Return
        return sample_path.out_degrees, sample_path.get_compensation_value()

    def sample_base(self, n_sample, timeout=None, return_list=False, outlier_removal=False, pre_sample=False,
                    pre_sample_split=0.2):
        # Do Pre-sampling if needed
        if pre_sample:
            # Get sample splits
            n_pre_sample = int(n_sample * pre_sample_split)
            n_sample_split = int(n_sample * (1 - pre_sample_split))
            if timeout:
                timeout_pre = int(timeout * pre_sample_split)
                start_time = time.time()

            # Pre_sample
            for _ in range(n_pre_sample):
                self.sample_path_wise()

                # Timeout check
                if timeout and (time.time() - start_time) > timeout_pre:
                    break
        else:
            n_sample_split = n_sample
            start_time = time.time()

        # Path-wise Sample
        estimate_list = []
        for _ in range(n_sample_split):
            estimate_list.append(self.both_single_sample())
            # Timeout check
            if timeout and (time.time() - start_time) > timeout:
                break

        # Build results
        result = EstimationResult(estimations_list=estimate_list, outlier_removal=outlier_removal)
        return result.get_return_value(return_list)

    # -------- Levelwise sampling
    def extend_and_evaluate_current_level(self, sample_collection):
        for path in sample_collection:
            self._extend_and_evaluate_path(path)

    def prune_current_level(self, sample_collection):
        for path in sample_collection:
            self._prune_path(path)

    def level_wise_sampling_base(self, n_sample, return_list=False, outlier_removal=False, dfs_compensation=False):
        # Initialize all sample paths
        nr_interesting_selectors = len(self.interesting_selectors)
        sample_collection = [PathSample(copy.copy(self.empty_path), copy.copy(self.interesting_selectors),
                                        nr_interesting_selectors, self.nr_non_interesting_sel)
                             for _ in range(n_sample)]

        # -------- Path sample processing
        # Iterate as long as every path is not yet at a leaf node
        while any(sample_collection):
            # Not required to go over all levels but instead until all paths are at a leaf node which
            # happens (in the worst case) if we are at max level

            # Next level extend and evaluate all possible future extensions
            self.extend_and_evaluate_current_level(sample_collection)

            # Next step for this level: prune after full "add_if_required" from before
            self.prune_current_level(sample_collection)

        # Get path estimates lists, first entry represents without correction (i.e., evaluation interesting)
        # second with correction (i.e., evaluated)
        estimate_list = [(self.evaluation_interesting_path_estimate_for_out_degree_list(path.out_degrees),
                          self.evaluated_path_estimate_for_out_degree_list(path.out_degrees,
                                                                           path.get_compensation_value()))
                         for path in sample_collection]

        # Build results
        result = EstimationResult(estimations_list=estimate_list, outlier_removal=outlier_removal)
        return result.get_return_value(return_list)

    # ---- stuff for top-k pruning from Pysubgroup - used and re-implemented here to have more control
    def add_if_required(self, sg, quality):
        if self.add_if_required_min_q_comparer(quality, self.min_quality):
            # Need to check for sg size constraint
            if sg.size_sg < self.min_sg_size:
                return
            # Need to check for duplicates due to sampling approach
            if (quality, sg) in self.top_k_result:
                return

            # no need for call for min required quality due to the check below
            if len(self.top_k_result) < self.result_set_size:
                heappush(self.top_k_result, (quality, sg))
            elif quality > self.top_k_result[0][0]:
                heappop(self.top_k_result)
                heappush(self.top_k_result, (quality, sg))

    def minimum_required_quality(self):
        if len(self.top_k_result) < self.result_set_size:
            return self.min_quality
        else:
            return self.top_k_result[0][0]

    # ---- interesting evaluation stuff
    def is_interesting(self, description):
        """
        !Assumes input is in a pysubgroup representation!

        Return False if the path would be pruned, Ture else

            uses:
                - min quality pruning
                - optimistic estimate pruning
                - min sgs_size pruning
                - top-k pruning over so far seen samples

        """
        optimistic_estimate, quality, description_size_count = self.evaluate_sg(description)
        return self.is_not_pruned(optimistic_estimate, quality, description_size_count, description)

    def evaluate_base(self, target_cover, description):
        if self.representation_type == ps.BitSetRepresentation:
            # Hardcoded bitset check as it is a lot faster
            description_positives_count = np.count_nonzero(
                np.logical_and(target_cover, description.representation))
        else:
            description_positives_count = np.count_nonzero(target_cover[description])

        return description_positives_count

    def evaluate_sg(self, description):
        # ---------- Get relevant statistics
        description_size_count = description.size_sg
        description_positives_count = self.evaluate_base(self.target_cover, description)

        # ---------- Get QF Values (WRAccQF hard-coded for now)
        optimistic_estimate = self.qf.optimistic_estimate(description_positives_count)

        # required here because even non-high enough optimistic estimates need to be added
        quality = self.qf.quality(description_size_count, description_positives_count)

        # collect stats
        self.evaluated_sgs += 1

        return optimistic_estimate, quality, description_size_count

    def is_not_pruned(self, optimistic_estimate, quality, description_size_count, description):
        """ Function to check if a subgroup is not pruned, keep not used parameters to guarantee similar interface"""

        # ---------- Pruning Check
        if self.sg_size_pruning and description_size_count < self.min_sg_size:
            # Return False if the size of the subgroup is too low and the algorithm  does prune based on this constraint
            self.pruned_by[0] += 1
            return False

        if self.min_q_comparer(optimistic_estimate, self.minimum_required_quality()):
            # This description would not be pruned
            self.not_pruned_sgs += 1
            return True

        # Return False if optimistic estimate is too low
        self.pruned_by[1] += 1
        return False


class PathSample:
    def __init__(self, empty_base_path, initial_extensions, initial_out_degree, initial_evaluation_overhead):
        """
            An internal object to represent a sampled Path.
            Parameters corresponds to relevant values needed at some point. (Not explained here)
        """
        self.out_degrees = [initial_out_degree]
        self.evaluated_but_not_interesting_counts = [initial_evaluation_overhead]
        self.length = 0
        self.path_sg = empty_base_path
        self.possible_extensions = initial_extensions
        self.at_leaf = False

        # Extension stuff
        self.last_sampled_extension = None
        self.stats_and_extensions = None
        self.mx_count = None

    def __bool__(self):
        # Make so that the sample is true if it is not yet finished
        return not self.at_leaf

    def set_at_leaf(self):
        self.at_leaf = True
        # Do some clean up to save memory
        del self.last_sampled_extension
        del self.stats_and_extensions
        del self.mx_count
        del self.path_sg
        del self.possible_extensions

    def extend(self):
        """ Sample path one-step extension """
        # Randomly choose next extension
        sampled_extension = random.choice(self.possible_extensions)
        self.path_sg.append_and(sampled_extension)
        self.length += 1

        # Remove sampled selector from interesting extensions
        self.possible_extensions.remove(sampled_extension)
        self.last_sampled_extension = sampled_extension

    def get_compensation_value(self):
        return self.evaluated_but_not_interesting_counts

    def update_compensation(self, uninteresting_patterns_checked_for_pruning_current_lvl):
        # Update default evaluated
        self.evaluated_but_not_interesting_counts.append(uninteresting_patterns_checked_for_pruning_current_lvl)


class EstimationResult:
    def __init__(self, estimations_list=None, outlier_removal=None,
                 ei_est=None, e_est=None, ei_est_list=None, e_est_list=None):
        """ Internal Object to compute the estimation results"""

        if ei_est is not None and e_est is not None:
            # Init for pre-sample estimation result (direct input)
            self.ei_est = ei_est
            self.e_est = e_est
            self.ei_est_list = [ei_est] if ei_est_list is None else ei_est_list
            self.e_est_list = [e_est] if e_est_list is None else e_est_list
            self.is_final_result = True
        elif estimations_list is not None:
            # Init for after-sample estimation result (build result input)
            self.outlier_removal = outlier_removal
            # Get results based on mode
            ei_est_list = [x[0] for x in estimations_list]
            e_est_list = [x[1] for x in estimations_list]
            ei_est = self.aggregate_estimates(ei_est_list, ei_mode=True)
            e_est = self.aggregate_estimates(e_est_list)

            self.ei_est_list = ei_est_list
            self.e_est_list = e_est_list
            self.e_est = e_est
            self.ei_est = ei_est
        else:
            raise ValueError("Missing Init values for EstimationResult")

    def aggregate_estimates(self, estimate_list, ei_mode=False):
        """ Aggregation Method for estimates """

        # true ei_mode to apply different aggregation to own estimate and not original method
        if self.outlier_removal and not ei_mode:
            # Remark: trimmed mean could be come empirical better when using % instead of number based on error delta
            # However, this might not be theoretical correct anymore

            error_delta = 0.0005  # Minimal error selected for mean
            to_trim = int(math.log2(1 / error_delta))

            # only apply trimmed mean if enough sample estimates exit
            # by default at least half of the estimates would need to remain after trimming
            if len(estimate_list) >= (4 * to_trim):
                estimate_list = sorted(estimate_list)[to_trim:-to_trim]
            agg_est = statistics.mean(estimate_list)
        else:
            agg_est = statistics.mean(estimate_list)

        return agg_est

    def get_return_value(self, return_list):
        if return_list:
            return self.ei_est, self.ei_est_list, self.e_est, self.e_est_list
        else:
            return self.ei_est, self.e_est


class StaticSpecializationOperator:
    """ StaticSpecializationOperator from Pysubgroup with minor adjustments to fit to this framework """

    def __init__(self, selectors):
        search_space_dict = defaultdict(list)
        for selector in selectors:
            search_space_dict[selector.selectors[0].attribute_name].append(selector)  # adjusted
        self.search_space = list(search_space_dict.values())
        self.search_space_index = {key: i for i, key in enumerate(search_space_dict.keys())}

    def refinements(self, subgroup):
        if subgroup.depth > 0:
            index_of_last = self.search_space_index[subgroup._selectors[-1].attribute_name]
            new_selectors = chain.from_iterable(self.search_space[index_of_last + 1:])
        else:
            new_selectors = chain.from_iterable(self.search_space)
        return new_selectors  # adjusted
