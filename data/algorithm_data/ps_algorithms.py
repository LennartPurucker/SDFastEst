import pysubgroup as ps
from itertools import combinations
import numpy as np
from collections import namedtuple
from heapq import heappush, heappop


# ---- Utils to control counting
def add_if_required(algo_object, result, sg, quality, task, statistics=None, check_duplicates=False):
    if quality > task.min_quality:
        algo_object.sgs_quality_interesting += 1
        if not ps.constraints_satisfied(task.constraints, sg, statistics, task.data):
            return

        # Need to check for duplicates due to sampling approach
        if check_duplicates and (quality, sg) in result:
            return

        if len(result) < task.result_set_size:
            heappush(result, (quality, sg, statistics))
            algo_object.sgs_top_k_interesting += 1
        elif quality > result[0][0]:
            heappop(result)
            heappush(result, (quality, sg, statistics))
            algo_object.sgs_top_k_interesting += 1


# ---- Algorithms
class SGCountingAprioriFixed:
    """
    Pysubgroup's Apriori adapted and streamlined to count number of evaluated subgroups
        + Added monotone constrain pruning
        + Added code to enable warm starting
    """

    def __init__(self, representation_type=None, combination_name='Conjunction', use_numba=False):
        self.combination_name = combination_name

        if representation_type is None:
            representation_type = ps.BitSetRepresentation
        self.representation_type = representation_type
        self.use_vectorization = True
        self.use_repruning = False
        self.optimistic_estimate_name = 'optimistic_estimate'
        self.next_level = self.get_next_level
        self.compiled_func = None

        # New
        self.sgs_evaluated = 0
        self.sgs_eval_interesting = 0
        self.sgs_quality_interesting = 0
        self.sgs_top_k_interesting = 0
        self.check_duplicates = False

    def get_next_level_candidates_vectorized(self, task, result, next_level_candidates):
        promising_candidates = []
        statistics = []
        optimistic_estimate_function = getattr(task.qf, self.optimistic_estimate_name)
        for sg in next_level_candidates:
            statistics.append(task.qf.calculate_statistics(sg, task.target, task.data))
            self.sgs_evaluated += 1

        tpl_class = statistics[0].__class__

        # Transform to vector
        vec_statistics = tpl_class._make(np.array(tpl) for tpl in zip(*statistics))

        # Use vector to get data
        qualities = task.qf.evaluate(None, task.target, task.data, vec_statistics)
        optimistic_estimates = optimistic_estimate_function(None, None, None, vec_statistics)

        for sg, quality, stats in zip(next_level_candidates, qualities, statistics):
            add_if_required(self, result, sg, quality, task, statistics=stats, check_duplicates=self.check_duplicates)

        # Rest minimal quality
        min_quality = ps.minimum_required_quality(result, task)

        # Prune unpromising candidates
        for sg, optimistic_estimate, stats in zip(next_level_candidates, optimistic_estimates, statistics):
            # CHANGELIST. added monotone constrains check here
            if optimistic_estimate >= min_quality and ps.constraints_satisfied(task.constraints_monotone, sg, stats,
                                                                               task.data):
                promising_candidates.append(sg.selectors)
                self.sgs_eval_interesting += 1

        return promising_candidates

    def get_next_level(self, promising_candidates):
        precomputed_list = list(
            (tuple(sg), sg[-1], hash(tuple(sg[:-1])), tuple(sg[:-1])) for sg in promising_candidates
        )

        next_level = list(
            (*sg1, new_selector) for (sg1, _, hash_l, selectors_l), (_, new_selector, hash_r, selectors_r) in
            combinations(precomputed_list, 2)
            if (hash_l == hash_r) and (selectors_l == selectors_r))

        return next_level

    def execute(self, task, initial_result=None):
        # Enable passing result set to execute function
        if initial_result is None:
            initial_result = []
        else:
            self.check_duplicates = True

        # Original code
        if not isinstance(task.qf, ps.BoundedInterestingnessMeasure):
            raise RuntimeWarning("Quality function is unbounded, long runtime expected")

        task.qf.calculate_constant_statistics(task.data, task.target)

        with self.representation_type(task.data, task.search_space) as representation:
            combine_selectors = getattr(representation.__class__, self.combination_name)
            result = initial_result[:]
            # init the first level
            next_level_candidates = []
            for sel in task.search_space:
                next_level_candidates.append(combine_selectors([sel]))

            # level-wise search
            depth = 1
            while next_level_candidates:
                # check sgs from the last level
                promising_candidates = self.get_next_level_candidates_vectorized(task, result,
                                                                                 next_level_candidates)
                # Break now, as everything after this is work for the next/current level
                if depth == task.depth:
                    break

                next_level_candidates_no_pruning = self.next_level(promising_candidates)

                # select those selectors and build a subgroup from them
                #   for which all subsets of length depth (=candidate length -1) are in the set of promising candidates
                set_promising_candidates = set(tuple(p) for p in promising_candidates)
                next_level_candidates = [combine_selectors(selectors) for selectors in
                                         next_level_candidates_no_pruning
                                         if all(
                        (subset in set_promising_candidates) for subset in combinations(selectors, depth))]
                depth = depth + 1

            if initial_result:
                # Re-format init results because subgroup "stats" might be not supplied because quality is enough
                result = [(sg_tuple[0], sg_tuple[1], None) for sg_tuple in result]

            result.sort(key=lambda x: x[0], reverse=True)
            return ps.SubgroupDiscoveryResult(result, task)


class SGCountingApriori:
    """
    Pysubgroup's Apriori adapted and streamlined to count number of evaluated subgroups
    """

    def __init__(self, representation_type=None, combination_name='Conjunction', use_numba=False):
        self.combination_name = combination_name

        if representation_type is None:
            representation_type = ps.BitSetRepresentation
        self.representation_type = representation_type
        self.use_vectorization = True
        self.use_repruning = False
        self.optimistic_estimate_name = 'optimistic_estimate'
        self.next_level = self.get_next_level
        self.compiled_func = None

        # New
        self.sgs_evaluated = 0
        self.sgs_eval_interesting = 0
        self.sgs_quality_interesting = 0
        self.sgs_top_k_interesting = 0

    def get_next_level_candidates_vectorized(self, task, result, next_level_candidates):
        promising_candidates = []
        statistics = []
        optimistic_estimate_function = getattr(task.qf, self.optimistic_estimate_name)
        for sg in next_level_candidates:
            statistics.append(task.qf.calculate_statistics(sg, task.target, task.data))
            self.sgs_evaluated += 1

        tpl_class = statistics[0].__class__

        # Transform to vector
        vec_statistics = tpl_class._make(np.array(tpl) for tpl in zip(*statistics))

        # Use vector to get data
        qualities = task.qf.evaluate(None, task.target, task.data, vec_statistics)
        optimistic_estimates = optimistic_estimate_function(None, None, None, vec_statistics)

        for sg, quality, stats in zip(next_level_candidates, qualities, statistics):
            add_if_required(self, result, sg, quality, task, statistics=stats)

        # Rest minimal quality
        min_quality = ps.minimum_required_quality(result, task)

        # Prune unpromising candidates
        for sg, optimistic_estimate, stats in zip(next_level_candidates, optimistic_estimates, statistics):
            if optimistic_estimate >= min_quality:
                promising_candidates.append(sg.selectors)
                self.sgs_eval_interesting += 1

        return promising_candidates

    def get_next_level(self, promising_candidates):
        precomputed_list = list(
            (tuple(sg), sg[-1], hash(tuple(sg[:-1])), tuple(sg[:-1])) for sg in promising_candidates
        )

        next_level = list(
            (*sg1, new_selector) for (sg1, _, hash_l, selectors_l), (_, new_selector, hash_r, selectors_r) in
            combinations(precomputed_list, 2)
            if (hash_l == hash_r) and (selectors_l == selectors_r))

        return next_level

    def execute(self, task):
        # Original code
        if not isinstance(task.qf, ps.BoundedInterestingnessMeasure):
            raise RuntimeWarning("Quality function is unbounded, long runtime expected")

        task.qf.calculate_constant_statistics(task.data, task.target)

        with self.representation_type(task.data, task.search_space) as representation:
            combine_selectors = getattr(representation.__class__, self.combination_name)
            result = []
            # init the first level
            next_level_candidates = []
            for sel in task.search_space:
                next_level_candidates.append(combine_selectors([sel]))

            # level-wise search
            depth = 1
            while next_level_candidates:
                # check sgs from the last level
                promising_candidates = self.get_next_level_candidates_vectorized(task, result,
                                                                                 next_level_candidates)
                # Break now, as everything after this is work for the next/current level
                if depth == task.depth:
                    break

                next_level_candidates_no_pruning = self.next_level(promising_candidates)

                # select those selectors and build a subgroup from them
                #   for which all subsets of length depth (=candidate length -1) are in the set of promising candidates
                set_promising_candidates = set(tuple(p) for p in promising_candidates)
                next_level_candidates = [combine_selectors(selectors) for selectors in
                                         next_level_candidates_no_pruning
                                         if all(
                        (subset in set_promising_candidates) for subset in combinations(selectors, depth))]
                depth = depth + 1

            result.sort(key=lambda x: x[0], reverse=True)
            return ps.SubgroupDiscoveryResult(result, task)


class SGCountingDFS:
    """
    Implementation of a depth-first-search without look-ahead using a provided data structure.
    """

    def __init__(self, apply_representation):
        self.target_bitset = None
        self.apply_representation = apply_representation
        self.operator = None
        self.params_tpl = namedtuple('StandardQF_parameters', ('size_sg', 'positives_count'))

        # New
        self.sgs_evaluated = 0
        self.sgs_eval_interesting = 0
        self.sgs_quality_interesting = 0
        self.sgs_top_k_interesting = 0

    def execute(self, task):
        self.operator = ps.StaticSpecializationOperator(task.search_space)
        task.qf.calculate_constant_statistics(task.data, task.target)
        result = []
        with self.apply_representation(task.data, task.search_space) as representation:
            self.search_internal(task, result, representation.Conjunction([]))
        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)

    def search_internal(self, task, result, sg):
        statistics = task.qf.calculate_statistics(sg, task.target, task.data)
        self.sgs_evaluated += 1
        if not ps.constraints_satisfied(task.constraints_monotone, sg, statistics, task.data):
            return
        optimistic_estimate = task.qf.optimistic_estimate(sg, task.target, task.data, statistics)
        if not optimistic_estimate > ps.minimum_required_quality(result, task):
            return
        self.sgs_eval_interesting += 1
        quality = task.qf.evaluate(sg, task.target, task.data, statistics)
        add_if_required(self, result, sg, quality, task, statistics=statistics)

        if sg.depth < task.depth:
            for new_sg in self.operator.refinements(sg):
                self.search_internal(task, result, new_sg)


class SGCountingDFS_FC:
    """
    Implementation of a depth-first-search without look-ahead using a provided data structure.
        + forward checking (FC), also called look-ahead using levelwise eval/prune approach
    """

    def __init__(self, apply_representation):
        self.target_bitset = None
        self.apply_representation = apply_representation
        self.operator = None
        self.params_tpl = namedtuple('StandardQF_parameters', ('size_sg', 'positives_count'))

        # New
        self.sgs_evaluated = 0
        self.sgs_eval_interesting = 0
        self.sgs_quality_interesting = 0
        self.sgs_top_k_interesting = 0

    def execute(self, task):
        self.operator = ps.StaticSpecializationOperator(task.search_space)
        task.qf.calculate_constant_statistics(task.data, task.target)
        result = []
        with self.apply_representation(task.data, task.search_space) as representation:
            # Eval empty pattern (to keep as close to original Pysubgroup's DFS as possible which does this)
            empty_pattern = representation.Conjunction([])
            optimistic_estimate, statistics, evaled_sg = self.eval_sg(task, result, empty_pattern)
            if self.prune_sg(task, result, evaled_sg, optimistic_estimate, statistics):
                # Start dfs
                self.dfs_fc(task, result, empty_pattern)

        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)

    def eval_sg(self, task, result, sg):
        statistics = task.qf.calculate_statistics(sg, task.target, task.data)
        optimistic_estimate = task.qf.optimistic_estimate(sg, task.target, task.data, statistics)
        quality = task.qf.evaluate(sg, task.target, task.data, statistics)
        add_if_required(self, result, sg, quality, task, statistics=statistics)
        self.sgs_evaluated += 1

        return optimistic_estimate, statistics, sg

    def prune_sg(self, task, result, sg, optimistic_estimate, statistics):
        if not ps.constraints_satisfied(task.constraints_monotone, sg, statistics, task.data):
            return False  # sg is pruned
        if not optimistic_estimate > ps.minimum_required_quality(result, task):
            return False  # sg is pruned
        self.sgs_eval_interesting += 1

        return True  # sg is not pruned

    def dfs_fc(self, task, result, sg):

        # Do forward checking but first do eval (since it might change top k)
        next_sgs_evaled = [self.eval_sg(task, result, new_sg) for new_sg in self.operator.refinements(sg)]
        # and then prune (not at the same time, otherwise FC is (almost) useless
        next_sgs_pruned = [evaled_sg for optimistic_estimate, statistics, evaled_sg in next_sgs_evaled if
                           self.prune_sg(task, result, evaled_sg, optimistic_estimate, statistics)]

        # Default dfs step but with filtered next_sgs instead
        if (sg.depth + 1) < task.depth:  # +1 needed because all sgs of depth sg.depth +1 have been evaluated by FC
            for new_sg in next_sgs_pruned:
                self.dfs_fc(task, result, new_sg)
