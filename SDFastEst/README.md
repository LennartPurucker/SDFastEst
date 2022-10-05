# Pratcial Improvements to the Algorithm 

Besides building a sample-top-k-set, additional improvements were found to make the estimation of SDFastEst more accurate and lessen the impact of top-k SD. Some of these improvements might also be applicable to FastEst. 

## Algorithm Profiles 
The exact realization of SDFastEst depends on the algorithm. Therefore, multiple different algorithm profiles have been created for different implementations of SD algorithms. These profiles include information specific to the implementation like the employed pruning techniques or whether the empty pattern is evaluated
The algorithm profiles try to capture the major difference in implementation and algorithmic components. Each profile was created by doing a source code analysis of an implementation. The profiles are used as input to the sampling framework to appropriately change the sampling procedure such that it induces a tree closer to the original algorithm’s tree.

## Level wise
Another improvement was to alter the sampling procedure to support level-wise sampling. 
In level-wise sampling, all sample paths are built at once. For each level (or while any sampled path is not at a leaf), all samples evaluate their extensions and the extensions are checked for pruning only after each sample does so. 
This creates a top-k-set closer to the algorithm’s top-k-set when considering algorithms with a level-wise-search enumeration strategy. Moreover, it makes the samples less dependent on one another and thus improves the sampling estimation. 

## Pre-Sampling
The default path-wise sampling can also be improved for level-wise-search through the usage of pre-sampling.
That is, the sampling procedure initially samples paths without storing the result while the sample-top-k-set is prefilled. 
Without pre-sampling, the initial samples are not pruned enough, since the sample-top-k-set created during these initial samples is far from the real top-k-set. 
With such under-pruned samples, the initial samples’ estimates are hugely overestimated and thus deteriorate the overall estimation. 

## Outlier removal, trimmed mean 
Additionally, the aggregation method of the final estimates can be improved. Instead of taking the average of all found path estimates outliers can be removed before taking the average. To do so, we used a trimmed mean. Removing outliers is only appropriate because of dependent sampling. 
Too large or too small path estimates, which result from an inaccurate sample-top-k-set, can be removed. 
For example, imagine a sampled path that goes to depth 4, even though a sample-top-k-set closer to the original top-k-set would have pruned such a path after depth 2. This path’s estimate might be extremely large in comparison to all other path estimates, which were taken later with a more accurate sample-top-k-set. Hence, they are outliers.

## Missing generalizability of improvements 
It is important to remark that not all improvements are appropriate for all algorithms. 
For instance, Pysubroup's DFS benefits from none of the improvements above. For DFS, it was only appropriate to adjust the computation of the compensation value.

# Supported Algorithms
Besides the default implementations of Pysubgroup, we support additonal algorithms. 

In Pysubgroup's Apriori implementation, anti-monotone constraint pruning was not implemented by default. Hence, a corresponding fix was applied to allow Apriori to perform anti-monotone constraint pruning by default (called "Apriori-fixed").
Moreover, DFS is by default pure DFS. That is, no extensions like forward checking are implemented. Here, an extension of DFS to do forward checking is implemented (called "DFS-FC"). 

Moreover, we also created an example profile for VIKAMINE. 



# Longer-Text Details on Parameters

## dfs_refinements

The set of possible extensions of a subgroup is different between a DFS approach and a level-wise approach. The full set
of extensions, consisting of all selectors, is reduced by the removal of mutually exclusive selectors, the removal of a
picked and extended extension (i.e., one per level), and the details of the specialization operator (also called
refinement operator). To illustrate, Pysubgroup’s specialization operator makes sure that no subgroup is enumerated
twice by employing a static order on the attributes. As a result, the set of extensions of a subgroup is a specific set
of selectors based on the attribute order. In this order, the number of extensions becomes smaller and smaller the
further to the right in the search tree the subgroup is located. Selectors of the right-most attribute have no
extensions, while the selectors of the left-most have all other selectors as extensions (expect for mutually exclusive
selectors). On the contrary, a level-wise approach removes selectors that can be pruned from the set of extensions
during previous levels.

Consequently, none of the practical improvements described in the paper and used by the algorithm seemed appropriate.
Nevertheless, a change to the compensation value seemed appropriate. The compensation value is increased by the average
of two values. The first value is the number of selectors that Pysubgroup's specialization operator would enumerate for
a subgroup but which are not in the set of possible extensions in the current level of the path sample. This shall
capture selectors that are not evaluated because the sampling approach removed these prune-able selectors in the
previous level. The second value is the compensation value of the last level. This shall capture, similar to DFS, that
the number of extensions of the current level is almost the same as the number of extensions in a previous level. The
first value would generally underestimate the compensation value, because it does not include duplicates like the rest
of the sampling tree. The second value would generally overestimate the compensation value, because it also counts
selectors that are already pruned. Hence, the average is used. 
