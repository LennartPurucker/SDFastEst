from applications import sss_spectrums
from general_utils.sd_utils import WRAcc

# (dataset_name, to evaluate algorithm settings, sample_settings_name)
# to evaluate algorithm settings: algorithm_name, target_value, quality_function, sdfastest_algo_profile_name
examples = [
    ("Adult_UCI_", ("Apriori_DP (PS)", ">50K", WRAcc, "PS-APRIORI"), "Apriori-Levelwise"),
    ("Adult_UCI_", ("Apriori_DP_fixed (PS)", ">50K", WRAcc, "PS-APRIORI-FIXED"), "Apriori-Levelwise"),
    ("Adult_UCI_", ("DFS_BSD_DP (PS)", ">50K", WRAcc, "PS-DFS"), "DFS")
]

for input_tuple in examples:
    # ---- Some Basic Usage Documentation
    # First interval defines the range to be included in the spectrum,
    # input_tuple defines settings as described above,
    # n_points defines number of samples within the range
    # n_samples_per_point sets number of samples SDFastEst should use per point
    # get_true_val sets that the true number of sgs is collected to compare it to the estimate (by solving the SD task)

    sss_spectrums.sd_spectra_min_quality([0, 0.2], input_tuple, n_points=100, n_sample_per_point=100, get_true_val=True)

    sss_spectrums.sd_spectra_min_sg_size([0, 7000], input_tuple, n_points=100, n_sample_per_point=100,
                                         get_true_val=True)

    sss_spectrums.sd_spectra_result_set_size([0, 2000], input_tuple, n_points=20, n_sample_per_point=100,
                                             get_true_val=True)

    sss_spectrums.sd_spectra_depth([1, 6], input_tuple, n_points=6, n_sample_per_point=100, get_true_val=True)
