# Sampling settings that are not connected to number of samples our time.
# These settings are theoretically the setting one should use to model certain enumeration strategies .
SPECIFIC_SAMPLING_PROFILES = {
    "Apriori-Levelwise": {  # Can not use timeout
        "level_wise": True,
        "outlier_removal": True
    },
    "Apriori-Pathwise": {  # Can use timeout
        "outlier_removal": True,
        "pre_sample": True
    },
    "DFS": {}  # No improvements for DFS, everything false, thus just empty
}
