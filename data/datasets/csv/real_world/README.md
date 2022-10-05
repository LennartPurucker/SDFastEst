We can not re-distribute the preprocessed real-world files. Please use the provided scripts to preprocess the original
datasets accordingly.

The sources of the original datasets are described in the paper. The names are described in the following.

# Hand-selected Datasets

The following datasets have been hand-selected from UCI:

```
adult, mushroom (agaricus-lepiota), balance-scale, breast-cancer-wisconsin,
census-income, german credit, heart (statlog),
house-votes-84 (congressional voting records), iris, chess kr-vs-kp, seismic-bumps,
tic-tac-toe, divorce, in-vehicle-coupon-recommendation, online_shoppers_intention,
covtype (Covertype), diabetic (Diabetes 130-US hospitals 1999-2008),
default_credit_card_client, and polish_bankruptcy_1/2/3/4/5
```

Additionally, the
[_healthcare-dataset-stroke-data_](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
from Kaggle, the _marketing dataset_ from Elements of Statistical Learning, and the
[_dry\_beans_](https://data.world/makeovermonday/2021w14) dataset from data.world have been selected.

## Preprocessing Remarks

To some of these datasets, specific preprocessing was applied. If a dataset was split into train and test data files
originally, these files were merged. For _breast-cancer-wisconsin_ the column corresponding to the ID number was
removed. For _mushroom (agaricus-lepiota)_, the missing value for _stalk-root_ was not removed, because it is a viable
value. For _Congressional Voting Records_, missing values are not removed since they are not actual missing values but a
valid observation in politics. For _census income_, the columns "migration code-change in msa", "migration code-change
in reg", "migration code-move within reg", "migration prev res in sunbelt" were dropped, because these had (very)
consistently missing values. If one would drop rows based on these missing values, almost half of the data would be
dropped. Furthermore, a column, which was not properly defined in the data description, was dropped (it could be
either "instance weight" or "federal income tax liability"). Per description, the column should be nominal with 10
values. However, if it is read as a string or integer, it has either 120683 or 6125 distinct values.

# Datasets from Meta-Learning Literature

From the paper "Instance spaces for machine learning classification"[1] the following 194 datasets have been selected.
These were originally taken from the data sources UCI, KEEL or DCoL.

```` 
abalone, abalone_ori, audiology_std, auto1, auto2, auto4, auto5, auto6_1, auto6_2, auto6_3, auto7_1, auto7_2, auto7_3,
auto8, bach, banknote, blogger, blood, breast_cancer_wis_ori, breast_cancer_wis_pro, breast_cancer_wis_pro2,
breast_tissue, breast_tissue_merged, car, cardio3, cardio10, climate, cnae9, connectionist_sonar,
connectionist_vowel, contraceptive, credit, dermatology, dresses, echocardio, ecoli, fertility, firm, first, forest,
gesture_raw, gesture_va, glass, grammatical_a1, grammatical_a2, grammatical_a3, grammatical_a4, grammatical_a5,
grammatical_a6, grammatical_a7, grammatical_a8, grammatical_a9, grammatical_b1, grammatical_b2, grammatical_b3,
grammatical_b4, grammatical_b5, grammatical_b6, grammatical_b7, grammatical_b8, grammatical_b9, haberman, hayes,
heart_cleverland, heart_hungarian, heart_va, heart_switzerland, hepatitis, hiv_746, hiv_1625, hiv_impens,
hiv_schilling, hv_no_noise, hv_noise, horse_colic_lesion, horse_colic_outcome, human, ilpd, image, internet, ionosphere,
isolet, japanese, leaf, lenses, libras, lsvt, lung, madelon, mammographics, mechanical, mice, molecular_promoter,
molecular_splice, monks1, monks2, monks3, onehund_mar, onehund_sha, onehund_tex, optical, ozone1, ozone8, page, pamap2,
parkinson_speech, parkinsons, pen, phishing, pima, pitt1_1, pitt1_2, pitt1_3, pitt1_4, pitt2_1, pitt2_3, pitt2_4,
pitt2_2, planning, post, primary, qsar, qualitative, robot1, robot2, robot3, robot4, robot5, secom, seeds,
soybean_large, soybean_small, spambase, spect_heart, spectf_heart, statlog_australian, statlog_is, statlog_ls,
statlog_vehicle, teaching, thyroid_new_thyroid, thyroid_allbp, thyroid_allhyper, thyroid_allhypo, thyroid_allrep,
thyroid_ann, thyroid_dis, thyroid_hypothyroid, thyroid_sick, trains, turkiye, urban, user, vert2, vert3, wall2, wall4,
wall24, wave, wave2, weight, wholesale, wilt, wine, wine_quality, winsconsin, yeast, zoo, appendicitis, australian,
automobile, banana, bankruptcy, breast, bupa, coil2000, crx, flare, led, lymphography, phoneme, ring, saheart, satimage,
tae, texture, titanic, twonorm, asbestos, fourclass, liver, tao, chronic_kidney_disease, wpbc
````

From the paper "Classifier Recommendation Using Data Complexity Measures"[2] the following 56 datasets have been
selected. These were originally taken from the data source OpenML.

```` 
cmc, PopularKids, volcanoes-a3, analcatdata_creditscore, ar4, heart-long-beach, volcanoes-a4, mfeat-morphological,
analcatdata_boxing2, pc3, musk,  mc1, mw1, prnn_synth, prnn_crabs, pc1, diggle_table_a2, pc4, visualizing_livestock,
ada_agnostic, volcanoes-a1, volcanoes-a2, scene, jEdit_4.0_4.2, synthetic_control, volcanoes-d4, analcatdata_authorship,
kc1-binary, analcatdata_germangss, pc1_req, volcanoes-d3, volcanoes-b5, backache, analcatdata_lawsuit, datatrieve, mc2,
mfeat-karhunen, mfeat-factors, acute-inflammations, kc3, mfeat-fourier, volcanoes-d1, thoracic-surgery,
analcatdata_dmft, steel-plates-fault, mfeat-zernike, badges2, oil_spill, rmftsa_sleepdata, semeion, mfeat-pixel,
credit-g, pc2, volcanoes-d2, analcatdata_boxing1, jEdit_4.2_4.3
```` 

We decided to drop columns instead of rows in the case of missing values to keep more instances for multiple datasets.
This was done for:

```
thyroid_hypothyroid, secom, internet, audiology_std, heart_hungarian, heart_va
```

# References

[1] M. A. Mu ̃noz, L. Villanova, D. Baatar, and K. Smith-Miles, “Instance spaces for machine learning classification,”
Machine Learning, vol. 107, no. 1, pp. 109–147, 2018.

[2] L. P. Garcia, A. C. Lorena, M. C. de Souto, and T. K. Ho, “Classifier recommendation using data complexity
measures,” in 2018 24th Inter- national Conference on Pattern Recognition (ICPR). IEEE, 2018, pp. 874–879. 