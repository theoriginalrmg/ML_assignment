BITS F464 Machine Learning - Assignment 1: Flat vs. Hierarchical Classification From Scratch

Environment
-----------
Python  3.13.17
numpy   2.5.3
pandas  3.0.5
No other third-party libraries are used (standard library: time, types).

Setup
-----
Put Frogs_MFCCs.csv (UCI Anuran Calls (MFCCs)) in the same folder as the .py files
and run every command from that folder. No absolute paths are used.

Commands
--------
python part1_decision_tree.py      # Part 1: single tree (Gini / entropy), sanity checks, depth sweep
python part2_bagging_boosting.py   # Part 2: Bagging (T=25) and AdaBoost SAMME (T=50, depth 3), stumps comparison
python part3_random_forest.py      # Part 3: Random Forest (T=50, m=4 per split), tree agreement
python part4_hierarchical.py       # Part 4: router + family experts, failure analysis, confusion matrices
python part5_naive_bayes.py        # Part 5: Gaussian Naive Bayes, comparison with trees, feature correlation

Lines starting with "MATRIX |" are the values for the results matrix; lines starting
with "SUPP |" (Part 4) fill the supplementary failure-analysis table.

Files
-----
common.py                 split function (as given, seed 42), data loading, label encoding,
                          species->family map (from training data), accuracy, confusion matrix
part1_decision_tree.py    DecisionTree class, reused by Parts 2-5
part2_bagging_boosting.py Bagging, AdaBoostSAMME
part3_random_forest.py    RandomForest
part4_hierarchical.py     HierarchicalClassifier
part5_naive_bayes.py      GaussianNB

Approximate running times: Part 1 ~10 s, Part 2 ~1.5 min, Part 3 ~1 min, Part 4 a few seconds, Part 5 ~15 s.
