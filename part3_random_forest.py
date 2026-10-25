import time

import numpy as np

from common import prepare, evaluate_flat, print_matrix_row, pct, accuracy
from part1_decision_tree import DecisionTree
from part2_bagging_boosting import Bagging, majority_vote


class RandomForest:
    def __init__(self, n_estimators=50, criterion="gini", max_features=4, seed=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features     # m = floor(sqrt(22)) = 4
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)   # one generator: bootstraps AND feature draws
        n = X.shape[0]
        self.n_classes_ = int(y.max()) + 1
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n, size=n)      # bootstrap sample of size N
            # The tree calls rng.choice(d, m, replace=False) at EVERY node.
            tree = DecisionTree(self.criterion, max_depth=None, min_samples_split=2,
                                max_features=self.max_features, rng=rng)
            self.trees_.append(tree.fit(X[idx], y[idx]))
        return self

    def tree_predictions(self, X):
        return np.array([t.predict(X) for t in self.trees_])   # (T, n)

    def predict(self, X):
        return majority_vote(self.tree_predictions(X), self.n_classes_)


def mean_pairwise_agreement(preds, n_classes):
    T, n = preds.shape
    counts = np.zeros((n, n_classes))
    rows = np.arange(n)
    for t in range(T):
        counts[rows, preds[t]] += 1
    agreeing = (counts * (counts - 1) / 2).sum(axis=1)
    return float(np.mean(agreeing / (T * (T - 1) / 2)))


def main():
    d = prepare()
    K = len(d.species_names)
    print(f"Train rows: {len(d.ytr)}, test rows: {len(d.yte)}")

    print("\n=== 3. Random Forest (T=50, m=4 features drawn at every split, unconstrained) ===")
    forests = {}
    for crit in ("gini", "entropy"):
        start = time.time()
        rf = RandomForest(50, crit, max_features=4).fit(d.Xtr, d.ytr)
        forests[crit] = rf
        tr, te, fam = evaluate_flat(rf, d)
        print(f"[{crit}] fit in {time.time() - start:.1f}s")
        print_matrix_row("3. Random Forest", crit.capitalize(), tr, te, fam)

    print("\n=== Comparison (test accuracy) and tree agreement (extra, not graded) ===")
    print("criterion | single tree | bagging | random forest | agreement bagging | agreement RF")
    for crit in ("gini", "entropy"):
        single = DecisionTree(crit).fit(d.Xtr, d.ytr)
        bag = Bagging(25, crit).fit(d.Xtr, d.ytr)
        rf = forests[crit]
        print(f"{crit:<9} | {pct(accuracy(d.yte, single.predict(d.Xte)))}      | "
              f"{pct(accuracy(d.yte, bag.predict(d.Xte)))}  | "
              f"{pct(accuracy(d.yte, rf.predict(d.Xte)))}        | "
              f"{pct(mean_pairwise_agreement(bag.tree_predictions(d.Xte), K))}            | "
              f"{pct(mean_pairwise_agreement(rf.tree_predictions(d.Xte), K))}")


if __name__ == "__main__":
    main()
