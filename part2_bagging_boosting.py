import time

import numpy as np

from common import prepare, evaluate_flat, print_matrix_row, pct, accuracy
from part1_decision_tree import DecisionTree


def majority_vote(preds, n_classes):
    T, n = preds.shape
    counts = np.zeros((n, n_classes), dtype=int)
    rows = np.arange(n)
    for t in range(T):
        counts[rows, preds[t]] += 1
    return np.argmax(counts, axis=1)   


class Bagging:

    def __init__(self, n_estimators=25, criterion="gini", seed=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)   # one generator for this whole model
        n = X.shape[0]
        self.n_classes_ = int(y.max()) + 1
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n, size=n)      # bootstrap: N rows with replacement
            tree = DecisionTree(self.criterion, max_depth=None, min_samples_split=2)
            self.trees_.append(tree.fit(X[idx], y[idx]))
        return self

    def tree_predictions(self, X):
        return np.array([t.predict(X) for t in self.trees_])   # (T, n)

    def predict(self, X):
        return majority_vote(self.tree_predictions(X), self.n_classes_)


class AdaBoostSAMME:

    def __init__(self, n_estimators=50, max_depth=3, criterion="gini"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X, y):
        n = X.shape[0]
        K = int(y.max()) + 1
        self.n_classes_ = K
        w = np.full(n, 1.0 / n)
        self.trees_, self.alphas_ = [], []
        self.stopped_early_ = False
        for _ in range(self.n_estimators):
            tree = DecisionTree(self.criterion, max_depth=self.max_depth, min_samples_split=2)
            tree.fit(X, y, sample_weight=w)
            miss = tree.predict(X) != y
            eps = float(np.sum(w[miss]))                 # weights sum to 1
            eps = min(max(eps, 1e-10), 1 - 1e-10)        # clip to [1e-10, 1 - 1e-10]
            if eps >= 1.0 - 1.0 / K:                     # no better than random guessing
                self.stopped_early_ = True               # discard this learner and stop
                break
            alpha = np.log((1 - eps) / eps) + np.log(K - 1)
            w = w * np.exp(alpha * miss)                 # only misclassified weights grow
            w = w / w.sum()
            self.trees_.append(tree)
            self.alphas_.append(alpha)
        return self

    def predict(self, X):
        n = X.shape[0]
        scores = np.zeros((n, self.n_classes_))
        rows = np.arange(n)
        for tree, alpha in zip(self.trees_, self.alphas_):
            scores[rows, tree.predict(X)] += alpha
        return np.argmax(scores, axis=1)                 # ties -> smallest label


def main():
    d = prepare()
    print(f"Train rows: {len(d.ytr)}, test rows: {len(d.yte)}")

    single_test = {}
    for crit in ("gini", "entropy"):
        tree = DecisionTree(crit).fit(d.Xtr, d.ytr)
        single_test[crit] = accuracy(d.yte, tree.predict(d.Xte))

    print("\n=== 2a. Bagging (T=25, bootstrap size N, all 22 features, unconstrained) ===")
    for crit in ("gini", "entropy"):
        start = time.time()
        model = Bagging(25, crit).fit(d.Xtr, d.ytr)
        tr, te, fam = evaluate_flat(model, d)
        print(f"[{crit}] fit in {time.time() - start:.1f}s")
        print_matrix_row("2a. Bagging", crit.capitalize(), tr, te, fam)

    print("\n=== 2b. AdaBoost SAMME (T=50, max_depth=3) ===")
    depth3 = {}
    for crit in ("gini", "entropy"):
        start = time.time()
        model = AdaBoostSAMME(50, 3, crit).fit(d.Xtr, d.ytr)
        depth3[crit] = model
        tr, te, fam = evaluate_flat(model, d)
        note = " (stopped early)" if model.stopped_early_ else ""
        print(f"[{crit}] fit in {time.time() - start:.1f}s, rounds kept: {len(model.trees_)}{note}")
        print_matrix_row("2b. AdaBoost", crit.capitalize(), tr, te, fam)

    print("\n=== AdaBoost with stumps (max_depth=1, T=50) vs max_depth=3, test accuracy (not in matrix) ===")
    for crit in ("gini", "entropy"):
        stump = AdaBoostSAMME(50, 1, crit).fit(d.Xtr, d.ytr)
        deep3 = depth3[crit]
        note = " (stopped early)" if stump.stopped_early_ else ""
        print(f"[{crit}] stumps: {pct(accuracy(d.yte, stump.predict(d.Xte)))} "
              f"({len(stump.trees_)} rounds{note})   depth 3: {pct(accuracy(d.yte, deep3.predict(d.Xte)))}")

    print("\nSingle tree test accuracy for comparison:  "
          + "   ".join(f"{c}: {pct(a)}" for c, a in single_test.items()))


if __name__ == "__main__":
    main()
