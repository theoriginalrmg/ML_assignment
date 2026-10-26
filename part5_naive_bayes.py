import numpy as np

from common import prepare, evaluate_flat, print_matrix_row, pct, accuracy
from part1_decision_tree import DecisionTree
from part3_random_forest import RandomForest


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        K, d = int(y.max()) + 1, X.shape[1]
        # epsilon = 1e-9 * largest per-feature variance of the TRAINING data
        eps = self.var_smoothing * np.max(np.var(X, axis=0))
        self.log_prior_ = np.zeros(K)
        self.mean_ = np.zeros((K, d))
        self.var_ = np.zeros((K, d))
        for c in range(K):
            Xc = X[y == c]
            self.log_prior_[c] = np.log(len(Xc) / len(X))   # pi_c = n_c / N
            self.mean_[c] = Xc.mean(axis=0)
            self.var_[c] = Xc.var(axis=0) + eps              # divides by n_c, then adds epsilon
        return self

    def log_joint(self, X):
        X = np.asarray(X, dtype=float)
        diff = X[:, None, :] - self.mean_[None, :, :]          # (n, K, d) via broadcasting
        ll = -0.5 * np.sum(np.log(2 * np.pi * self.var_)[None] + diff ** 2 / self.var_[None], axis=2)
        return self.log_prior_[None, :] + ll

    def predict(self, X):
        return np.argmax(self.log_joint(X), axis=1)


def sanity_check():
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    y = np.array([0, 0, 1, 1])
    nb = GaussianNB().fit(X, y)
    assert np.allclose(nb.mean_.ravel(), [1, 11])
    assert np.allclose(nb.var_.ravel(), [1, 1])
    assert list(nb.predict(np.array([[3.0], [9.0]]))) == [0, 1]
    print("Sanity check: x=3 -> class 0, x=9 -> class 1: passed")


def main():
    sanity_check()
    d = prepare()
    print(f"Train rows: {len(d.ytr)}, test rows: {len(d.yte)}")

    nb = GaussianNB(var_smoothing=1e-9).fit(d.Xtr, d.ytr)
    tr, te, fam = evaluate_flat(nb, d)
    print()
    print_matrix_row("5. Gaussian Naive Bayes", "—", tr, te, fam)

    print("\nComparison with tree-based models (test accuracy):")
    print(f"  Gaussian NB:    {pct(te)}")
    for crit in ("gini", "entropy"):
        tree = DecisionTree(crit).fit(d.Xtr, d.ytr)
        rf = RandomForest(50, crit, max_features=4).fit(d.Xtr, d.ytr)
        print(f"  [{crit}] single tree: {pct(accuracy(d.yte, tree.predict(d.Xte)))}   "
              f"random forest: {pct(accuracy(d.yte, rf.predict(d.Xte)))}")

    # Correlation between the 22 MFCCs (training data). NB assumes these are
    # independent within each class; strongly correlated features get their
    # evidence double-counted.
    corr = np.corrcoef(d.Xtr, rowvar=False)
    off = np.abs(corr[np.triu_indices(22, k=1)])
    i, j = sorted(np.unravel_index(np.argmax(np.abs(corr) - np.eye(22)), corr.shape))
    print("\nCorrelation of the 22 training features (overall, not per class):")
    print(f"  mean |r| over the 231 pairs: {off.mean():.3f}")
    print(f"  pairs with |r| > 0.5: {(off > 0.5).sum()},  |r| > 0.7: {(off > 0.7).sum()}")
    print(f"  strongest pair: MFCCs_{i + 1} & MFCCs_{j + 1}, r = {corr[i, j]:+.3f}")

    # Within-class correlation is what actually matters for the NB assumption.
    # A feature that is constant inside a class has undefined correlation there
    # (division by zero std); those entries are skipped with nanmean.
    within = []
    for c in range(len(d.species_names)):
        Xc = d.Xtr[d.ytr == c]
        with np.errstate(invalid="ignore", divide="ignore"):
            cc = np.corrcoef(Xc, rowvar=False)
        within.append(np.nanmean(np.abs(cc[np.triu_indices(22, k=1)])))
    print(f"  mean |r| within each species (averaged over species): {np.mean(within):.3f}")

    # Features with (near-)zero variance inside a class: their sigma^2 is just epsilon,
    # so the Gaussian for that class becomes extremely narrow on that feature.
    eps = 1e-9 * np.max(np.var(d.Xtr, axis=0))
    for c, j in zip(*np.where(nb.var_ - eps < 1e-12)):        # var_ = class variance + eps
        print(f"  note: MFCCs_{j + 1} is constant within {d.species_names[c]} "
              f"(variance = epsilon = {eps:.2e})")

    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print("\nFull 22x22 correlation matrix (training data):")
    print(corr)


if __name__ == "__main__":
    main()
