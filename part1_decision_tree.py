import time

import numpy as np

from common import prepare, evaluate_flat, print_matrix_row, pct, accuracy

TOL = 1e-12  # gains closer than this are treated as equal (floating-point noise)


# ----------------------------------------------------------------------------
# Impurity functions. Input: class WEIGHT sums with shape (..., C).
# With equal weights these are just class counts, so the same code
# serves the plain tree and the weighted tree used by AdaBoost.
# ----------------------------------------------------------------------------
def gini(class_weights):
    cw = np.asarray(class_weights, dtype=float)
    tot = np.maximum(cw.sum(axis=-1, keepdims=True), 1e-300)
    p = cw / tot
    return 1.0 - np.sum(p * p, axis=-1)


def entropy(class_weights):
    cw = np.asarray(class_weights, dtype=float)
    tot = np.maximum(cw.sum(axis=-1, keepdims=True), 1e-300)
    p = cw / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p), 0.0)   # 0 log2 0 := 0
    return -np.sum(terms, axis=-1)


class DecisionTree:

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 max_features=None, rng=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if max_features is not None and rng is None:
            raise ValueError("max_features needs an rng")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng
        self._imp = gini if criterion == "gini" else entropy

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        self.n_features_ = d
        self.n_classes_ = int(y.max()) + 1
        w = np.full(n, 1.0 / n) if sample_weight is None else np.asarray(sample_weight, dtype=float)

        # W[i, c] = w_i if y_i == c else 0.  Summing rows of W gives class weight sums.
        W = np.zeros((n, self.n_classes_))
        W[np.arange(n), y] = w

        # Node storage: parallel lists, turned into arrays at the end.
        self._feat, self._thr, self._left, self._right, self._val, self._gain = [], [], [], [], [], []
        self._grow(X, W, depth=0)

        self.feature_ = np.array(self._feat, dtype=int)     # -1 marks a leaf
        self.threshold_ = np.array(self._thr, dtype=float)
        self.left_ = np.array(self._left, dtype=int)
        self.right_ = np.array(self._right, dtype=int)
        self.value_ = np.array(self._val, dtype=int)        # majority class at node
        self.gain_ = np.array(self._gain, dtype=float)
        del self._feat, self._thr, self._left, self._right, self._val, self._gain
        return self

    def _grow(self, X, W, depth):
        node = len(self._feat)
        class_w = W.sum(axis=0)
        self._feat.append(-1)
        self._thr.append(0.0)
        self._left.append(-1)
        self._right.append(-1)
        self._val.append(int(np.argmax(class_w)))   # weighted majority, ties -> smallest label
        self._gain.append(0.0)

        n = X.shape[0]
        is_pure = np.count_nonzero(class_w > 0) <= 1
        too_small = n < self.min_samples_split
        too_deep = self.max_depth is not None and depth >= self.max_depth
        if is_pure or too_small or too_deep:
            return node

        split = self._best_split(X, W, class_w)
        if split is None:                    # no split with positive gain -> leaf
            return node

        f, t, g = split
        go_left = X[:, f] <= t
        self._feat[node] = int(f)
        self._thr[node] = float(t)
        self._gain[node] = float(g)
        self._left[node] = self._grow(X[go_left], W[go_left], depth + 1)
        self._right[node] = self._grow(X[~go_left], W[~go_left], depth + 1)
        return node

    def _best_split(self, X, W, class_w):
        n, d = X.shape
        if self.max_features is None:
            feats = np.arange(d)
        else:
            # Fresh random subset at EVERY node (Random Forest). Sorted so that
            # ties are still broken by lower feature index.
            feats = np.sort(self.rng.choice(d, self.max_features, replace=False))

        Xf = X[:, feats]                                   # (n, f)
        order = np.argsort(Xf, axis=0, kind="stable")      # sort each feature once
        xs = np.take_along_axis(Xf, order, axis=0)         # sorted values (n, f)
        left = np.cumsum(W[order], axis=0)[:-1]            # (n-1, f, C)
        right = np.maximum(class_w - left, 0.0)            # (n-1, f, C)

        w_total = class_w.sum()
        w_left = left.sum(axis=-1)
        w_right = right.sum(axis=-1)
        gain = (self._imp(class_w)
                - (w_left / w_total) * self._imp(left)
                - (w_right / w_total) * self._imp(right))  # (n-1, f)

        # A cut is only valid between two DISTINCT consecutive values.
        valid = xs[1:] > xs[:-1]
        gain = np.where(valid, gain, -np.inf)

        best = gain.max()
        if not best > TOL:
            return None

        # Tie-break: lowest feature index first, then lowest threshold.
        # Columns are in increasing feature index; rows in increasing threshold.
        tie = gain >= best - TOL
        j = int(np.argmax(tie.any(axis=0)))
        i = int(np.argmax(tie[:, j]))
        threshold = 0.5 * (xs[i, j] + xs[i + 1, j])       # midpoint
        return feats[j], threshold, gain[i, j]

    # -------------------------------------------------------------- predict
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        node = np.zeros(X.shape[0], dtype=int)
        rows = np.arange(X.shape[0])
        while True:
            f = self.feature_[node]
            active = f >= 0                  # samples not yet at a leaf
            if not active.any():
                break
            a = rows[active]
            go_left = X[a, f[active]] <= self.threshold_[node[a]]
            node[a] = np.where(go_left, self.left_[node[a]], self.right_[node[a]])
        return self.value_[node]

    # ---------------------------------------------------------- inspection
    def depth(self):
        depths = np.zeros(len(self.feature_), dtype=int)
        for i in range(len(self.feature_)):            # parents come before children
            if self.feature_[i] >= 0:
                depths[self.left_[i]] = depths[self.right_[i]] = depths[i] + 1
        return int(depths.max())

    def n_leaves(self):
        return int(np.sum(self.feature_ < 0))


# ----------------------------------------------------------------------------
# Sanity checks from the assignment
# ----------------------------------------------------------------------------
def sanity_checks():
    # (a) counts (3, 3): Gini 0.5, entropy 1 bit
    assert abs(gini([3, 3]) - 0.5) < 1e-12
    assert abs(entropy([3, 3]) - 1.0) < 1e-12

    # (b) one split at t = 2.5 with information gain 1, training accuracy 1
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    t = DecisionTree("entropy").fit(X, y)
    assert t.feature_[0] == 0 and abs(t.threshold_[0] - 2.5) < 1e-12
    assert abs(t.gain_[0] - 1.0) < 1e-12
    assert t.n_leaves() == 2 and accuracy(y, t.predict(X)) == 1.0

    # (c) three identical samples, labels [0,0,1], weights [0.1,0.1,0.8] -> leaf predicting 1
    X = np.zeros((3, 1))
    for crit in ("gini", "entropy"):
        t = DecisionTree(crit).fit(X, np.array([0, 0, 1]), sample_weight=np.array([0.1, 0.1, 0.8]))
        assert t.n_leaves() == 1 and np.all(t.predict(X) == 1)

    # (d) a pure node / single-class data -> one leaf, no crash
    for crit in ("gini", "entropy"):
        t = DecisionTree(crit).fit(np.random.RandomState(0).randn(5, 3), np.array([4] * 5))
        assert t.n_leaves() == 1 and np.all(t.predict(np.zeros((2, 3))) == 4)

    print("Sanity checks (a)-(d): all passed")


def main():
    sanity_checks()
    d = prepare()
    print(f"Train rows: {len(d.ytr)}, test rows: {len(d.yte)}")

    print("\n=== Flat single tree, predicting Species (max_depth=None) ===")
    for crit in ("gini", "entropy"):
        start = time.time()
        tree = DecisionTree(crit, max_depth=None, min_samples_split=2).fit(d.Xtr, d.ytr)
        tr, te, fam = evaluate_flat(tree, d)
        print(f"[{crit}] fit in {time.time() - start:.2f}s, depth {tree.depth()}, leaves {tree.n_leaves()}")
        print_matrix_row("1. Single Decision Tree", crit.capitalize(), tr, te, fam)

    print("\n=== Depth sweep (for understanding only, not part of the matrix) ===")
    for crit in ("gini", "entropy"):
        print(f"\n[{crit}]\n  max_depth | train % |  test %")
        for md in (1, 2, 3, 5, 8, 12, None):
            tree = DecisionTree(crit, max_depth=md).fit(d.Xtr, d.ytr)
            print(f"  {str(md):>9} | {pct(accuracy(d.ytr, tree.predict(d.Xtr)))}  | "
                  f"{pct(accuracy(d.yte, tree.predict(d.Xte)))}")


if __name__ == "__main__":
    main()
