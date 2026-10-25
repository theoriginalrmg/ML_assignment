import numpy as np

from common import (prepare, accuracy, confusion_matrix, print_confusion,
                    print_matrix_row, pct)
from part1_decision_tree import DecisionTree


class HierarchicalClassifier:

    def __init__(self, router, experts):
        self.router = router
        self.experts = experts
        self.family_names = sorted(experts)

    def predict_family(self, X):
        return self.router.predict(X)

    def predict_given_family(self, X, fam):
        out = np.empty(X.shape[0], dtype=int)
        for code, name in enumerate(self.family_names):
            mask = fam == code
            if mask.any():
                out[mask] = self.experts[name].predict(X[mask])
        return out

    def predict(self, X):
        return self.predict_given_family(X, self.predict_family(X))   # route first, then predict


def build(d, crit):
    router = DecisionTree(crit).fit(d.Xtr, d.ftr)                 # all training rows, Family label
    experts = {}
    for code, name in enumerate(d.family_names):
        mask = d.ftr == code                                      # TRUE family, never predicted
        experts[name] = DecisionTree(crit).fit(d.Xtr[mask], d.ytr[mask])
    return HierarchicalClassifier(router, experts)


def main():
    d = prepare()
    n_fam, n_sp = len(d.family_names), len(d.species_names)
    print(f"Train rows: {len(d.ytr)}, test rows: {len(d.yte)}")

    print("\nSpecies per family (training data):")
    for code, name in enumerate(d.family_names):
        sp = np.unique(d.ytr[d.ftr == code])
        print(f"  {name:<16} {len(sp)} species: {', '.join(d.species_names[sp])}")

    supp = []
    for crit in ("gini", "entropy"):
        print(f"\n================ Hierarchical tree, criterion = {crit} ================")
        h = build(d, crit)

        fam_tr, fam_te = h.predict_family(d.Xtr), h.predict_family(d.Xte)
        sp_tr, sp_te = h.predict(d.Xtr), h.predict(d.Xte)
        oracle_te = h.predict_given_family(d.Xte, d.fte)          # perfect router

        r_tr, r_te = accuracy(d.ftr, fam_tr), accuracy(d.fte, fam_te)
        o_tr, o_te = accuracy(d.ytr, sp_tr), accuracy(d.yte, sp_te)
        a_oracle = accuracy(d.yte, oracle_te)
        router_right = fam_te == d.fte
        p_a = float(np.mean(router_right & (sp_te != d.yte)))     # router right, expert wrong
        p_b = 1.0 - r_te                                          # router wrong

        print(f"Router accuracy   train {pct(r_tr)}   test {pct(r_te)}")
        print(f"Overall accuracy  train {pct(o_tr)}   test {pct(o_te)}")
        print(f"Oracle accuracy (test, perfect router): {pct(a_oracle)}")
        print(f"p_A (router right, expert wrong): {pct(p_a)}   p_B (router wrong): {pct(p_b)}")
        diff = abs(o_te - (r_te - p_a))
        print(f"Check A_overall = A_router - p_A:  {o_te:.6f} vs {r_te - p_a:.6f}  "
              f"(|diff| = {diff:.1e}) -> {'OK' if diff < 1e-12 else 'MISMATCH'}")
        print(f"Price of router errors (A_oracle - A_overall): {pct(a_oracle - o_te)}")

        print_confusion(confusion_matrix(d.fte, fam_te, n_fam), d.family_names,
                        "Router confusion matrix (test, 4x4)")
        print_confusion(confusion_matrix(d.yte, sp_te, n_sp), d.species_names,
                        "Final species confusion matrix (test, 10x10)")

        print("\nPer-family test accuracy (grouped by TRUE family):")
        print(f"  {'family':<16} {'n':>5} | router % | species % | oracle species %")
        for code, name in enumerate(d.family_names):
            m = d.fte == code
            print(f"  {name:<16} {m.sum():>5} |  {pct(accuracy(d.fte[m], fam_te[m]))}  |  "
                  f"{pct(accuracy(d.yte[m], sp_te[m]))}   |  {pct(accuracy(d.yte[m], oracle_te[m]))}")

        flat = DecisionTree(crit).fit(d.Xtr, d.ytr)
        flat_sp = flat.predict(d.Xte)
        print("\nComparison with the flat single tree (Part 1), test set:")
        print(f"  species: hierarchical {pct(o_te)}  vs flat {pct(accuracy(d.yte, flat_sp))}")
        print(f"  family:  router       {pct(r_te)}  vs flat derived {pct(accuracy(d.fte, d.sp2fam[flat_sp]))}")

        print()
        print_matrix_row("4. Hierarchical (overall)", crit.capitalize(), o_tr, o_te, r_te)
        supp.append((crit, r_te, a_oracle, p_a, p_b))

    print("\nSupplementary table (test, %):  criterion | router | oracle | (A) | (B)")
    for crit, r, o, a, b in supp:
        print(f"SUPP | {crit.capitalize():<7} | {pct(r)} | {pct(o)} | {pct(a)} | {pct(b)}")


if __name__ == "__main__":
    main()
