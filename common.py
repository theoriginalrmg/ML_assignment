from types import SimpleNamespace

import numpy as np
import pandas as pd

DATA_FILE = "Frogs_MFCCs.csv"  # relative path: run scripts from the folder holding the CSV


# ----------------------------------------------------------------------------
# The split function, copied exactly from the assignment PDF.
# ----------------------------------------------------------------------------
def stratified_split(labels, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    test = []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        test.extend(idx[:int(round(test_frac * len(idx)))])
    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(labels)), test)
    return train, test


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # The CSV names columns "MFCCs_ 1" ... "MFCCs_22"; sort them by their number.
    mfcc_cols = [c for c in df.columns if c.startswith("MFCCs")]
    mfcc_cols.sort(key=lambda c: int(c.split("_")[-1]))
    assert len(mfcc_cols) == 22, f"expected 22 MFCC columns, found {len(mfcc_cols)}"
    X = df[mfcc_cols].to_numpy(dtype=float)
    species = df["Species"].astype(str).str.strip().to_numpy()
    family = df["Family"].astype(str).str.strip().to_numpy()
    return X, species, family


def prepare(path=DATA_FILE):
    X, species, family = load_data(path)
    train, test = stratified_split(species)          # stratified on Species, seed 42

    species_names, y_sp = np.unique(species, return_inverse=True)
    family_names, y_fam = np.unique(family, return_inverse=True)

    # species -> family mapping, read from the TRAINING rows only
    sp2fam = np.full(len(species_names), -1, dtype=int)
    for s in range(len(species_names)):
        fams = np.unique(y_fam[train][y_sp[train] == s])
        assert len(fams) == 1, "each species must belong to exactly one family"
        sp2fam[s] = fams[0]

    return SimpleNamespace(
        Xtr=X[train], Xte=X[test],
        ytr=y_sp[train], yte=y_sp[test],            # species labels (10 classes)
        ftr=y_fam[train], fte=y_fam[test],          # family labels (4 classes)
        species_names=species_names, family_names=family_names,
        sp2fam=sp2fam,
    )


# ----------------------------------------------------------------------------
# Metrics (numpy only, no library helpers)
# ----------------------------------------------------------------------------
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def print_confusion(cm, names, title):
    short = [n[:10] for n in names]
    width = max(6, max(len(s) for s in short) + 1)
    print(f"\n{title}  (rows = true, cols = predicted)")
    print(" " * width + "".join(f"{i:>6d}" for i in range(len(names))))
    for i, row in enumerate(cm):
        print(f"{short[i]:<{width}}" + "".join(f"{v:>6d}" for v in row) + f"   [{i}] {names[i]}")


def pct(a):
    return f"{100 * a:6.2f}"


def evaluate_flat(model, d):
    p_tr = model.predict(d.Xtr)
    p_te = model.predict(d.Xte)
    tr = accuracy(d.ytr, p_tr)
    te = accuracy(d.yte, p_te)
    fam = accuracy(d.fte, d.sp2fam[p_te])   # map predicted species -> family
    return tr, te, fam


def print_matrix_row(method, criterion, tr, te, fam):
    fam_s = "    —" if fam is None else pct(fam)
    print(f"MATRIX | {method:<28} | {criterion:<7} | train {pct(tr)} | test {pct(te)} | family {fam_s}")
