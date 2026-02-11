import re
from pathlib import Path
import pandas as pd


HEX_SAMPLE_DIR = Path("./sample_data")
MICA_ROOT = Path("./mica/tcga_splits")


def _norm_id(x):
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    if re.fullmatch(r"-?\d+(\.0+)?", s):
        return str(int(float(s)))
    return s


def _col_ids(df, col):
    if col not in df.columns:
        return []
    out = []
    for x in df[col].dropna().tolist():
        y = _norm_id(x)
        if y is not None:
            out.append(y)
    return out


def patient_from_slide(slide_id: str) -> str:
    s = str(slide_id).strip()
    if s.startswith("TCGA-"):
        parts = s.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
    if "_" in s:
        return s.split("_", 1)[0]
    if "." in s:
        return s.split(".", 1)[0]
    return s


def _strict_5fold(train_sets, val_sets, tag):
    errs = []
    if len(train_sets) != 5 or len(val_sets) != 5:
        return [f"[FAIL] {tag}: expected 5 folds, got {len(train_sets)}"]

    all_pat = set().union(*[train_sets[i] | val_sets[i] for i in range(5)])

    for i in range(5):
        inter = train_sets[i] & val_sets[i]
        if inter:
            errs.append(f"[FAIL] {tag}: fold{i} train/val overlap n={len(inter)} e.g. {sorted(list(inter))[:5]}")

    union_val = set().union(*val_sets)
    total_val = sum(len(s) for s in val_sets)
    if len(union_val) != total_val:
        errs.append(f"[FAIL] {tag}: val overlap across folds")

    if union_val != all_pat:
        miss = all_pat - union_val
        if miss:
            errs.append(f"[FAIL] {tag}: patients missing in val n={len(miss)} e.g. {sorted(list(miss))[:5]}")

    for i in range(5):
        exp_train = all_pat - val_sets[i]
        if train_sets[i] != exp_train:
            errs.append(f"[FAIL] {tag}: fold{i} train != all_pat - val")
    return errs


def check_hex(sample_dir: Path):
    csvs = sorted(sample_dir.glob("*.csv"))
    if not csvs:
        return [f"[FAIL] HEX: no csv in {sample_dir}"]

    if len(csvs) == 1:
        df = pd.read_csv(csvs[0])
        need = {"patient_train", "patient_val", "train", "val"}
        if not need.issubset(set(df.columns)):
            miss = sorted(list(need - set(df.columns)))
            return [f"[FAIL] HEX(single): {csvs[0].name} missing cols {miss}"]
        pt = set(_col_ids(df, "patient_train"))
        pv = set(_col_ids(df, "patient_val"))
        inter = pt & pv
        if inter:
            return [f"[FAIL] HEX(single): patient_train/patient_val overlap n={len(inter)} e.g. {sorted(list(inter))[:5]}"]
        return []

    if len(csvs) == 5:
        train_sets, val_sets = [], []
        for c in csvs:
            df = pd.read_csv(c)
            if {"patient_train", "patient_val"}.issubset(set(df.columns)):
                tr = set(_col_ids(df, "patient_train"))
                va = set(_col_ids(df, "patient_val"))
            else:
                tr = set(patient_from_slide(s) for s in _col_ids(df, "train"))
                va = set(patient_from_slide(s) for s in _col_ids(df, "val"))
            train_sets.append(tr)
            val_sets.append(va)
        return _strict_5fold(train_sets, val_sets, f"HEX(5fold) {sample_dir}")

    return [f"[FAIL] HEX: expected 1 or 5 csv, got {len(csvs)} in {sample_dir}"]


def check_mica(root: Path):
    if not root.exists():
        return [f"[FAIL] MICA: root not found {root}"]
    errs = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        csvs = sorted(d.glob("splits_*.csv"))
        if len(csvs) != 5:
            errs.append(f"[FAIL] MICA:{d.name}: expected 5 splits_*.csv, got {len(csvs)}")
            continue

        def _k(p):
            m = re.search(r"(\d+)", p.stem)
            return int(m.group(1)) if m else 999

        csvs = sorted(csvs, key=_k)

        train_sets, val_sets = [], []
        for i, c in enumerate(csvs):
            df = pd.read_csv(c)
            if not {"train", "val"}.issubset(set(df.columns)):
                errs.append(f"[FAIL] MICA:{d.name}: {c.name} missing train/val cols")
                continue

            tr_slides = _col_ids(df, "train")
            va_slides = _col_ids(df, "val")

            tr_pat = [patient_from_slide(s) for s in tr_slides]
            va_pat = [patient_from_slide(s) for s in va_slides]

            tr_set, va_set = set(tr_pat), set(va_pat)

            if tr_set & va_set:
                errs.append(f"[FAIL] MICA:{d.name} fold{i}: train/val patient overlap")
            if len(va_pat) != len(va_set):
                errs.append(f"[FAIL] MICA:{d.name} fold{i}: val not 1-slide-per-patient")

            train_sets.append(tr_set)
            val_sets.append(va_set)

        if len(train_sets) == 5 and len(val_sets) == 5:
            errs += _strict_5fold(train_sets, val_sets, f"MICA:{d.name}")
    return errs


errs = []
errs += check_hex(HEX_SAMPLE_DIR)
errs += check_mica(MICA_ROOT)

if errs:
    print("\n".join(errs))
else:
    print("ALL PASS")
