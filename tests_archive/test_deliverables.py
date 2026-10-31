"""Sanity tests of the deliverables. Run:  python -m pytest tests -q   (from the notebook working directory)."""
import os, re, glob
import pandas as pd

EMOS = ["joy", "sadness", "fear", "anger", "surprise", "disgust"]
TEST_SIZE = {"ary": 812, "arq": 902}                 # official test sizes (assignment)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results", "results.csv")
PRED_FILES = sorted(glob.glob(os.path.join(ROOT, "predictions", "*_seed*.csv")))


def test_results_csv_schema():
    df = pd.read_csv(RES)
    assert {"part", "model", "lang", "split", "macro_f1", "extra"} <= set(df.columns)
    assert df.macro_f1.between(-1, 1).all()      # F1 in [0,1]; Pearson r (Track B) in [-1,1]
    assert set(df.lang) <= {"ary", "arq"}
    assert set(df.split) <= {"val", "test"}


def test_test_set_only_scored_in_part_F():
    """protocol: before Part F nothing is scored on test; test rows may only come from F / G"""
    df = pd.read_csv(RES); t = df[df.split == "test"]
    assert len(t) > 0, "no test result logged (Part F not run?)"
    assert t.part.str.match(r"^[FG]").all(), t[~t.part.str.match(r"^[FG]")]


def test_predictions_folder_not_empty():
    assert len(PRED_FILES) > 0


def test_prediction_file_names():
    for f in PRED_FILES:
        assert re.fullmatch(r".+_seed\d+\.csv", os.path.basename(f)), f


def test_prediction_file_format():
    cols = (["id", "lang", "seed"] + ["y_true_" + e for e in EMOS] + ["y_pred_" + e for e in EMOS] + ["p_" + e for e in EMOS])
    for f in PRED_FILES:
        d = pd.read_csv(f)
        assert list(d.columns) == cols, f
        for l, g in d.groupby("lang"):
            assert len(g) == TEST_SIZE[l], (f, l, len(g))
            assert g.id.is_unique, (f, l)
        for e in EMOS:
            assert set(d["y_true_" + e].unique()) <= {0, 1} and set(d["y_pred_" + e].unique()) <= {0, 1}, f
            assert d["p_" + e].between(0, 1).all(), f


def test_same_test_labels_for_every_model():
    ref = {}
    for f in PRED_FILES:
        d = pd.read_csv(f)
        for l, g in d.groupby("lang"):
            y = g.sort_values("id")[["id"] + ["y_true_" + e for e in EMOS]].reset_index(drop=True)
            if l in ref: assert ref[l].equals(y), (f, l)
            else: ref[l] = y
