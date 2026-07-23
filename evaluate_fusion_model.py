"""
Evaluation-only script for the fusion LogisticRegression classifier (ARL/fusion_model.pkl).

Does NOT touch train_fusion_step / ModalityFusionTransformer or any training code.
Reads the pickled model as-is and reports classification_report + confusion_matrix
instead of a single accuracy number, so false negatives (missed threats) and false
positives (false alarms) are visible separately.

Reproduces the exact feature construction / split used by the LogisticRegression
version of train_FS_MODEL (OlderVersions/rf-classification-cleaned-no-batching.ipynb):
    feature_cols = ["rf_confidence", "audio_mambo", "audio_bebop",
                    "audio_background", "visual_confidence"] + sensor_cols
    y = df["is_threat_gt"]
    train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

NOTE ON DATA: snapshot_log.csv is not present in the current working tree (removed
upstream by commit 71deaee). CSV_PATH below points at a version recovered from git
history (commit 71deaee^ / c25c75f, byte-identical at both) staged outside the repo.
Schema matches fusion_model.pkl's expected 51 features exactly, but exact provenance
against the Colab session that originally trained the pickle can't be proven -- pass
--csv to point at the real file if you have it.
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DEFAULT_MODEL_PATH = "./ARL/fusion_model.pkl"
DEFAULT_CSV_PATH = (
    "/private/tmp/claude-501/-Users-victoriaxiao-ARL/"
    "fe42c2c4-b78f-4842-a4f8-94df7b44a821/scratchpad/fusion_eval/snapshot_log.csv"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH)
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model: {type(model).__name__} from {args.model}")

    df = pd.read_csv(args.csv)
    df.fillna(0, inplace=True)

    sensor_cols = [col for col in df.columns if "sensor" in col]
    feature_cols = [
        "rf_confidence", "audio_mambo", "audio_bebop",
        "audio_background", "visual_confidence",
    ] + sensor_cols

    X = df[feature_cols]
    y = df["is_threat_gt"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Held-out test set: {len(X_test)} rows (train_test_split test_size=0.2, "
          f"random_state=42, stratify=y)")

    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Not Threat", "Threat"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("=== Confusion Matrix (raw 2x2) ===")
    print(cm)
    print()
    print(f"{'':>15s} {'Pred: Not Threat':>18s} {'Pred: Threat':>15s}")
    print(f"{'True: Not Threat':>15s} {tn:>18d} {fp:>15d}")
    print(f"{'True: Threat':>15s} {fn:>18d} {tp:>15d}")
    print()
    print(f"TN (correct clear)          = {tn}")
    print(f"FP (false alarm)            = {fp}")
    print(f"FN (missed threat)          = {fn}   <-- worst-case error for this system")
    print(f"TP (correctly caught threat) = {tp}")
    print()
    print(f"Accuracy: {(tn + tp) / (tn + fp + fn + tp):.4f}")


if __name__ == "__main__":
    main()
