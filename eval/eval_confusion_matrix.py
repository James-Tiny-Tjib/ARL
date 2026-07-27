"""
Read-only eval: compares deployed fusion_model.pkl (LogisticRegression) against
orphaned xgboostmodel.pkl (XGBoost) on master_snapshots_1000.parquet. Does not
import or modify train_fusion_step, ModalityFusionTransformer, or any training code.

Rebuilds the same 51-feature vector as train_FS_MODEL's feature_cols
(RF_Classification_Final.ipynb), since the parquet stores raw per-sensor
rf_logits/au_logits/vs_logits rather than those flat columns directly. See
build_feature_vector() below for the exact derivation.
"""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PARQUET_PATH = REPO_ROOT / "ARL" / "master_snapshots_1000.parquet"
DEFAULT_FUSION_MODEL_PATH = REPO_ROOT / "ARL" / "fusion_model.pkl"
DEFAULT_XGB_MODEL_PATH = REPO_ROOT / "ARL" / "xgboostmodel.pkl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR

NUM_SENSORS = 23
LABEL_NAMES = ["Not Threat", "Threat"]

FEATURE_COLS = ["rf_confidence", "audio_mambo", "audio_bebop", "audio_background", "visual_confidence"]
for _i in range(NUM_SENSORS):
    FEATURE_COLS += [f"sensor_{_i}_threat_conf", f"sensor_{_i}_friendly_conf"]

# Blue sequential ramp (dataviz skill reference palette), light -> dark.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def softmax(logits):
    x = np.asarray(logits, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def build_feature_vector(row):
    """Reconstructs the 51-feature train_FS_MODEL vector for one master_snapshot row."""
    sensor_list = row["sensor_list"]
    nearest = int(row["nearest_sensor"])

    triggered_threat_count = 0
    sensor_features = []
    for s in sensor_list:
        threat_conf, friendly_conf = 0.0, 0.0
        if s["triggered"] and s["rf_logits"] is not None:
            rf = softmax(s["rf_logits"])
            friendly_conf, threat_conf = float(rf[0]), float(rf[1])
            if threat_conf >= 0.5:
                triggered_threat_count += 1
        sensor_features.append(threat_conf)
        sensor_features.append(friendly_conf)

    rf_confidence = triggered_threat_count / len(sensor_list) * 100

    nearest_sensor = sensor_list[nearest]
    if nearest_sensor["au_logits"] is not None:
        au = softmax(nearest_sensor["au_logits"]) * 100
        audio_mambo, audio_bebop, audio_background = au.tolist()
    else:
        audio_mambo, audio_bebop, audio_background = 0.0, 0.0, 0.0

    visual_confidence = float(nearest_sensor["vs_logits"]) if nearest_sensor["vs_logits"] is not None else 0.0

    return [rf_confidence, audio_mambo, audio_bebop, audio_background, visual_confidence] + sensor_features


def load_dataset(parquet_path):
    df = pd.read_parquet(parquet_path)
    X = pd.DataFrame(df.apply(build_feature_vector, axis=1).tolist(), columns=FEATURE_COLS)
    y = df["is_threat_gt"].astype(int)
    return X, y


def evaluate_model(name, model_path, X, y):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n{'=' * 60}")
    print(f"{name}  ({type(model).__name__})")
    print(f"{'=' * 60}")
    print(f"\nConfusion matrix (rows=actual, cols=predicted, order={LABEL_NAMES}):")
    print(cm)
    print(f"\n{classification_report(y, y_pred, target_names=LABEL_NAMES)}")
    print(f"False Negative Rate (missed threats) = {fn}/{fn + tp} = {fnr:.4f}")
    print(f"False Positive Rate (false alarms)   = {fp}/{fp + tn} = {fpr:.4f}")

    return {"model_name": name, "model_type": type(model).__name__, "cm": cm, "fnr": fnr, "fpr": fpr}


def plot_confusion_matrix(result, output_path):
    cm = result["cm"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)

    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=200)
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABEL_NAMES, fontsize=12)
    ax.set_yticklabels(LABEL_NAMES, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
    ax.set_ylabel("Actual", fontsize=13, labelpad=10)
    ax.set_title(
        f"{result['model_name']}\n({result['model_type']})",
        fontsize=15, fontweight="bold", pad=14,
    )

    threshold = cm.max() * 0.6
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / total * 100 if total else 0.0
            color = "white" if count > threshold else "#0b0b0b"
            ax.text(
                j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=14, fontweight="bold", color=color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved heatmap -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET_PATH))
    parser.add_argument("--fusion-model", default=str(DEFAULT_FUSION_MODEL_PATH))
    parser.add_argument("--xgb-model", default=str(DEFAULT_XGB_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.parquet)
    print(f"Loaded {len(X)} rows from {args.parquet}")
    print(f"Ground truth (is_threat_gt) distribution: {y.value_counts().to_dict()}")

    results = [
        evaluate_model("Fusion Model (deployed)", args.fusion_model, X, y),
        evaluate_model("XGBoost Model (orphaned)", args.xgb_model, X, y),
    ]

    print(f"\n{'=' * 60}")
    print("OPERATIONAL SUMMARY (lower is better for both)")
    print(f"{'=' * 60}")
    for r in results:
        print(f"{r['model_name']:28s}  FNR (missed threats) = {r['fnr']:.4f}   FPR (false alarms) = {r['fpr']:.4f}")

    plot_confusion_matrix(results[0], output_dir / "confusion_matrix_fusion_model.png")
    plot_confusion_matrix(results[1], output_dir / "confusion_matrix_xgboost_model.png")


if __name__ == "__main__":
    main()
