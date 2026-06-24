"""
Evaluates visual classifiers against the demo_images dataset (300 images).
  - ResNet50 (resnet50_drone_weights.pth): binary, drone vs. bird/plane
  - best_cnn.pt: probed automatically
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEMO_DIR = Path("./demo_images")
WEIGHTS_DIR = Path("./ARL")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_dataset():
    """Returns list of (image_path, label_int, label_str). drone=1, bird=0, plane=0."""
    data = []
    for category in ["drones", "birds", "planes"]:
        for img_path in sorted((DEMO_DIR / category).glob("*.jpg")):
            label = 1 if category == "drones" else 0
            data.append((img_path, label, category))
    return data

def run_inference(model, data):
    preds, labels = [], []
    probs_list = []
    model.eval()
    with torch.no_grad():
        for img_path, label, _ in data:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            output = model(tensor)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob >= 0.5 else 0
            preds.append(pred)
            labels.append(label)
            probs_list.append(prob)
    return np.array(labels), np.array(preds), np.array(probs_list)

def confusion_matrix_manual(labels, preds):
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    return tp, tn, fp, fn

def print_metrics(name, labels, preds, probs, data):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    tp, tn, fp, fn = confusion_matrix_manual(labels, preds)
    accuracy  = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n  Accuracy:    {accuracy:.1%}  ({tp+tn}/{len(labels)} correct)")
    print(f"  Precision:   {precision:.1%}  (of drone predictions, how many were drones)")
    print(f"  Recall:      {recall:.1%}  (of actual drones, how many were caught)")
    print(f"  Specificity: {specificity:.1%}  (of non-drones, how many were cleared)")
    print(f"  F1 Score:    {f1:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Drone   Not-Drone")
    print(f"  Actual Drone    {tp:>5}   {fn:>9}")
    print(f"  Actual Not-Dr.  {fp:>5}   {tn:>9}")

    # Per-category breakdown
    print(f"\n  Per-category breakdown:")
    for cat in ["drones", "birds", "planes"]:
        cat_data = [(i, p, prob) for i, (_, lbl, c) in enumerate(data)
                    for p, prob in [(preds[i], probs[i])]
                    if c == cat]
        if cat_data:
            idxs = [x[0] for x in cat_data]
            cat_preds = preds[idxs]
            cat_labels = labels[idxs]
            correct = (cat_preds == cat_labels).sum()
            avg_conf = probs[idxs].mean()
            print(f"    {cat:8s}: {correct:3d}/100 correct  (avg confidence: {avg_conf:.1%})")

# ── ResNet50 ──────────────────────────────────────────────────────────────────
def evaluate_resnet50(data):
    weights_path = WEIGHTS_DIR / "resnet50_drone_weights.pth"
    if not weights_path.exists():
        print(f"[SKIP] {weights_path} not found")
        return

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    labels, preds, probs = run_inference(model, data)
    print_metrics("ResNet50  (resnet50_drone_weights.pth)", labels, preds, probs, data)

# ── best_cnn.pt ───────────────────────────────────────────────────────────────
def evaluate_best_cnn(data):
    weights_path = WEIGHTS_DIR / "best_cnn.pt"
    if not weights_path.exists():
        print(f"[SKIP] {weights_path} not found")
        return

    # Try as ResNet50 binary (same architecture, different weights)
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        labels, preds, probs = run_inference(model, data)
        print_metrics("best_cnn.pt  (loaded as ResNet50 binary)", labels, preds, probs, data)
        return
    except Exception as e:
        print(f"  [best_cnn.pt] ResNet50 load failed: {e}")

    # Fallback: try loading as a generic model and probe output shape
    try:
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        print(f"  [best_cnn.pt] checkpoint keys: {list(checkpoint.keys())[:10]}")
    except Exception as e:
        print(f"  [best_cnn.pt] could not load checkpoint: {e}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    data = load_dataset()
    print(f"Loaded {len(data)} images  ({sum(1 for _,l,_ in data if l==1)} drones, "
          f"{sum(1 for _,l,_ in data if l==0)} non-drones)")

    evaluate_resnet50(data)
    evaluate_best_cnn(data)
    print()
