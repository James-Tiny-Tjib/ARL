"""
data_loading.py
===============
Generates synchronized (RF, Audio, Video, label) training samples,
saves them to a Parquet file, and optionally pushes to Hugging Face.

Usage (in notebook):
    # Generate and save locally
    df = build_and_save_dataset(n_per_class=2000, save_path="./ARL/multimodal_dataset.parquet")

    # Push to Hugging Face
    push_to_huggingface(parquet_path="./ARL/multimodal_dataset.parquet",
                        repo_id="your-username/drone-multimodal-dataset")

    # Load back for training
    dataset = MultiModalDataset(parquet_path="./ARL/multimodal_dataset.parquet")
"""

import os, random
import numpy as np
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset
from torchvision import transforms, datasets as tv_datasets
from PIL import Image
from io import BytesIO
import base64

# ── Paths — inherit from notebook kernel if defined ───────────────────────────
try:
    _audio_dir  = AUDIO_DATASET_DIR
    _visual_dir = VISUAL_DATASET_DIR
except NameError:
    _audio_dir  = "./DroneAudioDataset/Multiclass_Drone_Audio"
    _visual_dir = "./drone_demo_dataset"

visual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── RF: synthetic IQ generation ───────────────────────────────────────────────

def generate_rf_samples(n_per_class: int = 2000):
    """
    Generates synthetic IQ samples.
    label 0 = friendly (BPSK/QPSK), label 1 = adversary (16-QAM injected).
    Returns X: [2N, 2, 128] tensor, y: [2N] tensor.
    """
    seq_sz      = 128
    scale_qpsk  = 1.0 / np.sqrt(2)
    scale_16qam = 1.0 / np.sqrt(10)
    noise_std   = 0.1

    X, y = [], []
    for label in [0, 1]:
        for _ in range(n_per_class):
            p    = np.random.uniform(0, 1)
            bpsk = (2 * np.random.randint(0, 2, seq_sz) - 1) + 0j
            qpsk = scale_qpsk * ((2 * np.random.randint(0, 2, seq_sz) - 1)
                                  + 1j * (2 * np.random.randint(0, 2, seq_sz) - 1))
            mask = np.random.uniform(0, 1, seq_sz) <= p
            sig  = mask * bpsk + (1 - mask) * qpsk

            if label == 1:
                mp  = np.array([-3, -1, 1, 3])
                adv = scale_16qam * (mp[np.random.randint(0, 4, seq_sz)]
                                     + 1j * mp[np.random.randint(0, 4, seq_sz)])
                sig = adv + sig

            sig += noise_std * (np.random.randn(seq_sz) + 1j * np.random.randn(seq_sz))
            X.append(np.array([np.real(sig), np.imag(sig)]))
            y.append(label)

    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))


# ── Audio: WAV → mel-spectrogram ──────────────────────────────────────────────

def load_audio_files():
    """Returns (files, labels) — drone WAVs = 1, background = 0."""
    dirs = {
        1: [os.path.join(_audio_dir, "membo_1"),
            os.path.join(_audio_dir, "bebop_1")],
        0: [os.path.join(_audio_dir, "bg noise")],
    }
    files, labels = [], []
    for label, paths in dirs.items():
        for path in paths:
            if not os.path.exists(path):
                print(f"  Warning: {path} not found, skipping")
                continue
            for f in os.listdir(path):
                if f.endswith(".wav"):
                    files.append(os.path.join(path, f))
                    labels.append(label)
    return files, labels


def wav_to_melspec(path: str, sr: int = 22050, duration: float = 1.0) -> torch.Tensor:
    """Returns [1, 128, T] mel-spectrogram tensor."""
    y, _ = librosa.load(path, duration=duration, sr=sr)
    n = int(sr * duration)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    spec    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
    return torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0)


# ── Visual: image loading ─────────────────────────────────────────────────────

def load_visual_files():
    """Returns (files, labels) — drones/ = 1, birds/planes/ = 0."""
    if not os.path.exists(_visual_dir):
        print(f"  Warning: {_visual_dir} not found")
        return [], []
    dataset   = tv_datasets.ImageFolder(root=_visual_dir)
    drone_idx = dataset.class_to_idx.get("drones", dataset.class_to_idx.get("drone", -1))
    files, labels = [], []
    for path, cls_idx in dataset.samples:
        files.append(path)
        labels.append(1 if cls_idx == drone_idx else 0)
    return files, labels


def load_image_tensor(path: str) -> torch.Tensor:
    """Returns [3, 224, 224] normalised image tensor."""
    img = Image.open(path).convert("RGB")
    return visual_transforms(img)


# ── Parquet serialisation helpers ─────────────────────────────────────────────

def tensor_to_list(t: torch.Tensor) -> list:
    """Converts tensor to nested Python list for Parquet storage."""
    return t.numpy().tolist()


def list_to_tensor(l: list, dtype=torch.float32) -> torch.Tensor:
    """Reconstructs tensor from nested list."""
    return torch.tensor(np.array(l), dtype=dtype)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_and_save_dataset(n_per_class: int = 2000,
                           save_path: str = "./ARL/multimodal_dataset.parquet") -> pd.DataFrame:
    """
    Generates n_per_class samples for each class (0=background, 1=drone).

    Each sample is a randomly drawn combination of RF, Audio, and Video
    from their respective pools — not 1-to-1 pairing. This means you can
    generate far more samples than any single modality has files.

    e.g. 2000 RF × 118 audio × 100 images = millions of valid combinations,
    we just sample n_per_class of them.
    """
    print("Generating RF samples...")
    rf_X, rf_y = generate_rf_samples(max(n_per_class * 2, 4000))
    rf_by_label = {0: rf_X[rf_y == 0], 1: rf_X[rf_y == 1]}

    print("Loading audio files...")
    audio_files, audio_labels = load_audio_files()
    audio_by_label = {0: [], 1: []}
    for f, l in zip(audio_files, audio_labels):
        audio_by_label[l].append(f)

    print("Loading visual files...")
    vis_files, vis_labels = load_visual_files()
    vis_by_label = {0: [], 1: []}
    for f, l in zip(vis_files, vis_labels):
        vis_by_label[l].append(f)

    # Report pool sizes
    for lbl, name in [(0, "background"), (1, "drone")]:
        print(f"  {name}: RF={len(rf_by_label[lbl])}  "
              f"audio={len(audio_by_label[lbl])}  "
              f"visual={len(vis_by_label[lbl])}")

    print(f"\nBuilding {n_per_class} samples per class by random combination...")

    rows = []
    for label in [0, 1]:
        rf_pool    = rf_by_label[label]
        audio_pool = audio_by_label[label] if audio_by_label[label] else None
        vis_pool   = vis_by_label[label]   if vis_by_label[label]   else None

        if len(rf_pool) == 0:
            print(f"  Warning: no RF samples for label {label}, skipping")
            continue

        for i in range(n_per_class):
            # Randomly draw from each modality pool independently
            rf    = rf_pool[random.randint(0, len(rf_pool) - 1)]
            audio = wav_to_melspec(random.choice(audio_pool)) if audio_pool \
                    else torch.randn(1, 128, 44)
            video = load_image_tensor(random.choice(vis_pool)) if vis_pool \
                    else torch.randn(3, 224, 224)

            rows.append({
                "rf_input":    tensor_to_list(rf),
                "audio_input": tensor_to_list(audio),
                "video_input": tensor_to_list(video),
                "label":       label,
            })

        print(f"  Class {label}: {n_per_class} samples done")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    df.to_parquet(save_path, index=False)
    print(f"\nSaved {len(df)} samples → {save_path}")
    print(f"File size: {os.path.getsize(save_path) / 1e6:.1f} MB")
    return df


# ── Hugging Face upload ───────────────────────────────────────────────────────

def push_to_huggingface(parquet_path: str,
                        repo_id: str = "James-ARL-2026/Drone-Multi-Modal-Model",
                        token: str = None):
    """
    Pushes the Parquet dataset to Hugging Face Hub.

    Args:
        parquet_path : local path to the .parquet file
        repo_id      : e.g. "your-username/drone-multimodal-dataset"
        token        : HF token (or set HF_TOKEN env var / huggingface-cli login)

    Requires: pip install huggingface_hub datasets
    """
    try:
        from datasets import Dataset as HFDataset
        import huggingface_hub
    except ImportError:
        print("Run: !pip install datasets huggingface_hub")
        return

    print(f"Loading parquet from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    print(f"Pushing to HuggingFace: {repo_id}")
    hf_dataset = HFDataset.from_pandas(df)
    hf_dataset.push_to_hub(repo_id, token=token)
    print(f"Done → https://huggingface.co/datasets/{repo_id}")


# ── PyTorch Dataset (loads from saved Parquet) ────────────────────────────────

class MultiModalDataset(Dataset):
    """
    Loads pre-generated (RF, Audio, Video, label) triplets from a Parquet file.
    Use build_and_save_dataset() to generate the parquet first.
    """

    def __init__(self, parquet_path: str = "./ARL/multimodal_dataset.parquet"):
        df = pd.read_parquet(parquet_path)
        self.rf     = [list_to_tensor(r) for r in df["rf_input"]]
        self.audio  = [list_to_tensor(r) for r in df["audio_input"]]
        self.video  = [list_to_tensor(r) for r in df["video_input"]]
        self.labels = df["label"].tolist()
        print(f"Loaded {len(self.labels)} samples from {parquet_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "rf_input":    self.rf[idx],
            "audio_input": self.audio[idx],
            "video_input": self.video[idx],
            "label":       torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Quick usage example ───────────────────────────────────────────────────────
#
# Step 1 — generate and save:
#   df = build_and_save_dataset(n_per_class=2000, save_path="./ARL/multimodal_dataset.parquet")
#
# Step 2 — push to HuggingFace:
#   push_to_huggingface("./ARL/multimodal_dataset.parquet")
#   # or with explicit repo:
#   push_to_huggingface("./ARL/multimodal_dataset.parquet", "James-ARL-2026/Drone-Multi-Modal-Model")
#
# Step 3 — load for training:
#   dataset = MultiModalDataset("./ARL/multimodal_dataset.parquet")
#   loader  = DataLoader(dataset, batch_size=32, shuffle=True)
