import os, random
import numpy as np
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset
from torchvision import transforms, datasets as tv_datasets
from PIL import Image
from io import BytesIO

try:
    _audio_dir  = AUDIO_DATASET_DIR
    _visual_dir = VISUAL_DATASET_DIR
except NameError:
    _audio_dir  = "./DroneAudioDataset/Multiclass_Drone_Audio"
    _visual_dir = "./drone_demo_dataset"

visual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── RF ────────────────────────────────────────────────────────────────────────

def generate_rf_samples(n_per_class=2000):
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


# ── Audio ─────────────────────────────────────────────────────────────────────

def _augment_audio(y, sr=22050):
    augs = []
    augs.append(librosa.effects.time_stretch(y, rate=1.1))
    augs.append(librosa.effects.time_stretch(y, rate=0.9))
    augs.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
    augs.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    augs.append(y + np.random.normal(0, 0.005, len(y)))
    return augs


def load_audio_files(augment_background=True):
    drone_dirs = [os.path.join(_audio_dir, "membo_1"),
                  os.path.join(_audio_dir, "bebop_1")]
    bg_dirs    = [os.path.join(_audio_dir, "bg noise"),
                  os.path.join(_audio_dir, "airport"),
                  os.path.join(_audio_dir, "airport_no_human")]
    files, labels = [], []

    for path in drone_dirs:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found")
            continue
        for f in os.listdir(path):
            if f.endswith(".wav"):
                files.append(os.path.join(path, f))
                labels.append(1)

    for path in bg_dirs:
        if not os.path.exists(path):
            continue
        for f in os.listdir(path):
            if f.endswith(".wav"):
                fp = os.path.join(path, f)
                files.append(fp)
                labels.append(0)
                if augment_background:
                    try:
                        y, sr = librosa.load(fp, sr=22050, duration=1.0)
                        for aug in _augment_audio(y, sr):
                            files.append(("array", aug, sr))
                            labels.append(0)
                    except Exception:
                        pass

    print(f"  Audio: {sum(l==1 for l in labels)} drone, {sum(l==0 for l in labels)} background")
    return files, labels


def wav_to_melspec(src, sr=22050, duration=1.0):
    if isinstance(src, tuple) and src[0] == "array":
        y, sr = src[1], src[2]
    else:
        y, sr = librosa.load(src, duration=duration, sr=sr)
    n = int(sr * duration)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    spec    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
    return torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0)


# ── Visual ────────────────────────────────────────────────────────────────────

def load_visual_files():
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


def load_image_tensor(path):
    return visual_transforms(Image.open(path).convert("RGB"))


# ── Serialisation ─────────────────────────────────────────────────────────────

def tensor_to_bytes(t):
    buf = BytesIO()
    torch.save({"data": t.numpy(), "shape": list(t.shape)}, buf)
    return buf.getvalue()


def bytes_to_tensor(b, dtype=torch.float32):
    d = torch.load(BytesIO(b), weights_only=False)
    return torch.tensor(d["data"], dtype=dtype).reshape(d["shape"])


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_and_save_dataset(n_per_class=2000,
                            save_path="./ARL/multimodal_dataset.parquet"):
    print("Generating RF samples...")
    rf_X, rf_y = generate_rf_samples(max(n_per_class * 2, 4000))
    rf_by_label = {0: rf_X[rf_y == 0], 1: rf_X[rf_y == 1]}

    print("Loading audio...")
    audio_files, audio_labels = load_audio_files()
    audio_by_label = {0: [], 1: []}
    for f, l in zip(audio_files, audio_labels):
        audio_by_label[l].append(f)

    print("Loading visuals...")
    vis_files, vis_labels = load_visual_files()
    vis_by_label = {0: [], 1: []}
    for f, l in zip(vis_files, vis_labels):
        vis_by_label[l].append(f)

    for lbl, name in [(0, "background"), (1, "drone")]:
        print(f"  {name}: RF={len(rf_by_label[lbl])}  "
              f"audio={len(audio_by_label[lbl])}  "
              f"visual={len(vis_by_label[lbl])}")

    rows = []
    for label in [0, 1]:
        rf_pool    = rf_by_label[label]
        audio_pool = audio_by_label[label] or None
        vis_pool   = vis_by_label[label]   or None

        for _ in range(n_per_class):
            rf    = rf_pool[random.randint(0, len(rf_pool) - 1)]
            audio = wav_to_melspec(random.choice(audio_pool)) if audio_pool \
                    else torch.randn(1, 128, 44)
            video = load_image_tensor(random.choice(vis_pool)) if vis_pool \
                    else torch.randn(3, 224, 224)
            rows.append({
                "rf_input":    tensor_to_bytes(rf),
                "audio_input": tensor_to_bytes(audio),
                "video_input": tensor_to_bytes(video),
                "label":       label,
            })
        print(f"  Class {label}: {n_per_class} samples done")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_parquet(save_path, index=False)
    print(f"Saved {len(df)} samples → {save_path}  ({os.path.getsize(save_path)/1e6:.1f} MB)")
    return df


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class MultiModalDataset(Dataset):
    def __init__(self, parquet_path="./ARL/multimodal_dataset.parquet"):
        df = pd.read_parquet(parquet_path)
        self.rf     = [bytes_to_tensor(r) for r in df["rf_input"]]
        self.audio  = [bytes_to_tensor(r) for r in df["audio_input"]]
        self.video  = [bytes_to_tensor(r) for r in df["video_input"]]
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
