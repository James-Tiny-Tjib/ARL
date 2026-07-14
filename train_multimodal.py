"""
train_multimodal.py
===================
Paste this into a notebook cell AFTER running the multimodal.py cell.
All classes (MultiModalProjectionNetwork, ModalityFusionTransformer,
AdaptiveSupConLoss, EMBEDDING_DIM, NUM_CLASSES, DEVICE, etc.) are
already in the kernel namespace.

Prerequisites (run in Colab before this cell):
    !git clone https://github.com/saraalemadi/DroneAudioDataset.git
    # and ensure drone_demo_dataset/ is unzipped
"""

import os, random
import numpy as np
import torch
import torch.nn as nn
import librosa
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets as tv_datasets
from sklearn.metrics import accuracy_score, classification_report

# ── Paths — inherit from notebook kernel if defined, else fall back ───────────
try:
    _audio_dir  = AUDIO_DATASET_DIR   # defined in notebook config cell
    _visual_dir = VISUAL_DATASET_DIR
except NameError:
    _audio_dir  = "./DroneAudioDataset/Multiclass_Drone_Audio"
    _visual_dir = "./drone_demo_dataset"

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE           = 32
EPOCHS_P1            = 30    # SupCon alignment phase
EPOCHS_P2            = 30    # Fusion transformer phase
LR                   = 1e-3
VAL_SPLIT            = 0.15
TEST_SPLIT           = 0.15
RF_SAMPLES_PER_CLASS = 2000


# ── RF synthetic data generation ──────────────────────────────────────────────

def generate_rf_samples(n_per_class=RF_SAMPLES_PER_CLASS):
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

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


# ── Audio file loading ────────────────────────────────────────────────────────

def load_audio_files():
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


def wav_to_melspec(path, sr=22050, duration=1.0):
    y, _ = librosa.load(path, duration=duration, sr=sr)
    n = int(sr * duration)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    spec    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
    return torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0)  # [1, 128, T]


# ── Visual file loading ───────────────────────────────────────────────────────

visual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


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


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiModalDataset(Dataset):
    """
    Synchronized (RF, Audio, Video) triplets, all sharing the same ground truth label.
    RF is generated synthetically. Audio and Video are loaded from disk.
    Labels are always ground truth — individual model errors are intentional
    and SupCon learns to correct the embedding geometry regardless.
    """

    def __init__(self, n_per_class=RF_SAMPLES_PER_CLASS):
        print("Building dataset...")

        # RF
        rf_X, rf_y = generate_rf_samples(n_per_class)
        self.rf_by_label = {0: rf_X[rf_y == 0], 1: rf_X[rf_y == 1]}

        # Audio
        audio_files, audio_labels = load_audio_files()
        self.audio_by_label = {0: [], 1: []}
        for f, l in zip(audio_files, audio_labels):
            self.audio_by_label[l].append(f)

        # Visual
        vis_files, vis_labels = load_visual_files()
        self.vis_by_label = {0: [], 1: []}
        for f, l in zip(vis_files, vis_labels):
            self.vis_by_label[l].append(f)

        # Balance classes
        n0 = min(len(self.rf_by_label[0]),
                 len(self.audio_by_label[0]) if self.audio_by_label[0] else n_per_class,
                 len(self.vis_by_label[0])   if self.vis_by_label[0]   else n_per_class)
        n1 = min(len(self.rf_by_label[1]),
                 len(self.audio_by_label[1]) if self.audio_by_label[1] else n_per_class,
                 len(self.vis_by_label[1])   if self.vis_by_label[1]   else n_per_class)
        n_each = min(n0, n1, n_per_class)

        if n_each == 0:
            raise RuntimeError(
                "Dataset is empty. Make sure the audio and visual datasets are available.\n"
                "Run: !git clone https://github.com/saraalemadi/DroneAudioDataset.git"
            )

        self.labels = [0] * n_each + [1] * n_each
        print(f"  {n_each} background + {n_each} drone = {len(self.labels)} samples total")

        # Preload visual tensors to avoid repeated disk I/O per batch
        from PIL import Image
        self.vis_tensors_by_label = {}
        for lbl in [0, 1]:
            pool = self.vis_by_label[lbl]
            if pool:
                tensors = []
                for path in random.sample(pool, min(n_each * 2, len(pool))):
                    img = Image.open(path).convert("RGB")
                    tensors.append(visual_transforms(img))
                self.vis_tensors_by_label[lbl] = tensors
            else:
                self.vis_tensors_by_label[lbl] = [torch.randn(3, 224, 224)] * (n_each * 2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        rf    = self.rf_by_label[label][idx % len(self.rf_by_label[label])]
        audio = wav_to_melspec(random.choice(self.audio_by_label[label])) \
                if self.audio_by_label[label] else torch.randn(1, 128, 44)
        video = self.vis_tensors_by_label[label][idx % len(self.vis_tensors_by_label[label])]
        return {
            "rf_input":    rf,
            "audio_input": audio,
            "video_input": video,
            "label":       torch.tensor(label, dtype=torch.long),
        }


# ── Training utilities ────────────────────────────────────────────────────────

def _train_supcon_epoch(proj_net, loader, optimizer, criterion, device):
    proj_net.train()
    total = 0.0
    for batch in loader:
        rf     = batch["rf_input"].to(device)
        audio  = batch["audio_input"].to(device)
        video  = batch["video_input"].to(device)
        labels = batch["label"].to(device)

        token_r, token_a, token_v = proj_net(rf, audio, video)
        features = torch.cat([token_r, token_a, token_v], dim=0)
        lbls     = torch.cat([labels,  labels,  labels],  dim=0)

        loss = criterion(features, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def _train_fusion_epoch(proj_net, fusion_net, loader, optimizer, criterion, device):
    proj_net.eval()
    fusion_net.train()
    total = 0.0
    for batch in loader:
        rf     = batch["rf_input"].to(device)
        audio  = batch["audio_input"].to(device)
        video  = batch["video_input"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            token_r, token_a, token_v = proj_net(rf, audio, video)

        loss = criterion(fusion_net(token_r, token_a, token_v), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _evaluate(proj_net, fusion_net, loader, device):
    proj_net.eval()
    fusion_net.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        rf     = batch["rf_input"].to(device)
        audio  = batch["audio_input"].to(device)
        video  = batch["video_input"].to(device)
        token_r, token_a, token_v = proj_net(rf, audio, video)
        preds  = fusion_net(token_r, token_a, token_v).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].tolist())
    return accuracy_score(all_labels, all_preds), all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────

def train(n_per_class=RF_SAMPLES_PER_CLASS):
    # Data
    dataset = MultiModalDataset(n_per_class=n_per_class)
    n       = len(dataset)
    n_val   = int(n * VAL_SPLIT)
    n_test  = int(n * TEST_SPLIT)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}\n")

    # Models
    proj_net   = MultiModalProjectionNetwork().to(DEVICE)
    fusion_net = ModalityFusionTransformer().to(DEVICE)

    supcon_crit = AdaptiveSupConLoss(temperature=0.07)
    ce_crit     = nn.CrossEntropyLoss()

    proj_opt  = torch.optim.Adam([p for p in proj_net.parameters() if p.requires_grad], lr=LR)
    fuse_opt  = torch.optim.Adam(fusion_net.parameters(), lr=LR)

    # Phase 1 — SupCon alignment
    print("=" * 50)
    print("PHASE 1 — Contrastive Alignment")
    print("=" * 50)
    for epoch in range(1, EPOCHS_P1 + 1):
        loss = _train_supcon_epoch(proj_net, train_loader, proj_opt, supcon_crit, DEVICE)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS_P1}  SupCon loss: {loss:.4f}")

    # Phase 2 — Fusion transformer
    print()
    print("=" * 50)
    print("PHASE 2 — Fusion Transformer")
    print("=" * 50)
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS_P2 + 1):
        loss = _train_fusion_epoch(proj_net, fusion_net, train_loader, fuse_opt, ce_crit, DEVICE)
        if epoch % 5 == 0 or epoch == 1:
            val_acc, _, _ = _evaluate(proj_net, fusion_net, val_loader, DEVICE)
            print(f"  Epoch {epoch:3d}/{EPOCHS_P2}  CE loss: {loss:.4f}  Val acc: {val_acc:.3f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"proj_net": proj_net.state_dict(),
                            "fusion_net": fusion_net.state_dict()},
                           "./ARL/multimodal_best.pt")
                print(f"    ↑ checkpoint saved (val acc: {val_acc:.3f})")

    # Test
    print()
    print("=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    ckpt = torch.load("./ARL/multimodal_best.pt", map_location=DEVICE)
    proj_net.load_state_dict(ckpt["proj_net"])
    fusion_net.load_state_dict(ckpt["fusion_net"])

    test_acc, preds, labels = _evaluate(proj_net, fusion_net, test_loader, DEVICE)
    print(f"Test accuracy: {test_acc:.3f}")
    print(classification_report(labels, preds, target_names=["Background", "Threat"]))

    return proj_net, fusion_net


# Call this after running the multimodal.py cell and ensuring datasets are available
# proj_net, fusion_net = train()
