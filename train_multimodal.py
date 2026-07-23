"""
train_multimodal.py
===================
Paste into a notebook cell AFTER running multimodal.py and data_loading.py cells.

Reads pre-generated data from a Parquet file (created by data_loading.py).
All model classes are already in the kernel namespace from multimodal.py.

Workflow:
    1. Run multimodal.py cell        — defines model classes
    2. Run data_loading.py cell      — defines dataset utilities
    3. Generate data once:
         df = build_and_save_dataset(n_per_class=2000)
    4. Run this cell, then call:
         proj_net, fusion_net = train()
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report

# ── Hyperparameters ───────────────────────────────────────────────────────────
PARQUET_PATH = "./ARL/multimodal_dataset.parquet"
BATCH_SIZE   = 32
EPOCHS_P1    = 30     # SupCon alignment
EPOCHS_P2    = 30     # Fusion transformer
LR           = 1e-3
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15


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


# ── Main training loop ────────────────────────────────────────────────────────

def train(parquet_path=PARQUET_PATH):
    # Load pre-generated dataset from parquet
    dataset = MultiModalDataset(parquet_path=parquet_path)
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

    proj_opt = torch.optim.Adam(
        [p for p in proj_net.parameters() if p.requires_grad], lr=LR
    )
    fuse_opt = torch.optim.Adam(fusion_net.parameters(), lr=LR)

    # Phase 1 — SupCon alignment (trains projectors only)
    print("=" * 50)
    print("PHASE 1 — Contrastive Alignment")
    print("=" * 50)
    for epoch in range(1, EPOCHS_P1 + 1):
        loss = _train_supcon_epoch(proj_net, train_loader, proj_opt, supcon_crit, DEVICE)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS_P1}  SupCon loss: {loss:.4f}")

    # Phase 2 — Fusion transformer (projectors frozen)
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
                torch.save({"proj_net":   proj_net.state_dict(),
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


# proj_net, fusion_net = train()
