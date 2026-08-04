"""
api.py — FastAPI wrapper for the RF Threat Classification System

Endpoints:
  POST /detect-all          — run full multimodal detection on a single snapshot
  POST /detect/rf       — RF-only inference (raw IQ tensor as JSON)
  POST /detect/audio    — audio-only inference (WAV file upload)
  POST /detect/visual   — visual-only inference (image file upload)
  GET  /snapshot/latest — return the most recent master snapshot
  GET  /snapshot/history — return snapshot log rows (paginated)
  POST /chat            — LLM assistant query against latest snapshot
  GET  /health          — liveness check

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import os
import csv
import time
import copy
import random
import pickle
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import pandas as pd
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR           = Path("./ARL")
RF_MODEL_PATH      = BASE_DIR / "sensor_client.pkl"
AUDIO_MODEL_PATH   = BASE_DIR / "drone_multi_classifier.pt"
VISUAL_MODEL_PATH  = BASE_DIR / "resnet50_drone_weights.pth"
SUPCON_MODEL_PATH  = BASE_DIR / "multi_modal_model" / "supcon_model.pt"
FUSION_MODEL_PATH  = BASE_DIR / "multi_modal_model" / "fusion_model.pt"
XGBOOST_MODEL_PATH = BASE_DIR / "xgboostmodel.pkl"
SNAPSHOT_CSV_PATH  = BASE_DIR / "snapshot_log.csv"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Model architectures (copied verbatim from notebook)
# ---------------------------------------------------------------------------
import torch.nn as nn
from torchvision import models as tv_models


class DroneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 32 * 11, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


class IQCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.layer_dims: list = []
        self.layers = nn.ModuleList()
        for in_ch, out_ch in [(2, 8), (8, 16), (16, 32), (32, 64)]:
            self.layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False))
        self.conv_num = len(self.layers)
        self.layers.append(nn.Linear(64, 256, bias=False))
        self.layers.append(nn.Linear(256, num_classes, bias=False))
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for i in range(self.conv_num):
            x = F.relu(self.layers[i](x))
        x = self.global_avg_pool(x).squeeze(-1)
        for i in range(self.conv_num, len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)

EMBEDDING_DIM = 256
NUM_CLASSES   = 2


class FrozenRFEncoder(nn.Module):
    def __init__(self, backbone: IQCNN):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.projector = nn.Sequential(
            nn.Linear(256, EMBEDDING_DIM), nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(inplace=True), nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        )

    def forward(self, x):
        with torch.no_grad():
            for i in range(self.backbone.conv_num):
                x = F.relu(self.backbone.layers[i](x))
            x = self.backbone.global_avg_pool(x).squeeze(-1)
            for i in range(self.backbone.conv_num, len(self.backbone.layers) - 1):
                x = F.relu(self.backbone.layers[i](x))
        return self.projector(x)  # [B, 256]


class FrozenAudioEncoder(nn.Module):
    def __init__(self, backbone: DroneCNN):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.projector = nn.Sequential(
            nn.Linear(128, EMBEDDING_DIM), nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(inplace=True), nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone.conv_layers(x)
            x = self.backbone.fc_layers[0](x)
            x = self.backbone.fc_layers[1](x)
            x = self.backbone.fc_layers[2](x)
        return self.projector(x)  # [B, 256]


class FrozenVideoEncoder(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        for p in backbone.parameters():
            p.requires_grad = False
        self.feature_extractor = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,
        )
        self.projector = nn.Sequential(
            nn.Linear(2048, EMBEDDING_DIM), nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(inplace=True), nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        )

    def forward(self, x):
        with torch.no_grad():
            hidden = self.feature_extractor(x).flatten(1)
        return self.projector(hidden)  # [B, 256]

class ModalityFusionTransformer(nn.Module):
    def __init__(self, embed_dim=EMBEDDING_DIM, num_heads=4, num_layers=2,
                 num_classes=NUM_CLASSES, dropout=0.1):
        super().__init__()
        self.modality_tags = nn.Parameter(torch.randn(3, embed_dim) * 0.02)
        self.cls_token     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                  enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, token_r, token_a, token_v):
        B = token_r.shape[0]
        x_r = token_r + self.modality_tags[0]
        x_a = token_a + self.modality_tags[1]
        x_v = token_v + self.modality_tags[2]
        seq = torch.cat([self.cls_token.expand(B, -1, -1),
                         torch.stack([x_r, x_a, x_v], dim=1)], dim=1)
        cls_out = self.transformer(seq)[:, 0, :]
        return self.classifier(cls_out)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Holds all loaded models. Loaded once at startup."""
    rf_model:     IQCNN | None = None
    audio_model:  DroneCNN | None = None
    visual_model: nn.Module | None = None
    rf_encoder:   FrozenRFEncoder | None = None
    audio_encoder: FrozenAudioEncoder | None = None
    video_encoder: FrozenVideoEncoder | None = None
    fusion_transformer: ModalityFusionTransformer | None = None
    xgb_model:    Any = None  # sklearn-compatible XGBoost


registry = ModelRegistry()


def _load_all_models() -> None:
    log.info("Loading models…")

    # RF — stored as a pickle (sensor_client.pkl) with a custom CPU unpickler
    if RF_MODEL_PATH.exists():
        class _CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
                return super().find_class(module, name)
        with open(RF_MODEL_PATH, "rb") as f:
            raw = _CPUUnpickler(f).load()
        # raw may be an OrderedDict state_dict or a Client-wrapped object
        state_dict = raw if isinstance(raw, dict) else raw.model.state_dict()
        rf = IQCNN(num_classes=2)
        rf.load_state_dict(state_dict)
        registry.rf_model = rf.to(DEVICE).eval()
        registry.rf_encoder = FrozenRFEncoder(registry.rf_model).to(DEVICE).eval()
        log.info("RF model loaded")
    else:
        log.warning(f"RF weights not found at {RF_MODEL_PATH}")

    # Audio
    if AUDIO_MODEL_PATH.exists():
        au = DroneCNN()
        au.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location="cpu", weights_only=False))
        registry.audio_model = au.to(DEVICE).eval()
        registry.audio_encoder = FrozenAudioEncoder(registry.audio_model).to(DEVICE).eval()
        log.info("Audio model loaded")
    else:
        log.warning(f"Audio weights not found at {AUDIO_MODEL_PATH}")

    # Visual
    if VISUAL_MODEL_PATH.exists():
        vs = tv_models.resnet50(weights=None)
        vs.fc = nn.Linear(vs.fc.in_features, 1)
        ckpt = torch.load(VISUAL_MODEL_PATH, map_location="cpu", weights_only=False)
        vs.load_state_dict(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
        registry.visual_model = vs.to(DEVICE).eval()
        registry.video_encoder = FrozenVideoEncoder(registry.visual_model).to(DEVICE).eval()
        log.info("Visual model loaded")
    else:
        log.warning(f"Visual weights not found at {VISUAL_MODEL_PATH}")

    # Fusion transformer (SupCon-aligned projection + cross-modal attention)
    if SUPCON_MODEL_PATH.exists() and FUSION_MODEL_PATH.exists():
        supcon_ckpt = torch.load(SUPCON_MODEL_PATH, map_location="cpu", weights_only=False)
        fusion_ckpt = torch.load(FUSION_MODEL_PATH, map_location="cpu", weights_only=False)
        # Reload fresh encoders from the trained supcon checkpoint
        if registry.rf_encoder:
            registry.rf_encoder.projector.load_state_dict(
                {k.replace("rf_encoder.projector.", ""): v
                 for k, v in supcon_ckpt.items() if k.startswith("rf_encoder.projector.")},
                strict=False,
            )
        if registry.audio_encoder:
            registry.audio_encoder.projector.load_state_dict(
                {k.replace("audio_encoder.projector.", ""): v
                 for k, v in supcon_ckpt.items() if k.startswith("audio_encoder.projector.")},
                strict=False,
            )
        if registry.video_encoder:
            registry.video_encoder.projector.load_state_dict(
                {k.replace("video_encoder.projector.", ""): v
                 for k, v in supcon_ckpt.items() if k.startswith("video_encoder.projector.")},
                strict=False,
            )
        ft = ModalityFusionTransformer()
        ft.load_state_dict(fusion_ckpt)
        registry.fusion_transformer = ft.to(DEVICE).eval()
        log.info("Fusion transformer loaded")
    else:
        log.warning("Fusion transformer checkpoints not found — /detect-all will use XGBoost fallback")

    # XGBoost fusion fallback
    if XGBOOST_MODEL_PATH.exists():
        with open(XGBOOST_MODEL_PATH, "rb") as f:
            registry.xgb_model = pickle.load(f)
        log.info("XGBoost model loaded")
    else:
        log.warning(f"XGBoost model not found at {XGBOOST_MODEL_PATH}")

    log.info("All models loaded.")


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

VISUAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_audio_bytes(audio_bytes: bytes, sr: int = 22050, duration: float = 1.0) -> torch.Tensor:
    """Load WAV bytes → normalised mel-spectrogram tensor [1, 1, 128, T]."""
    buf = io.BytesIO(audio_bytes)
    y, file_sr = librosa.load(buf, sr=sr, duration=duration)
    n = int(sr * duration)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    spec    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
    return torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,128,T]


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Load image bytes → normalised tensor [1, 3, 224, 224]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return VISUAL_TRANSFORMS(img).unsqueeze(0)  # [1, 3, 224, 224]


def generate_rf_tensor(noise_std: float = 0.1, p: float = 0.5,
                       is_threat: bool = False, dist: int = 0) -> torch.Tensor:
    """Synthesise a single IQ sample matching the notebook's generator."""
    seq_sz = 128
    scale_qpsk  = 1.0 / np.sqrt(2)
    scale_16qam = 1.0 / np.sqrt(10)
    bpsk = (2 * np.random.randint(0, 2, seq_sz) - 1) + 0j
    qpsk = scale_qpsk * ((2 * np.random.randint(0, 2, seq_sz) - 1)
                          + 1j * (2 * np.random.randint(0, 2, seq_sz) - 1))
    mask = np.random.uniform(0, 1, seq_sz) <= p
    sig  = mask * bpsk + (1 - mask) * qpsk
    if is_threat:
        mp  = np.array([-3, -1, 1, 3])
        adv = scale_16qam * (mp[np.random.randint(0, 4, seq_sz)]
                             + 1j * mp[np.random.randint(0, 4, seq_sz)])
        sig = (1.0 / max(dist, 1)) * adv + sig
    sig += noise_std * (np.random.randn(seq_sz) + 1j * np.random.randn(seq_sz))
    arr = np.array([np.real(sig), np.imag(sig)], dtype=np.float32)
    return torch.tensor(arr).unsqueeze(0)  # [1, 2, 128]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def infer_rf(rf_tensor: torch.Tensor) -> dict:
    """Run RF model. Returns threat_prob, friendly_prob, is_threat."""
    if registry.rf_model is None:
        raise HTTPException(503, "RF model not loaded")
    with torch.inference_mode():
        logits = registry.rf_model(rf_tensor.to(DEVICE))
        probs  = torch.softmax(logits, dim=1).cpu().squeeze(0)
    return {
        "friendly_prob": float(probs[0]),
        "threat_prob":   float(probs[1]),
        "is_threat":     bool(probs[1] >= 0.5),
    }


def infer_audio(audio_tensor: torch.Tensor) -> dict:
    """Run audio model. Returns mambo/bebop/background probs and is_threat."""
    if registry.audio_model is None:
        raise HTTPException(503, "Audio model not loaded")
    with torch.inference_mode():
        logits = registry.audio_model(audio_tensor.to(DEVICE))
        probs  = torch.softmax(logits, dim=1).cpu().squeeze(0)
    mambo, bebop, background = float(probs[0]), float(probs[1]), float(probs[2])
    return {
        "mambo_prob":      mambo,
        "bebop_prob":      bebop,
        "background_prob": background,
        "is_threat":       bool(mambo + bebop >= 0.5),
    }


def infer_visual(image_tensor: torch.Tensor) -> dict:
    """Run visual model. Returns drone_prob and is_threat."""
    if registry.visual_model is None:
        raise HTTPException(503, "Visual model not loaded")
    with torch.inference_mode():
        logit = registry.visual_model(image_tensor.to(DEVICE))
        prob  = torch.sigmoid(logit).cpu().squeeze().item()
    return {
        "drone_prob": float(prob),
        "is_threat":  bool(prob >= 0.5),
    }


def infer_fusion_transformer(rf_tensor: torch.Tensor,
                             audio_tensor: torch.Tensor,
                             image_tensor: torch.Tensor) -> dict:
    """Run full multimodal fusion pipeline. Returns threat/clear probs."""
    enc_r = registry.rf_encoder
    enc_a = registry.audio_encoder
    enc_v = registry.video_encoder
    fuse  = registry.fusion_transformer
    if None in (enc_r, enc_a, enc_v, fuse):
        raise HTTPException(503, "Fusion transformer not fully loaded")
    with torch.inference_mode():
        token_r = enc_r(rf_tensor.to(DEVICE))
        token_a = enc_a(audio_tensor.to(DEVICE))
        token_v = enc_v(image_tensor.to(DEVICE))
        logits  = fuse(token_r, token_a, token_v)
        probs   = torch.softmax(logits, dim=1).cpu().squeeze(0)
    return {
        "clear_prob":  float(probs[0]),
        "threat_prob": float(probs[1]),
        "is_threat":   bool(probs[1] >= 0.5),
    }


def infer_xgboost(rf_result: dict, audio_result: dict, visual_result: dict,
                  sensor_features: list[float] | None = None) -> dict:
    """Fallback: run XGBoost fusion model on aggregated confidence features."""
    if registry.xgb_model is None:
        raise HTTPException(503, "XGBoost model not loaded")
    features = [
        rf_result.get("threat_prob", 0.0) * 100,
        audio_result.get("mambo_prob", 0.0) * 100,
        audio_result.get("bebop_prob", 0.0) * 100,
        audio_result.get("background_prob", 0.0) * 100,
        visual_result.get("drone_prob", 0.0) * 100,
    ]
    if sensor_features:
        features.extend(sensor_features)
    else:
        features.extend([0.0] * 46)  # 23 sensors × 2 columns

    NUM_SENSORS = 23
    sensor_cols = []
    for i in range(NUM_SENSORS):
        sensor_cols += [f"sensor_{i}_threat_conf", f"sensor_{i}_friendly_conf"]
    col_names = ["rf_confidence", "audio_mambo", "audio_bebop",
                 "audio_background", "visual_confidence"] + sensor_cols
    X = pd.DataFrame([features], columns=col_names)
    probs   = registry.xgb_model.predict_proba(X)[0]
    return {
        "clear_prob":  float(probs[0]),
        "threat_prob": float(probs[1]),
        "is_threat":   bool(probs[1] >= 0.5),
    }


# ---------------------------------------------------------------------------
# Snapshot store (in-memory ring buffer + CSV write)
# ---------------------------------------------------------------------------

_snapshot_history: list[dict] = []
_MAX_HISTORY = 500


def _record_snapshot(snap: dict) -> None:
    _snapshot_history.append(snap)
    if len(_snapshot_history) > _MAX_HISTORY:
        _snapshot_history.pop(0)
    # Append to CSV
    _write_snapshot_csv(snap)


def _write_snapshot_csv(snap: dict) -> None:
    row = {
        "timestamp":        snap.get("timestamp"),
        "is_threat_gt":     snap.get("is_threat_gt", ""),
        "rf_confidence":    snap.get("rf", {}).get("threat_prob", 0.0),
        "rf_is_threat":     snap.get("rf", {}).get("is_threat", False),
        "audio_mambo":      snap.get("audio", {}).get("mambo_prob", 0.0),
        "audio_bebop":      snap.get("audio", {}).get("bebop_prob", 0.0),
        "audio_background": snap.get("audio", {}).get("background_prob", 0.0),
        "audio_is_threat":  snap.get("audio", {}).get("is_threat", False),
        "visual_confidence": snap.get("visual", {}).get("drone_prob", 0.0),
        "visual_is_threat": snap.get("visual", {}).get("is_threat", False),
        "fusion_threat_prob": snap.get("fusion", {}).get("threat_prob", 0.0),
        "fusion_is_threat": snap.get("fusion", {}).get("is_threat", False),
    }
    file_exists = SNAPSHOT_CSV_PATH.exists()
    with open(SNAPSHOT_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RFInput(BaseModel):
    """Raw IQ tensor as nested list: shape [2, 128] (real/imag channels)."""
    iq_data:   list[list[float]] = Field(..., description="Shape [2, 128]")
    noise_std: float = Field(0.1, description="Noise std (used for synthetic generation fallback)")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User question about the current sensor state")


class DetectionResponse(BaseModel):
    timestamp:  float
    rf:         dict
    audio:      dict
    visual:     dict
    fusion:     dict
    is_threat:  bool


class HealthResponse(BaseModel):
    status:         str
    device:         str
    models_loaded:  dict[str, bool]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RF Threat Classification API",
    description="Multimodal drone detection: RF + Audio + Visual fusion",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    _load_all_models()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/ping", tags=["System"])
def ping() -> dict:
    """Dead-simple connectivity check — no models needed."""
    return {"message": "pong", "timestamp": time.time()}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        models_loaded={
            "rf":               registry.rf_model is not None,
            "audio":            registry.audio_model is not None,
            "visual":           registry.visual_model is not None,
            "fusion_transformer": registry.fusion_transformer is not None,
            "xgboost":          registry.xgb_model is not None,
        },
    )


# ---------------------------------------------------------------------------
# Modality-specific endpoints
# ---------------------------------------------------------------------------

@app.post("/detect/rf", tags=["Detection"])
def detect_rf(body: RFInput) -> dict:
    """
    RF-only inference. Pass a [2, 128] IQ tensor or omit iq_data to use
    a synthetic sample.
    """
    if body.iq_data:
        arr = np.array(body.iq_data, dtype=np.float32)
        if arr.shape != (2, 128):
            raise HTTPException(422, f"iq_data must be shape [2, 128], got {list(arr.shape)}")
        rf_tensor = torch.tensor(arr).unsqueeze(0)
    else:
        rf_tensor = generate_rf_tensor(noise_std=body.noise_std)
    return {"timestamp": time.time(), **infer_rf(rf_tensor)}


@app.post("/detect/audio", tags=["Detection"])
async def detect_audio(file: UploadFile = File(..., description="WAV file")) -> dict:
    """Audio-only inference. Upload a .wav file."""
    audio_bytes  = await file.read()
    audio_tensor = preprocess_audio_bytes(audio_bytes)
    return {"timestamp": time.time(), **infer_audio(audio_tensor)}


@app.post("/detect/visual", tags=["Detection"])
async def detect_visual(file: UploadFile = File(..., description="Image file (JPEG/PNG)")) -> dict:
    """Visual-only inference. Upload an image file."""
    image_bytes  = await file.read()
    image_tensor = preprocess_image_bytes(image_bytes)
    return {"timestamp": time.time(), **infer_visual(image_tensor)}


# ---------------------------------------------------------------------------
# Full multimodal detection
# ---------------------------------------------------------------------------

@app.post("/detect-all", response_model=DetectionResponse, tags=["Detection"])
async def detect(
    audio_file:  UploadFile = File(None, description="WAV file (optional)"),
    visual_file: UploadFile = File(None, description="Image file (optional)"),
    iq_data:     str        = Query(None, description="JSON-encoded [2,128] IQ array (optional)"),
) -> DetectionResponse:
    """
    Full multimodal detection endpoint.

    - Provide audio_file, visual_file, and/or iq_data to use real sensor data.
    - Any omitted modality falls back to a synthetic/random sample so the
      fusion model always receives all three inputs.
    - Uses the fusion transformer when available, otherwise falls back to XGBoost.
    """
    import json

    # RF
    if iq_data:
        arr = np.array(json.loads(iq_data), dtype=np.float32)
        if arr.shape != (2, 128):
            raise HTTPException(422, f"iq_data must be [2,128], got {list(arr.shape)}")
        rf_tensor = torch.tensor(arr).unsqueeze(0)
    else:
        rf_tensor = generate_rf_tensor()

    # Audio
    if audio_file:
        audio_bytes  = await audio_file.read()
        audio_tensor = preprocess_audio_bytes(audio_bytes)
    else:
        audio_tensor = torch.randn(1, 1, 128, 44)  # synthetic fallback

    # Visual
    if visual_file:
        image_bytes  = await visual_file.read()
        image_tensor = preprocess_image_bytes(image_bytes)
    else:
        image_tensor = torch.randn(1, 3, 224, 224)  # synthetic fallback

    # Per-modality results
    rf_result     = infer_rf(rf_tensor)
    audio_result  = infer_audio(audio_tensor)
    visual_result = infer_visual(image_tensor)

    # Fusion
    if registry.fusion_transformer is not None:
        fusion_result = infer_fusion_transformer(rf_tensor, audio_tensor, image_tensor)
    elif registry.xgb_model is not None:
        fusion_result = infer_xgboost(rf_result, audio_result, visual_result)
    else:
        # Majority vote fallback
        votes = sum([rf_result["is_threat"], audio_result["is_threat"], visual_result["is_threat"]])
        fusion_result = {"is_threat": votes >= 2, "threat_prob": votes / 3.0, "clear_prob": 1 - votes / 3.0}

    snap = {
        "timestamp": time.time(),
        "rf":        rf_result,
        "audio":     audio_result,
        "visual":    visual_result,
        "fusion":    fusion_result,
        "is_threat": fusion_result["is_threat"],
    }
    _record_snapshot(snap)

    return DetectionResponse(**snap)


# ---------------------------------------------------------------------------
# Snapshot history
# ---------------------------------------------------------------------------

@app.get("/snapshot/latest", tags=["Snapshots"])
def snapshot_latest() -> dict:
    """Return the most recent detection snapshot."""
    if not _snapshot_history:
        raise HTTPException(404, "No snapshots recorded yet")
    return _snapshot_history[-1]


@app.get("/snapshot/history", tags=["Snapshots"])
def snapshot_history(
    limit:  int = Query(50,  ge=1, le=500, description="Max rows to return"),
    offset: int = Query(0,   ge=0,         description="Skip N most-recent rows"),
) -> dict:
    """Return paginated snapshot history (newest first)."""
    total   = len(_snapshot_history)
    sliced  = list(reversed(_snapshot_history))[offset: offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "snapshots": sliced}


# ---------------------------------------------------------------------------
# LLM chat
# ---------------------------------------------------------------------------

@app.post("/chat", tags=["Assistant"])
async def chat(body: ChatRequest) -> dict:
    """
    Ask the LLM assistant a question about the current sensor state.
    Requires Ollama running locally with the llama3.2:1b-drone-queries model.
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
        from tabulate import tabulate
    except ImportError as e:
        raise HTTPException(500, f"LLM dependencies not installed: {e}")

    snap = _snapshot_history[-1] if _snapshot_history else {}

    INSTRUCTION = (
        "You are the AI assistant for an airport counter-drone threat-detection system. "
        "Answer questions about the current sensor readings clearly and concisely. "
        "Do not invent numbers not present in the snapshot."
    )

    snap_text = (
        f"RF:     threat_prob={snap.get('rf', {}).get('threat_prob', 'N/A'):.3f}, "
        f"is_threat={snap.get('rf', {}).get('is_threat', 'N/A')}\n"
        f"Audio:  mambo={snap.get('audio', {}).get('mambo_prob', 'N/A'):.3f}, "
        f"bebop={snap.get('audio', {}).get('bebop_prob', 'N/A'):.3f}, "
        f"is_threat={snap.get('audio', {}).get('is_threat', 'N/A')}\n"
        f"Visual: drone_prob={snap.get('visual', {}).get('drone_prob', 'N/A'):.3f}, "
        f"is_threat={snap.get('visual', {}).get('is_threat', 'N/A')}\n"
        f"Fusion: threat_prob={snap.get('fusion', {}).get('threat_prob', 'N/A'):.3f}, "
        f"verdict={'THREAT' if snap.get('is_threat') else 'CLEAR'}"
    ) if snap else "No snapshot data available yet."

    llm = ChatOllama(model="llama3.2:1b-drone-queries", temperature=0)
    messages = [
        SystemMessage(content=INSTRUCTION),
        HumanMessage(content=f"USER QUESTION:\n{body.message}\n\nCURRENT READINGS:\n{snap_text}"),
    ]
    response = await llm.ainvoke(messages)
    return {"reply": response.content, "snapshot_timestamp": snap.get("timestamp")}

