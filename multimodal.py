import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_DIM = 256
NUM_CLASSES   = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RF_WEIGHTS_PATH    = "ARL/best_cnn.pt"
AUDIO_WEIGHTS_PATH = "ARL/drone_multi_classifier.pt"
VIDEO_WEIGHTS_PATH = "ARL/resnet50_drone_weights.pth"


# ── Existing model architectures (must match originals for state_dict loading) ─

class IQCNN(nn.Module):
    # RF IQ classifier. Input: [B, 2, 128]. Intercept at layer index 4 (256-D).
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.layer_dims = []
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(2,  8,  kernel_size=7, padding=3, bias=False))
        self.layers.append(nn.Conv1d(8,  16, kernel_size=7, padding=3, bias=False))
        self.layers.append(nn.Conv1d(16, 32, kernel_size=7, padding=3, bias=False))
        self.layers.append(nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False))
        self.conv_num = len(self.layers)  # 4
        self.layers.append(nn.Linear(64,  256, bias=False))          # index 4 ← intercept
        self.layers.append(nn.Linear(256, num_classes, bias=False))  # index 5 ← head
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for i in range(self.conv_num):
            x = F.relu(self.layers[i](x))
        x = self.global_avg_pool(x).squeeze(-1)
        for i in range(self.conv_num, len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)


class DroneCNN(nn.Module):
    # Audio mel-spec classifier. Input: [B, 1, 128, 44]. Intercept after fc[0..2] (128-D).
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # fc[0]=Linear(22528→128), fc[1]=ReLU, fc[2]=Dropout, fc[3]=Linear(128→3)
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 32 * 11, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


def _build_resnet50(num_classes: int = 1) -> nn.Module:
    # ResNet50 with custom head. Intercept at avgpool (2048-D).
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── Weight loading helpers ────────────────────────────────────────────────────

def _load_state_dict(path: str, map_location="cpu") -> dict:
    raw = torch.load(path, map_location=map_location)
    if isinstance(raw, nn.Module):
        return raw.state_dict()
    if isinstance(raw, dict):
        return raw.get("model", raw.get("state_dict", raw))
    raise ValueError(f"Unrecognised checkpoint format: {type(raw)}")

def load_iqcnn(path: str = RF_WEIGHTS_PATH, num_classes: int = 2) -> IQCNN:
    m = IQCNN(num_classes=num_classes)
    m.load_state_dict(_load_state_dict(path))
    return m.eval()

def load_drone_cnn(path: str = AUDIO_WEIGHTS_PATH) -> DroneCNN:
    m = DroneCNN()
    m.load_state_dict(_load_state_dict(path))
    return m.eval()

def load_resnet50(path: str = VIDEO_WEIGHTS_PATH, num_classes: int = 1) -> nn.Module:
    m = _build_resnet50(num_classes=num_classes)
    m.load_state_dict(_load_state_dict(path))
    return m.eval()


# ── Frozen encoder wrappers ───────────────────────────────────────────────────

class FrozenRFEncoder(nn.Module):
    # Loads IQCNN, freezes it, projects 256-D hidden state → EMBEDDING_DIM.
    def __init__(self, weights_path: str = RF_WEIGHTS_PATH):
        super().__init__()
        self.backbone = load_iqcnn(weights_path)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.projector = nn.Sequential(
            nn.Linear(256, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        )

    def forward(self, x):
        # x: [B, 2, 128]
        with torch.no_grad():
            for i in range(self.backbone.conv_num):
                x = F.relu(self.backbone.layers[i](x))
            x = self.backbone.global_avg_pool(x).squeeze(-1)  # [B, 64]
            for i in range(self.backbone.conv_num, len(self.backbone.layers) - 1):
                x = F.relu(self.backbone.layers[i](x))        # [B, 256]
        return self.projector(x).unsqueeze(1)                 # [B, 1, EMBEDDING_DIM]


class FrozenAudioEncoder(nn.Module):
    # Loads DroneCNN, freezes it, projects 128-D hidden state → EMBEDDING_DIM.
    def __init__(self, weights_path: str = AUDIO_WEIGHTS_PATH):
        super().__init__()
        self.backbone = load_drone_cnn(weights_path)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.projector = nn.Sequential(
            nn.Linear(128, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        )

    def forward(self, x):
        # x: [B, 1, 128, 44]
        with torch.no_grad():
            x = self.backbone.conv_layers(x)       # [B, 22528]
            x = self.backbone.fc_layers[0](x)      # Linear → [B, 128]
            x = self.backbone.fc_layers[1](x)      # ReLU
            x = self.backbone.fc_layers[2](x)      # Dropout (no-op in eval)
        return self.projector(x).unsqueeze(1)      # [B, 1, EMBEDDING_DIM]


class FrozenVideoEncoder(nn.Module):
    # Loads ResNet50, freezes it, projects 2048-D avgpool output → EMBEDDING_DIM.
    def __init__(self, weights_path: str = VIDEO_WEIGHTS_PATH):
        super().__init__()
        self.backbone = load_resnet50(weights_path)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,   self.backbone.bn1,
            self.backbone.relu,    self.backbone.maxpool,
            self.backbone.layer1,  self.backbone.layer2,
            self.backbone.layer3,  self.backbone.layer4,
            self.backbone.avgpool,                          # → [B, 2048, 1, 1]
        )
        self.projector = nn.Sequential(
            nn.Linear(2048, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        )

    def forward(self, x):
        # x: [B, 3, 224, 224]
        with torch.no_grad():
            hidden = self.feature_extractor(x).flatten(1)  # [B, 2048]
        return self.projector(hidden).unsqueeze(1)         # [B, 1, EMBEDDING_DIM]


# ── Multi-modal projection network ────────────────────────────────────────────

class MultiModalProjectionNetwork(nn.Module):
    """
    Owns all three frozen encoders + trainable projectors.
    Returns flat [B, EMBEDDING_DIM] tokens for each sensor.
    Tokens can be stacked for SupCon or cat'd into [B, 3, EMBEDDING_DIM] for fusion.
    """
    def __init__(
        self,
        rf_weights_path:    str = RF_WEIGHTS_PATH,
        audio_weights_path: str = AUDIO_WEIGHTS_PATH,
        video_weights_path: str = VIDEO_WEIGHTS_PATH,
    ):
        super().__init__()
        self.rf_encoder    = FrozenRFEncoder(rf_weights_path)
        self.audio_encoder = FrozenAudioEncoder(audio_weights_path)
        self.video_encoder = FrozenVideoEncoder(video_weights_path)

    def forward(self, rf_input, audio_input, video_input):
        # Returns token_r, token_a, token_v — each [B, EMBEDDING_DIM]
        token_r = self.rf_encoder(rf_input).squeeze(1)
        token_a = self.audio_encoder(audio_input).squeeze(1)
        token_v = self.video_encoder(video_input).squeeze(1)
        return token_r, token_a, token_v


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

class AdaptiveSupConLoss(nn.Module):
    """
    SupCon loss (Khosla et al. 2020).
    L2-normalises features, builds an all-to-all cosine similarity matrix
    scaled by temperature, then minimises the negative log-probability of
    positive pairs (same label) against all non-self pairs.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [N, D]  labels: [N]
        device = features.device
        N = features.shape[0]

        z = F.normalize(features, p=2, dim=1)                          # unit hypersphere
        sim = torch.matmul(z, z.T) / self.temperature                  # [N, N]

        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, float("-inf"))                 # zero out diagonal

        labels   = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask  # [N, N]

        num_pos  = pos_mask.sum(dim=1).float()
        valid    = num_pos > 0

        log_prob = sim - torch.logsumexp(
            sim.masked_fill(self_mask, float("-inf")), dim=1, keepdim=True
        )
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (num_pos + 1e-8)

        return -mean_log_prob_pos[valid].mean()


# ── Contrastive training step ─────────────────────────────────────────────────

def train_contrastive_step(
    projection_net: MultiModalProjectionNetwork,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    criterion: AdaptiveSupConLoss,
    device: torch.device,
) -> float:
    """
    One alignment step. Stacks RF/Audio/Video tokens into a [3B, EMBEDDING_DIM]
    pool with labels replicated ×3, then runs SupCon to cluster same-class
    tokens across all sensors and push apart different-class tokens.
    """
    projection_net.train()

    rf_input    = batch["rf_input"].to(device)
    audio_input = batch["audio_input"].to(device)
    video_input = batch["video_input"].to(device)
    labels      = batch["labels"].to(device)
    B = labels.shape[0]

    token_r, token_a, token_v = projection_net(rf_input, audio_input, video_input)
    # each token: [B, EMBEDDING_DIM]

    unified_features = torch.cat([token_r, token_a, token_v], dim=0)  # [3B, EMBEDDING_DIM]
    unified_labels   = torch.cat([labels,  labels,  labels],  dim=0)  # [3B]

    loss = criterion(unified_features, unified_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ── Verification (mock weights, no file I/O) ──────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")

    class _MockRFEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = IQCNN(num_classes=2)
            for p in self.backbone.parameters(): p.requires_grad = False
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
            return self.projector(x).unsqueeze(1)

    class _MockAudioEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DroneCNN()
            for p in self.backbone.parameters(): p.requires_grad = False
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
            return self.projector(x).unsqueeze(1)

    class _MockVideoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            bb = _build_resnet50(num_classes=1)
            for p in bb.parameters(): p.requires_grad = False
            self.feature_extractor = nn.Sequential(
                bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                bb.layer1, bb.layer2, bb.layer3, bb.layer4, bb.avgpool,
            )
            self.projector = nn.Sequential(
                nn.Linear(2048, EMBEDDING_DIM), nn.BatchNorm1d(EMBEDDING_DIM),
                nn.ReLU(inplace=True), nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
            )
        def forward(self, x):
            with torch.no_grad():
                hidden = self.feature_extractor(x).flatten(1)
            return self.projector(hidden).unsqueeze(1)

    class _MockNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.rf_encoder    = _MockRFEncoder()
            self.audio_encoder = _MockAudioEncoder()
            self.video_encoder = _MockVideoEncoder()
        def forward(self, rf, audio, video):
            return (
                self.rf_encoder(rf).squeeze(1),
                self.audio_encoder(audio).squeeze(1),
                self.video_encoder(video).squeeze(1),
            )

    net       = _MockNet().to(DEVICE)
    criterion = AdaptiveSupConLoss(temperature=0.07).to(DEVICE)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)

    B = 8
    dummy_rf     = torch.randn(B, 2, 128,      device=DEVICE)
    dummy_audio  = torch.randn(B, 1, 128, 44,  device=DEVICE)
    dummy_video  = torch.randn(B, 3, 224, 224, device=DEVICE)
    dummy_labels = torch.tensor([1,1,0,0,1,0,1,0], dtype=torch.long, device=DEVICE)

    net.eval()
    with torch.no_grad():
        tr = net.rf_encoder(dummy_rf)
        ta = net.audio_encoder(dummy_audio)
        tv = net.video_encoder(dummy_video)

    print("Encoder output shapes:")
    print(f"  RF    : {tr.shape}")
    print(f"  Audio : {ta.shape}")
    print(f"  Video : {tv.shape}")
    assert tr.shape == (B, 1, EMBEDDING_DIM)
    assert ta.shape == (B, 1, EMBEDDING_DIM)
    assert tv.shape == (B, 1, EMBEDDING_DIM)
    print("  Shapes OK\n")

    net.train()
    step_loss = train_contrastive_step(
        projection_net=net,
        batch={"rf_input": dummy_rf, "audio_input": dummy_audio,
               "video_input": dummy_video, "labels": dummy_labels},
        optimizer=optimizer, criterion=criterion, device=DEVICE,
    )
    print(f"Contrastive step loss: {step_loss:.4f}")
    print("All checks passed.")
