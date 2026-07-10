import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

EMBEDDING_DIM = 256
NUM_CLASSES   = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR           = "./ARL"
RF_WEIGHTS_PATH    = f"{BASE_DIR}/sensor_client.pt"
AUDIO_WEIGHTS_PATH = f"{BASE_DIR}/drone_multi_classifier.pt"
VIDEO_WEIGHTS_PATH = f"{BASE_DIR}/resnet50_drone_weights.pth"


# ── Backbone architectures ────────────────────────────────────────────────────

class IQCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.layer_dims = []
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(2,  8,  kernel_size=7, padding=3, bias=False))
        self.layers.append(nn.Conv1d(8,  16, kernel_size=7, padding=3, bias=False))
        self.layers.append(nn.Conv1d(16, 32, kernel_size=7, padding=3, bias=False))
        self.layers.append(nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False))
        self.conv_num = len(self.layers)
        self.layers.append(nn.Linear(64,  256, bias=False))          # index 4, intercept here
        self.layers.append(nn.Linear(256, num_classes, bias=False))  # index 5, head
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for i in range(self.conv_num):
            x = F.relu(self.layers[i](x))
        x = self.global_avg_pool(x).squeeze(-1)
        for i in range(self.conv_num, len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)


class DroneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 32 * 11, 128), nn.ReLU(), nn.Dropout(0.3),  # fc[0..2], intercept here
            nn.Linear(128, 3),                                           # fc[3], head
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


def _build_resnet50(num_classes: int = 1) -> nn.Module:
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ── Weight loaders ────────────────────────────────────────────────────────────

def load_iqcnn(path: str = RF_WEIGHTS_PATH) -> IQCNN:
    m = IQCNN(num_classes=2)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()

def load_drone_cnn(path: str = AUDIO_WEIGHTS_PATH) -> DroneCNN:
    m = DroneCNN()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()

def load_resnet50(path: str = VIDEO_WEIGHTS_PATH) -> nn.Module:
    m = _build_resnet50(num_classes=1)
    ckpt = torch.load(path, map_location="cpu")
    m.load_state_dict(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
    return m.eval()


# ── Frozen encoders ───────────────────────────────────────────────────────────

class FrozenRFEncoder(nn.Module):
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
        with torch.no_grad():
            for i in range(self.backbone.conv_num):
                x = F.relu(self.backbone.layers[i](x))
            x = self.backbone.global_avg_pool(x).squeeze(-1)
            for i in range(self.backbone.conv_num, len(self.backbone.layers) - 1):
                x = F.relu(self.backbone.layers[i](x))          # [B, 256]
        return self.projector(x).unsqueeze(1)                    # [B, 1, 256]


class FrozenAudioEncoder(nn.Module):
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
        with torch.no_grad():
            x = self.backbone.conv_layers(x)
            x = self.backbone.fc_layers[0](x)   # Linear → [B, 128]
            x = self.backbone.fc_layers[1](x)   # ReLU
            x = self.backbone.fc_layers[2](x)   # Dropout (no-op in eval)
        return self.projector(x).unsqueeze(1)   # [B, 1, 256]


class FrozenVideoEncoder(nn.Module):
    def __init__(self, weights_path: str = VIDEO_WEIGHTS_PATH):
        super().__init__()
        self.backbone = load_resnet50(weights_path)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,  self.backbone.bn1,
            self.backbone.relu,   self.backbone.maxpool,
            self.backbone.layer1, self.backbone.layer2,
            self.backbone.layer3, self.backbone.layer4,
            self.backbone.avgpool,                       # [B, 2048, 1, 1]
        )
        self.projector = nn.Sequential(
            nn.Linear(2048, EMBEDDING_DIM),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        )

    def forward(self, x):
        with torch.no_grad():
            hidden = self.feature_extractor(x).flatten(1)  # [B, 2048]
        return self.projector(hidden).unsqueeze(1)          # [B, 1, 256]


# ── Multi-modal projection network ────────────────────────────────────────────

class MultiModalProjectionNetwork(nn.Module):
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
        token_r = self.rf_encoder(rf_input).squeeze(1)       # [B, 256]
        token_a = self.audio_encoder(audio_input).squeeze(1) # [B, 256]
        token_v = self.video_encoder(video_input).squeeze(1) # [B, 256]
        return token_r, token_a, token_v


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

class AdaptiveSupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        N = features.shape[0]

        z        = F.normalize(features, p=2, dim=1)
        sim      = torch.matmul(z, z.T) / self.temperature
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim      = sim.masked_fill(self_mask, float("-inf"))

        labels   = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
        num_pos  = pos_mask.sum(dim=1).float()
        valid    = num_pos > 0

        log_prob = sim - torch.logsumexp(sim.masked_fill(self_mask, float("-inf")), dim=1, keepdim=True)
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (num_pos + 1e-8)

        return -mean_log_prob_pos[valid].mean()


# ── Contrastive training step ─────────────────────────────────────────────────

def train_contrastive_step(projection_net, batch, optimizer, criterion, device) -> float:
    projection_net.train()

    rf_input    = batch["rf_input"].to(device)
    audio_input = batch["audio_input"].to(device)
    video_input = batch["video_input"].to(device)
    labels      = batch["labels"].to(device)

    token_r, token_a, token_v = projection_net(rf_input, audio_input, video_input)

    # stack all three modality tokens → [3B, 256], labels → [3B]
    features = torch.cat([token_r, token_a, token_v], dim=0)
    lbls     = torch.cat([labels,  labels,  labels],  dim=0)

    loss = criterion(features, lbls)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ── Fusion Transformer ────────────────────────────────────────────────────────

class ModalityFusionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim:   int   = EMBEDDING_DIM,
        num_heads:   int   = 4,
        num_layers:  int   = 2,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = 0.1,
    ):
        super().__init__()
        # one learnable ID tag per sensor [3, 256]
        self.modality_tags = nn.Parameter(torch.randn(3, embed_dim) * 0.02)
        # prepended CLS token
        self.cls_token     = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, token_r, token_a, token_v):
        B = token_r.shape[0]

        # inject modality ID tags
        x_r = token_r + self.modality_tags[0]
        x_a = token_a + self.modality_tags[1]
        x_v = token_v + self.modality_tags[2]

        # [B, 3, 256] → prepend CLS → [B, 4, 256]
        seq = torch.cat([self.cls_token.expand(B, -1, -1),
                         torch.stack([x_r, x_a, x_v], dim=1)], dim=1)

        # self-attention across all 4 tokens
        cls_out = self.transformer(seq)[:, 0, :]   # harvest CLS → [B, 256]
        return self.classifier(cls_out)             # [B, num_classes]


# ── Fusion training step ──────────────────────────────────────────────────────

def train_fusion_step(projection_net, fusion_net, batch, optimizer, criterion, device) -> float:
    projection_net.eval()
    fusion_net.train()

    rf_input    = batch["rf_input"].to(device)
    audio_input = batch["audio_input"].to(device)
    video_input = batch["video_input"].to(device)
    labels      = batch["labels"].to(device)

    with torch.no_grad():
        token_r, token_a, token_v = projection_net(rf_input, audio_input, video_input)

    logits = fusion_net(token_r, token_a, token_v)
    loss   = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ── Verification (mock weights) ───────────────────────────────────────────────

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
            return (self.rf_encoder(rf).squeeze(1),
                    self.audio_encoder(audio).squeeze(1),
                    self.video_encoder(video).squeeze(1))

    B            = 8
    dummy_rf     = torch.randn(B, 2, 128,      device=DEVICE)
    dummy_audio  = torch.randn(B, 1, 128, 44,  device=DEVICE)
    dummy_video  = torch.randn(B, 3, 224, 224, device=DEVICE)
    dummy_labels = torch.tensor([1,1,0,0,1,0,1,0], dtype=torch.long, device=DEVICE)

    net       = _MockNet().to(DEVICE)
    supcon    = AdaptiveSupConLoss(temperature=0.07).to(DEVICE)
    opt_proj  = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)

    # shape check
    net.eval()
    with torch.no_grad():
        tr, ta, tv = net(dummy_rf, dummy_audio, dummy_video)
    assert tr.shape == (B, EMBEDDING_DIM)
    assert ta.shape == (B, EMBEDDING_DIM)
    assert tv.shape == (B, EMBEDDING_DIM)
    print(f"Projection shapes: RF {tr.shape}  Audio {ta.shape}  Video {tv.shape}")

    # contrastive step
    net.train()
    con_loss = train_contrastive_step(net, {"rf_input": dummy_rf, "audio_input": dummy_audio,
                                            "video_input": dummy_video, "labels": dummy_labels},
                                      opt_proj, supcon, DEVICE)
    print(f"SupCon loss:       {con_loss:.4f}")

    # fusion step
    fusion   = ModalityFusionTransformer().to(DEVICE)
    opt_fuse = torch.optim.Adam(fusion.parameters(), lr=1e-4)
    ce       = nn.CrossEntropyLoss()

    fuse_loss = train_fusion_step(net, fusion, {"rf_input": dummy_rf, "audio_input": dummy_audio,
                                                "video_input": dummy_video, "labels": dummy_labels},
                                  opt_fuse, ce, DEVICE)
    print(f"Fusion CE loss:    {fuse_loss:.4f}")
    print("\nAll checks passed.")
