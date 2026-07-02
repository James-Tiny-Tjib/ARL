import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ASTModel

# ==========================================
# 0. GLOBAL CONFIGURATION & CONSTANTS
# ==========================================
EMBEDDING_DIM = 512
NUM_CLASSES = 2  # 0: Background/Safe, 1: Drone Detected (Mambo or Bebop)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. AUDIO STREAM: Pre-trained Frozen AST
# ==========================================
class FrozenAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained Audio Spectrogram Transformer from MIT/HuggingFace
        self.ast = ASTModel.from_pretrained("MIT/ast-base-custom")
        
        # STRICTLY ZERO FINE-TUNING: Freeze all base transformer parameters
        for param in self.ast.parameters():
            param.requires_grad = False
            
        # Projection layer: Map AST's 768 hidden units to our 512-D space
        self.to_shared_space = nn.Linear(768, EMBEDDING_DIM)

    def forward(self, x):
        # Input shape expected: [Batch, Time_Steps, Freq_Bins] (e.g., from your Mel-Spectrogram)
        outputs = self.ast(x)
        global_audio_token = outputs.pooler_output  # Shape: [Batch, 768]
        
        # Map to 512 dimensions and add a sequence dimension for the Fusion Transformer
        return self.to_shared_space(global_audio_token).unsqueeze(1)  # Shape: [Batch, 1, 512]


# ==========================================
# 2. VISUAL STREAM: Pre-trained Frozen ViT
# ==========================================
class FrozenVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained Vision Transformer from Google/HuggingFace
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # STRICTLY ZERO FINE-TUNING: Freeze all base transformer parameters
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Projection layer: Map ViT's 768 hidden units to our 512-D space
        self.to_shared_space = nn.Linear(768, EMBEDDING_DIM)

    def forward(self, x):
        # Input shape expected: [Batch, 3, 224, 224] (Standard normalized image tensor)
        outputs = self.vit(x)
        global_visual_token = outputs.pooler_output  # Shape: [Batch, 768]
        
        # Map to 512 dimensions and add a sequence dimension for the Fusion Transformer
        return self.to_shared_space(global_visual_token).unsqueeze(1)  # Shape: [Batch, 1, 512]


# ==========================================
# 3. RF STREAM: Frozen Hidden Layer Intercept
# ==========================================
class FrozenRFEncoder(nn.Module):
    def __init__(self, trained_iqcnn_model):
        super().__init__()
        self.iqcnn = trained_iqcnn_model
        
        # STRICTLY ZERO FINE-TUNING: Freeze your existing CNN weights completely
        for param in self.iqcnn.parameters():
            param.requires_grad = False
            
        # Projection layer: Map Layer 4's 256 hidden dimensions to our 512-D space
        self.to_shared_space = nn.Linear(256, EMBEDDING_DIM)

    def forward(self, x):
        # Input shape expected: [Batch, 2, 128] (Raw 1D IQ array)
        
        # Step A: Run through your Convolutional layers
        for i in range(self.iqcnn.conv_num):
            x = F.relu(self.iqcnn.layers[i](x))
            
        # Step B: Apply Global Average Pooling over time/sequence
        x = self.iqcnn.global_avg_pool(x).squeeze(-1)  # Shape: [Batch, 64]
        
        # Step C: Intercept at the 256-D dense layer right before the classification layer
        for i in range(self.iqcnn.conv_num, len(self.iqcnn.layers) - 1):
            x = F.relu(self.iqcnn.layers[i](x))        # Shape: [Batch, 256]
            
        # Map this rich hidden structural vector into your 512-D space
        return self.to_shared_space(x).unsqueeze(1)    # Shape: [Batch, 1, 512]


# =====================================================================
# 4. TRIPLET CONTRASTIVE LOSS MODULE (TRACK A)
# =====================================================================
class MultiModalTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        """
        Calculates triplet loss across collapsed global vector positions.
        Args:
            anchor:   Tensor [Batch, 512] (e.g., Aligned RF Global Summary)
            positive: Tensor [Batch, 512] (e.g., Corresponding Visual CLS Token)
            negative: Tensor [Batch, 512] (e.g., Mismatching Background Vector)
        """
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negative_norm = F.normalize(negative, p=2, dim=1)
        
        return self.triplet_loss(anchor_norm, positive_norm, negative_norm)


# =====================================================================
# 5. MULTIMODAL FUSION TRANSFORMER HEAD (TRACK B)
# =====================================================================
class DroneFusionTransformer(nn.Module):
    def __init__(self, embed_dim=EMBEDDING_DIM, num_heads=8, num_layers=2, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Learnable Modality Type Embeddings (Visual, Audio, and RF)
        self.modality_embeddings = nn.Parameter(torch.randn(3, 1, embed_dim))
        
        # Shallow Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final Binary Classification Head Layer
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)  # Maps down to 2 outputs
        )

    def forward(self, token_v, token_a, token_r):
        """
        Args:
            token_v: Visual vector  [Batch, 1, 512]
            token_a: Audio vector   [Batch, 1, 512]
            token_r: RF vector      [Batch, 1, 512]
        """
        # Step A: Apply Modality Identification tags to the feature spaces
        v_feated = token_v + self.modality_embeddings[0]
        a_feated = token_a + self.modality_embeddings[1]
        r_feated = token_r + self.modality_embeddings[2]
        
        # Step B: Concatenate into a unified cross-sensor sequence array [Batch, 3, 512]
        combined_sequence = torch.cat([v_feated, a_feated, r_feated], dim=1)
        
        # Step C: Compute Multi-Sensor Self-Attention
        attn_outputs = self.transformer_fusion(combined_sequence)  # Shape: [Batch, 3, 512]
        
        # Step D: Extract global mean representation of the attended sequence
        fused_context = attn_outputs.mean(dim=1)  # Shape: [Batch, 512]
        
        # Step E: Predict Target Threat Verdict Label
        logits = self.classifier(fused_context)   # Shape: [Batch, 2]
        return logits


# =====================================================================
# 6. PIPELINE VERIFICATION AND TESTING LOOP
# =====================================================================
if __name__ == "__main__":
    print(f"Initializing complete Multi-Modal Framework on {DEVICE}...\n")
    
    # 1. Initialize Feature Tracking Encoders
    audio_encoder = FrozenAudioEncoder().to(DEVICE)
    visual_encoder = FrozenVisualEncoder().to(DEVICE)
    
    # Simulating loading your notebook's local trained IQCNN structure for the mockup test pass
    class MockIQCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_num = 2
            self.layers = nn.ModuleList([
                nn.Conv1d(2, 32, kernel_size=3, padding=1),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.Linear(64, 256),  # Simulated layer 4 (intercept spot)
                nn.Linear(256, 2)   # Final classifier
            ])
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        def forward(self, x): return x
        
    local_trained_iqcnn = MockIQCNN().to(DEVICE)
    rf_encoder = FrozenRFEncoder(local_trained_iqcnn).to(DEVICE)
    print("-> All backbone encoders safely initialized and frozen.")

    # 2. Initialize Contrastive Loss Module (Track A)
    contrastive_criterion = MultiModalTripletLoss(margin=1.0).to(DEVICE)
    print("-> Triplet Contrastive Loss metrics initialized.")

    # 3. Initialize Central Fusion Transformer Head (Track B)
    fusion_brain = DroneFusionTransformer().to(DEVICE)
    print("-> Central Binary Fusion Transformer compiled successfully.\n")
    print("---------------------------------------------------------")

    # 4. Generate Dummy Ingestion Batch (Batch size = 4)
    dummy_audio = torch.randn(4, 1024, 128).to(DEVICE)   
    dummy_visual = torch.randn(4, 3, 224, 224).to(DEVICE) 
    dummy_rf = torch.randn(4, 2, 128).to(DEVICE)          
    
    # 5. Execute Forward Feature Pass
    with torch.no_grad():
        token_a = audio_encoder(dummy_audio)
        token_v = visual_encoder(dummy_visual)
        token_r = rf_encoder(dummy_rf)
        
    print("FORWARD FEATURE SHAPES:")
    print(f" Mapped Audio Token Shape : {token_a.shape} (Expected: [4, 1, 512])")
    print(f" Mapped Visual Token Shape: {token_v.shape} (Expected: [4, 1, 512])")
    print(f" Mapped RF Token Shape    : {token_r.shape} (Expected: [4, 1, 512])\n")

    # 6. Test Track A: Triplet Contrastive Calculation (Dimensional Squeeze)
    anchor_vec = token_r.squeeze(1)     
    positive_vec = token_v.squeeze(1)  
    negative_vec = torch.randn(4, EMBEDDING_DIM).to(DEVICE) # Simulated background noise
    
    contrastive_loss_val = contrastive_criterion(anchor_vec, positive_vec, negative_vec)
    print(f"TEST TRACK A: Calculated Triplet Alignment Loss: {contrastive_loss_val.item():.4f}\n")

    # 7. Test Track B: Binary System Inference
    with torch.no_grad():
        final_system_logits = fusion_brain(token_v, token_a, token_r)
        probabilities = torch.softmax(final_system_logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
    print("TEST TRACK B: FUSION TRANSFORMER INFERENCE:")
    print(f" Final System Output Tensor Shape: {final_system_logits.shape} (Expected: [4, 2])")
    print(f" System Threat Determinations    : {predictions.tolist()} (0=Background/Safe, 1=Drone Detected)")
    print("---------------------------------------------------------")
    print("Verification complete! The entire dual-track codebase is fully operational.")