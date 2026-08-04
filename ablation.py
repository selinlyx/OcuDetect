import torch
import torch.nn as nn
from torchvision import models

class NoAttention(nn.Module):
    '''
    Control / ablation baseline: passes features through unchanged.
    '''
    def __init__(self, feature_dim=1280, **kwargs):
        super().__init__()
 
    def forward(self, x):
        # x: [batch, C, H, W]
        return x
 
 
class SelfAttentionBlock(nn.Module):
    '''
    Multi-head self-attention over spatial positions, matching the
    block used in ocudetect_v1.py. Operates on [batch, C, H, W].
    '''
    def __init__(self, feature_dim=1280, attn_dim=256, num_heads=1, dropout_rate=0.5, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
 
        self.proj_down = nn.Linear(feature_dim, attn_dim)
        self.preattention_norm = nn.LayerNorm(attn_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        self.proj_up = nn.Linear(attn_dim, feature_dim)
 
    def forward(self, x):
        batch_size, C, H, W = x.shape
 
        # reshape for attention: [batch, C, H, W] -> [batch, H*W, C]
        x_seq = x.view(batch_size, C, -1).permute(0, 2, 1)
 
        z = self.proj_down(x_seq)
        z = self.preattention_norm(z)
        attn_out, _ = self.attention(z, z, z)
        z = z + attn_out
        z = self.proj_up(z)  # [batch, H*W, C]
 
        # reshape back to [batch, C, H, W]
        out = z.permute(0, 2, 1).view(batch_size, C, H, W)
        return out
 
 
class SEBlock(nn.Module):
    '''
    Squeeze-and-Excitation block: channel-wise attention only.
    Cheap alternative to self-attention -- no spatial/sequence
    interactions, just a per-channel gate learned from global context.
    '''
    def __init__(self, feature_dim=1280, reduction_ratio=16, **kwargs):
        super().__init__()
        reduced_dim = max(feature_dim // reduction_ratio, 8)
 
        # squeeze: global average pool per channel (done in forward)
        # excitation: two FC layers producing a per-channel gate in [0, 1]
        self.excitation = nn.Sequential(
            nn.Linear(feature_dim, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, feature_dim),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        batch_size, C, H, W = x.shape
 
        # squeeze: [batch, C, H, W] -> [batch, C]
        z = x.view(batch_size, C, -1).mean(dim=2)
 
        # excitation: [batch, C] -> [batch, C] gate
        gate = self.excitation(z)
 
        # reweight channels: [batch, C] -> [batch, C, 1, 1]
        gate = gate.view(batch_size, C, 1, 1)
        return x * gate
 
 
class CBAMBlock(nn.Module):
    '''
    Convolutional Block Attention Module: channel attention (SE-style,
    using both avg- and max-pooled context) followed by spatial attention.
    '''
    def __init__(self, feature_dim=1280, reduction_ratio=16, spatial_kernel_size=7, **kwargs):
        super().__init__()
        reduced_dim = max(feature_dim // reduction_ratio, 8)
 
        # --- channel attention ---
        self.channel_mlp = nn.Sequential(
            nn.Linear(feature_dim, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, feature_dim)
        )
 
        # --- spatial attention ---
        self.spatial_conv = nn.Conv2d(
            in_channels=2, out_channels=1,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            bias=False
        )
 
    def forward(self, x):
        batch_size, C, H, W = x.shape
 
        # ----- channel attention -----
        avg_pool = x.view(batch_size, C, -1).mean(dim=2)          # [batch, C]
        max_pool = x.view(batch_size, C, -1).max(dim=2).values     # [batch, C]
 
        channel_gate = torch.sigmoid(
            self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)
        )  # [batch, C]
        channel_gate = channel_gate.view(batch_size, C, 1, 1)
        x = x * channel_gate
 
        # ----- spatial attention -----
        avg_map = x.mean(dim=1, keepdim=True)             # [batch, 1, H, W]
        max_map = x.max(dim=1, keepdim=True).values        # [batch, 1, H, W]
        spatial_input = torch.cat([avg_map, max_map], dim=1)  # [batch, 2, H, W]
 
        spatial_gate = torch.sigmoid(self.spatial_conv(spatial_input))  # [batch, 1, H, W]
        x = x * spatial_gate
 
        return x
 
 
ATTENTION_REGISTRY = {
    "none": NoAttention,
    "self_attention": SelfAttentionBlock,
    "se": SEBlock,
    "cbam": CBAMBlock,
}

class OcuDetectAblation(nn.Module):
    '''
    OcuDetect variant for the attention ablation study. Identical
    backbone, pooling, and classifier head across all configs -- only
    the attention_type argument changes which block sits between the
    backbone features and pooling.
 
    attention_type: one of "none", "self_attention", "se", "cbam"
    '''
 
    def __init__(self, num_classes=8, freeze_backbone=True, attention_type="none",
                 num_heads=1, dropout_rate=0.5, reduction_ratio=16, **attn_kwargs):
        super().__init__()
 
        if attention_type not in ATTENTION_REGISTRY:
            raise ValueError(
                f"Unknown attention_type '{attention_type}'. "
                f"Choose from {list(ATTENTION_REGISTRY.keys())}."
            )
 
        self.name = f"OcuDetect_ablation_{attention_type}"
        self.attention_type = attention_type
        self.feature_dim = 1280
 
        # 1) BACKBONE ===============================
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
 
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
 
        # remove EfficientNet's original classifier; we use our own head
        self.backbone.classifier = nn.Identity()
 
        # 2) ATTENTION BLOCK (swappable) =============
        attn_class = ATTENTION_REGISTRY[attention_type]
        self.attn_block = attn_class(
            feature_dim=self.feature_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            reduction_ratio=reduction_ratio,
            **attn_kwargs
        )
 
        # 3) GLOBAL AVERAGE POOLING ==================
        self.global_pool = nn.AdaptiveAvgPool2d(1)
 
        # 4) CLASSIFIER HEAD ==========================
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        # extract features from EfficientNet-B0 backbone
        x = self.backbone.features(x)  # [batch, 1280, 7, 7]
 
        # apply the swappable attention block (identity if "none")
        x = self.attn_block(x)
 
        # global average pooling
        x = self.global_pool(x)          # [batch, 1280, 1, 1]
        x = x.flatten(1)                 # [batch, 1280]
 
        # classifier
        x = self.classifier(x)
        return x
 
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"[{self.name}] Backbone unfrozen for hyperparameter tuning.")
 
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[{self.name}] Backbone frozen.")


# none: 82504 / 4090052 (2%) trainable
# self_attention:
# se:  
# cbam: