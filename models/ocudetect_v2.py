import torch
import torch.nn as nn
from torchvision import models

class OcuDetect2(nn.Module):
    '''
    ocudetect_v1: CNN with self-attention for multi-label ocular disease detection

    Architecture Description
    1) Backbone: EfficientNet-B0 (pretrained on ImageNet)
    2) Self-Attention: Multi-head attention for capturing non-local dependencies
    3) Pooling: Global Average Pooling
    4) Custome Classifier: FC layers with ReLU activation
    5) Output: Sigmoid activation for multi-label classification

    - use_attention is a constructor argument. When False, the self-attention
      block (proj_down / preattention_norm / attention / proj_up) is not even
      built, so the model has fewer parameters, not just a skipped step. This is
      what powers the self-attention ablation (attention vs. no attention).

    Last Updated: 2026-08-04
    '''

    def __init__(self, num_classes=8, freeze_backbone=True, num_heads=1,
                 dropout_rate=0.5, attn_dim=256, use_attention=True):
        super(OcuDetect2, self).__init__()
        self.name = 'OcuDetect_v2'
        self.attn_dim = attn_dim

        # ablation switch: set False to test the model with the self-attention
        # block removed entirely (same backbone + classifier, no attention)
        self.use_attention = use_attention

        # 1) BACKBONE ===============================
        # load pre-trained EfficientNet-B3 as backbone
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

        # freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # EfficientNet-B3 feature dimension: 1536
        self.feature_dim = 1536

        # remove EfficientNet's original classifier
        self.backbone.classifier = nn.Identity()

        # 2) SELF-ATTENTION BLOCK =====================
        # only built when use_attention=True, so the no-attention ablation
        # variant genuinely has fewer parameters rather than just skipping
        # these layers at forward time
        if self.use_attention:
            self.proj_down = nn.Linear(self.feature_dim, self.attn_dim)

            # layer normalization before attention
            self.preattention_norm = nn.LayerNorm(self.attn_dim)

            # multihead attention
            self.attention = nn.MultiheadAttention(
                embed_dim=self.attn_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout_rate
            )

            self.proj_up = nn.Linear(self.attn_dim, self.feature_dim)

        # 3) GLOBAL AVERAGE POOLING ==================
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 4) CLASSIFIER HEAD ==========================
        # FC layers with ReLU activation
        self.classifier = nn.Sequential(
            # Layer 1: 1280 -> 256 (Heavy expansion)
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),  # Higher dropout on big layer

            # Layer 2: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.0),

            # Layer 3: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),

            # Layer 4: 64 -> 8 (Output)
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):

        # extract features from EfficientNet-B0 backbone
        x = self.backbone.features(x) # x shape: [batch, 1280, 7, 7]

        # reshape for self-attention
        batch_size = x.shape[0]
        x = x.view(batch_size, self.feature_dim, -1) # [batch, 1280, 7, 7] -> [batch, 1280, 49]
        x = x.permute(0, 2, 1) # [batch, 1280, 49] -> [batch, 49, 1280] to match expected format

        # skipped entirely when use_attention=False (ablation mode) -- x just
        # passes straight through to global average pooling below
        if self.use_attention:
            x = self.proj_down(x)
            # layer normalization
            x = self.preattention_norm(x)

            # apply self attention
            attention_out, _ = self.attention(x, x, x)
            x = x + attention_out
            x = self.proj_up(x)

        # global average pooling
        x = x.permute(0, 2, 1) # [batch, 49, 1280] -> [batch, 1280, 49]
        x = self.global_pool(x)  # [batch, 1280, 1]
        x = x.squeeze(-1)  # [batch, 1280]

        # classifier with ReLU activation
        x = self.classifier(x)

        return x


    def unfreeze_backbone(self):
        '''
        Unfreezes the EfficientNet-B0 backbone for hyperparameter tuning
        '''
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen for hyperparameter tuning.")

    def freeze_backbone(self):
        '''
        Freezes the EfficientNet-B0 backbone
        '''
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen.")