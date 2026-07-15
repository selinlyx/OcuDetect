import torch
import torch.nn as nn
from torchvision import models

class BaselineModel(nn.Module):
    def __init__(self, num_classes=8, freeze_backbone=True):
        super(BaselineModel, self).__init__()
        self.name = 'Baseline'
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        if freeze_backbone:
            # freeze backbone model weights
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # replace classifier head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)
    

