import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

kernel_size = 2
depth_size1 = 3
depth_size2 = 3


# ==================== ATTENTION MODULES ====================

class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) from CBAM"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) from CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
    
    def forward(self, x):
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x


class AttentionResNetBlock(nn.Module):
    """ResNet Block with CBAM Attention"""
    def __init__(self, module, in_channels):
        super(AttentionResNetBlock, self).__init__()
        self.module = module
        self.cbam = CBAM(in_channels)
    
    def forward(self, x):
        out = self.module(x)
        out = self.cbam(out)
        return out


# ==================== ORIGINAL MODELS (PRESERVED) ====================

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()
        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(
                *list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(original_model.fc.in_features, num_classes)
            )
        elif arch.startswith('densenet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Linear(original_model.classifier.in_features, num_classes)
            )
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            raise ("Finetuning not supported on this architecture yet")
        
        self.modelName = arch
    
    def forward(self, x):
        f = self.features(x)
        if self.modelName.startswith('densenet'):
            out = F.relu(f, inplace=True)
            out = F.avg_pool2d(out, kernel_size=7).view(f.size(0), -1)
            y = self.classifier(out)
        else:
            f = f.view(f.size(0), -1)
            y = self.classifier(f)
        return y


# ==================== ATTENTION RESNET18 WITH HIERARCHICAL SOFTMAX ====================

class FineTuneModel_Hierarchical_Attention(nn.Module):
    """
    Attention ResNet18 with Hierarchical Softmax
    Implements CBAM attention mechanism for enhanced feature learning
    """
    def __init__(self, original_model, arch, args, gr_0_size=3, gr_1_size=3):
        super(FineTuneModel_Hierarchical_Attention, self).__init__()
        self.args = args
        
        if arch.startswith('resnet'):
            # Extract ResNet18 layers
            layers = list(original_model.children())
            
            # Initial layers (conv1, bn1, relu, maxpool)
            self.conv1 = layers[0]
            self.bn1 = layers[1]
            self.relu = layers[2]
            self.maxpool = layers[3]
            
            # ResNet blocks with attention
            self.layer1 = AttentionResNetBlock(layers[4], 64)   # 64 channels
            self.layer2 = AttentionResNetBlock(layers[5], 128)  # 128 channels
            self.layer3 = AttentionResNetBlock(layers[6], 256)  # 256 channels
            self.layer4 = AttentionResNetBlock(layers[7], 512)  # 512 channels
            
            # Global average pooling
            self.avgpool = layers[8]
            
            # Get output size
            output_size = original_model.fc.in_features  # 512 for ResNet18
            
            # Dropout for regularization (helps prevent overfitting)
            self.dropout = nn.Dropout(0.5)
            
            # Hierarchical softmax heads
            self.level_0 = nn.Linear(output_size, 1)  # Binary: Clupeids vs Sandeels
            self.level_1_0 = nn.Linear(output_size, gr_0_size)  # 3 sandeel species
            self.level_1_1 = nn.Linear(output_size, gr_1_size)  # 3 clupeid species
            
        elif arch.startswith('densenet'):
            self.features = original_model.features
            # Add attention to DenseNet features
            output_size = original_model.classifier.in_features
            self.attention = CBAM(output_size)
            self.dropout = nn.Dropout(0.5)
            
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)
            
        elif arch.startswith('alexnet'):
            self.features = original_model.features
            output_size = 256 * 6 * 6
            self.dropout = nn.Dropout(0.5)
            
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)
            
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            output_size = 25088
            self.dropout = nn.Dropout(0.5)
            
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)
        else:
            raise ("Finetuning not supported on this architecture yet")
        
        self.modelName = arch
        self.arch = arch
    
    def forward(self, x):
        if self.arch.startswith('resnet'):
            # Forward through ResNet with attention
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            features = torch.flatten(x, 1)
            features = self.dropout(features)
            
        elif self.arch.startswith('densenet'):
            features = self.features(x)
            features = F.relu(features, inplace=True)
            features = F.avg_pool2d(features, kernel_size=7)
            features = self.attention(features)
            features = features.view(features.size(0), -1)
            features = self.dropout(features)
            
        else:
            features = self.features(x)
            features = features.view(features.size(0), -1)
            features = self.dropout(features)
        
        return features


# Original hierarchical model (kept for backward compatibility)
class FineTuneModel_Hierarchical(nn.Module):
    def __init__(self, original_model, arch, args, gr_0_size=3, gr_1_size=3):
        super(FineTuneModel_Hierarchical, self).__init__()
        self.args = args
        
        if arch.startswith('alexnet'):
            self.features = original_model.features
            output_size = 256 * 6 * 6
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)
            
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(
                *list(original_model.children())[:-1])
            output_size = original_model.fc.in_features
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)
            
        elif arch.startswith('densenet'):
            self.features = original_model.features
            output_size = original_model.classifier.in_features
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)
            
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            output_size = 25088
            self.level_0 = nn.Linear(output_size, 1)
            self.level_1_0 = nn.Linear(output_size, gr_0_size)
            self.level_1_1 = nn.Linear(output_size, gr_1_size)
        else:
            raise ("Finetuning not supported on this architecture yet")
        
        self.modelName = arch
    
    def forward(self, x):
        features = self.features(x)
        if self.modelName.startswith('densenet'):
            features = F.relu(features, inplace=True)
            features = F.avg_pool2d(features, kernel_size=7).view(
                features.size(0), -1)
        else:
            features = features.view(features.size(0), -1)
        
        return features


class CNNs(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), n_outputs=17):
        super(CNNs, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, depth_size1, kernel_size),
            nn.ReLU(),
            nn.Conv2d(depth_size1, depth_size2, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        
        self.flat_fts = self.get_flat_fts(input_shape, self.features)
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_fts, n_outputs),
            nn.Sigmoid()
        )
    
    def get_flat_fts(self, input_shape, fts):
        f = fts(Variable(torch.rand(1, *input_shape)))
        return int(np.prod(f.size()[1:]))
    
    def forward(self, x):
        fts = self.features(x)
        flat_fts = fts.view(-1, self.flat_fts)
        out = self.classifier(flat_fts)
        return out
