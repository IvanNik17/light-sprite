import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_condition_features):
        super(ConditionalInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.InstanceNorm2d(num_features, affine=False)
        self.fc_gamma = nn.Linear(num_condition_features, num_features)
        self.fc_beta = nn.Linear(num_condition_features, num_features)

    def forward(self, x, condition):
        normalized = self.bn(x)
        gamma = self.fc_gamma(condition)
        beta = self.fc_beta(condition)
        out = gamma.unsqueeze(2).unsqueeze(3) * normalized + beta.unsqueeze(2).unsqueeze(3)
        return out

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_condition_features=3):
        super(Unet, self).__init__()
        self.num_condition_features = num_condition_features
        
        # U-Net encoder
        self.conv1 = double_conv(in_channels, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        
        # Conditional Instance Normalization layers
        self.cin1 = ConditionalInstanceNorm2d(64, num_condition_features)
        self.cin2 = ConditionalInstanceNorm2d(128, num_condition_features)
        self.cin3 = ConditionalInstanceNorm2d(256, num_condition_features)
        self.cin4 = ConditionalInstanceNorm2d(512, num_condition_features)
        
        # U-Net decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv5 = double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6 = double_conv(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7 = double_conv(128, 64)
        self.conv8 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, condition):
        # U-Net encoder
        x1 = self.conv1(x)
        x2 = self.conv2(F.max_pool2d(x1, 2))
        x3 = self.conv3(F.max_pool2d(x2, 2))
        x4 = self.conv4(F.max_pool2d(x3, 2))
        
        # Conditional Instance Normalization
        x1 = self.cin1(x1, condition)
        x2 = self.cin2(x2, condition)
        x3 = self.cin3(x3, condition)
        x4 = self.cin4(x4, condition)
        
        # U-Net decoder
        x = F.relu(self.upconv1(x4))
        x = self.conv5(torch.cat([x, x3], dim=1))
        x = F.relu(self.upconv2(x))
        x = self.conv6(torch.cat([x, x2], dim=1))
        x = F.relu(self.upconv3(x))
        x = self.conv7(torch.cat([x, x1], dim=1))
        x = self.conv8(x)
        return x
