import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64 * 2 * 2, 240)
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, 10)
    
    def forward(self, x):
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        
        # Adjust the view for the fully connected layer
        x = x.reshape(-1, 64 * 2 * 2)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def fuse_model(self):
        # Fuse Conv2d + BatchNorm2d + ReLU for each of the conv/bn/relu triples
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'bn2', 'relu2'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv3', 'bn3', 'relu3'], inplace=True)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inner_channel = 4 * growth_rate
        
        # Reordered to Conv2d -> BatchNorm2d -> ReLU
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return torch.cat([x, self.bottleneck(x)], 1)
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self.bottleneck, ['0', '1', '2'], inplace=True)
        torch.quantization.fuse_modules(self.bottleneck, ['3', '4', '5'], inplace=True)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        # Reordered to Conv2d -> BatchNorm2d -> ReLU
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2)
        )
        
    def forward(self, x):
        return self.downsample(x)
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self.downsample, ['0', '1', '2'], inplace=True)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_layers=[6, 12, 24, 16], reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, num_layers in enumerate(block_layers):
            self.dense_blocks.append(self._make_dense_block(num_channels, growth_rate, num_layers))
            num_channels += num_layers * growth_rate
            
            if i != len(block_layers) - 1:
                out_channels = int(num_channels * reduction)
                self.transitions.append(Transition(num_channels, out_channels))
                num_channels = out_channels
                
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_channels, num_classes)
        
    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        for dense_block, transition in zip(self.dense_blocks, self.transitions):
            x = dense_block(x)
            x = transition(x)
        x = self.dense_blocks[-1](x)
        x = self.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def fuse_model(self):
        for module in self.dense_blocks:
            for submodule in module:
                submodule.fuse_model()
        for module in self.transitions:
            module.fuse_model()