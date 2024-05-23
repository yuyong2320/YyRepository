import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        max = self.max_pool(x)
        avg = self.fc2(self.relu1(self.fc1(avg)))
        max = self.fc2(self.relu1(self.fc1(max)))
        out = self.sigmoid(avg + max)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], dim=1)
        x = self.conv1(x)
        out = self.sigmoid(x)
        return out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class CBAMResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16, kernel_size=7, use_bn=True):
        super(CBAMResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.cbam = CBAM(out_channels, reduction_ratio=reduction_ratio, kernel_size=kernel_size)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        # print(identity.shape)
        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out

# class CBAMResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000, reduction_ratio=16, kernel_size=7, use_bn=True):
#         super(CBAMResNet, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0], stride=1, reduction_ratio=reduction_ratio, kernel_size=kernel_size, use_bn=use_bn)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction_ratio=reduction_ratio, kernel_size=kernel_size, use_bn=use_bn)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction_ratio=reduction_ratio, kernel_size=kernel_size, use_bn=use_bn)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction_ratio=reduction_ratio, kernel_size=kernel_size, use_bn=use_bn)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)
#
#     def _make_layer(self, block, out_channels, blocks, stride=1, reduction_ratio=16, kernel_size=7, use_bn=True):
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, reduction_ratio, kernel_size, use_bn))
#         self.in_channels = out_channels
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels, reduction_ratio=reduction_ratio, kernel_size=kernel_size, use_bn=use_bn))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.maxpool(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         return out



model = CBAMResidualBlock(1,32)

# 构造一个假设的输入
input = torch.randn(16,1,28,28)

# 计算输出
output = model(input)

# 输出结果
print(output.shape)