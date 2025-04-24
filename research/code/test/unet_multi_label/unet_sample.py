import torch
import torch.nn as nn
import torch.nn.functional as F

"""
https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset
"""

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = DoubleConv(3, 64)
        self.down_conv2 = DoubleConv(64, 128)
        self.down_conv3 = DoubleConv(128, 256)
        self.down_conv4 = DoubleConv(256, 512)
        self.down_conv5 = DoubleConv(512, 1024)
        self.up_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        self.up_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        self.up_trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up_trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.down_conv1(x)
        x = self.maxpool(conv1)
        conv2 = self.down_conv2(x)
        x = self.maxpool(conv2)
        conv3 = self.down_conv3(x)
        x = self.maxpool(conv3)
        conv4 = self.down_conv4(x)
        x = self.maxpool(conv4)
        conv5 = self.down_conv5(x)
        # Decoder
        x = self.up_trans1(conv5)
        x = self.up_conv1(torch.cat([x, conv4], dim=1))
        x = self.up_trans2(x)
        x = self.up_conv2(torch.cat([x, conv3], dim=1))
        x = self.up_trans3(x)
        x = self.up_conv3(torch.cat([x, conv2], dim=1))
        x = self.up_trans4(x)
        x = self.up_conv4(torch.cat([x, conv1], dim=1))
        # Output
        out = self.out_conv(x)
        return out

# Example usage
num_classes = 20
model = UNet(num_classes)
input_tensor = torch.randn(1, 3, 256, 256) 
output_tensor = model(input_tensor)
print(output_tensor.shape) # Expected output: torch.Size([1, 20, 256, 256])

# Loss function
criterion = nn.CrossEntropyLoss()

# Dummy target (replace with your actual target)
target_tensor = torch.randint(0, num_classes, (1, 256, 256)).long()

# Calculate loss
loss = criterion(output_tensor, target_tensor)
print(loss)
