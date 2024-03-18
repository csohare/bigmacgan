from torchvision.models.segmentation import deeplabv3_resnet101
import torch
from torch import nn


class DeepLabV3Transfer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        # Load a pre-trained DeepLabV3 model
        self.model = deeplabv3_resnet101(weights=True)

        # Replace the first convolution layer in the backbone
        self.model.backbone.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Verify and replace the classifier's last convolution layer correctly
        in_features = self.model.classifier[4].in_channels  # Replace 256 with the actual in_channels
        self.model.classifier[4] = nn.Conv2d(in_features, output_channels, kernel_size=1)

        # Initialize weights for the newly created layers
        nn.init.kaiming_normal_(self.model.backbone.conv1[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.model.backbone.conv1[1].weight, 1)
        nn.init.constant_(self.model.backbone.conv1[1].bias, 0)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=16, stride=16),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.scaling_layers = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
            # This will get the feature map size down from 200x200 to below 4x4
        )

    def forward(self, x):
        # Run the modified DeepLabV3 model
        x = self.model(x)['out']
        x = self.upsample(x)
        x = self.scaling_layers(x)
        x = nn.functional.adaptive_avg_pool2d(x, (4, 4))
        return x