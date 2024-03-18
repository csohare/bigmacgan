from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import torch
from torch import nn
from torch.nn import functional as F

class DeepLabV3Transfer(nn.Module):
    def __init__(self, input_channels, output_channels, dropout_rate=0.5):
        super().__init__()
        # Load a pre-trained DeepLabV3 model
        self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT, progress=False)

        # Replace the first convolution layer in the backbone with a new one
        # that matches the input_channels and includes batch normalization and dropout
        self.model.backbone.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Replace the classifier's last convolution layer to adapt to the required output_channels
        self.model.classifier[4] = nn.Conv2d(256, output_channels, kernel_size=1)

        # Add batch normalization after the modified classifier's convolution
        self.batch_norm = nn.BatchNorm2d(output_channels)

        # Insert an adaptive layer to ensure the final output is 4x4 in size
        self.adaptive_output_layer = nn.AdaptiveAvgPool2d((4, 4))

        # Initialize weights for the newly created layers
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Run the modified DeepLabV3 model
        x = self.model(x)['out']

        # Apply batch normalization
        x = self.batch_norm(x)
        
        # Resize the output to 4x4
        x = self.adaptive_output_layer(x)
        
        return x
    
    @property
    def backbone(self):
        return self.model.backbone

    @property
    def classifier(self):
        return self.model.classifier
