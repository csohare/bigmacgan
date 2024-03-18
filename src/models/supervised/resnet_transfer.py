from torchvision.models.segmentation import fcn_resnet101
import torch
from torch import nn
from torch.nn import functional as F

class FCNResnetTransfer(nn.Module):
    def __init__(self, input_channels, output_channels, scale_factor=50, **kwargs):
        """
        Loads the fcn_resnet101 model from torch hub,
        then replaces the first and last layer of the network
        in order to adapt it to our current problem, 
        the first convolution of the fcn_resnet must be changed
        to an input_channels -> 64 Conv2d with (7,7) kernel size,
        (2,2) stride, (3,3) padding and no bias.

        The last layer must be changed to be a 512 -> output_channels
        conv2d layer, with (1,1) kernel size and (1,1) stride. 

        A final pooling layer must then be added to pool each 50x50
        patch down to a 1x1 image, as the original FCN resnet is trained to
        have the segmentation be the same resolution as the input.
        
        Input:
            input_channels: number of input channels of the image
            of shape (batch, input_channels, width, height)
            output_channels: number of output channels of prediction,
            prediction is shape (batch, output_channels, width//scale_factor, height//scale_factor)
            scale_factor: number of input pixels that map to 1 output pixel,
            for example, if the input is 800x800 and the output is 16x6
            then the scale factor is 800/16 = 50.
        """
        super().__init__()
        # Load fcn_resnet101 model
        self.model = fcn_resnet101(pretrained=True, **kwargs)
        
        # Replaced first and last layer of network, # channels fits the img as well as # classes we're predicting
        self.model.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model.classifier[4] = nn.Conv2d(512, output_channels, kernel_size=(1,1), stride=(1,1))
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.scale_factor = scale_factor


    def forward(self, x):
        """
        Runs predictions on the modified FCN resnet
        followed by pooling

        Input:
            x: image to run a prediction of, of shape
            (batch, self.input_channels, width, height)
            with width and height divisible by
            self.scale_factor
        Output:
            pred_y: predicted labels of size
            (batch, self.output_channels, width//self.scale_factor, height//self.scale_factor)
        """
        # Get features from the model
        x = self.model(x)['out']

        # Pool the output, resolution is input size divided by scale_factor
        output_size = (x.size(2) // self.scale_factor, x.size(3) // self.scale_factor)
        
        # Each 50x50 patch to 1x1, used adaptive average pooling
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        x = self.adaptive_pool(x)

        # Reshape the pooled output
        x = F.interpolate(x, size=output_size, mode='nearest')
        return x