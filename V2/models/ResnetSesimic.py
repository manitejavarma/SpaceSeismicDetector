import torch
import torch.nn as nn
import torchvision.models as models


# Define the modified ResNet34 for start time prediction
class ResNetSeismic(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetSeismic, self).__init__()

        # Load the ResNet-34 model
        self.resnet = models.resnet34(pretrained=pretrained)

        # Modify the first convolutional layer to accept 1 input channel (for spectrogram)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Remove the final fully connected layer (fc) from the original ResNet
        num_ftrs = self.resnet.fc.in_features  # This is 512 in ResNet-34
        self.resnet.fc = nn.Identity()  # Removing ResNet's own FC layer

        # Add your custom fully connected layer for regression (start time prediction)
        self.fc_start = nn.Linear(num_ftrs, 1)  # Start time prediction

    def forward(self, x):
        # Pass through ResNet backbone
        x = self.resnet(x)

        # Start time prediction (linear output)
        start_output = self.fc_start(x)

        return start_output


# Initialize the model and move it to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = ResNetSeismic().to(device)
print(next(model.parameters()).device)
