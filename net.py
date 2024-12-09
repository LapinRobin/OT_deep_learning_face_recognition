import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First conv layer: 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Max pooling layer with 2x2 window and stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # Second conv layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # First fully connected layer: 16*5*5 inputs, 32 outputs
        self.fc1 = nn.Linear(16 * 5 * 5, 32)
        # Second fully connected layer: 32 inputs, 16 outputs
        self.fc2 = nn.Linear(32, 16)
        # Output layer: 16 inputs, 2 outputs (one for each class)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        # First conv + ReLU + pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the 2D feature maps into 1D vector
        x = x.view(-1, 16 * 5 * 5)
        '''
        First fully connected layer with ReLU
        -1 in view() automatically calculates the batch size dimension
        16 * 5 * 5 = 400 comes from:
          - Starting with 32x32 input image
          - After conv1 (5x5 kernel): 28x28x6 
          - After first pooling (2x2): 14x14x6
          - After conv2 (5x5 kernel): 10x10x16
          - After second pooling (2x2): 5x5x16
          - Flattened size = 16 channels * 5 height * 5 width = 400
        '''
        x = F.relu(self.fc1(x))
        # Second fully connected layer with ReLU
        x = F.relu(self.fc2(x))
        # Output layer (no activation - will be handled by loss function)
        x = self.fc3(x)
        return x


