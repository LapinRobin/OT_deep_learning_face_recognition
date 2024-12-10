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
        # First fully connected layer: 16*6*6 inputs, 32 outputs
        self.fc1 = nn.Linear(16 * 6 * 6, 32)
        # Second fully connected layer: 32 inputs, 16 outputs
        self.fc2 = nn.Linear(32, 16)
        # Output layer: 16 inputs, 2 outputs (one for each class)
        self.fc3 = nn.Linear(16, 2)
        # print("Network initialized with architecture:")
        # print(f"conv1: {self.conv1}")
        # print(f"pool: {self.pool}")
        # print(f"conv2: {self.conv2}")
        # print(f"fc1: {self.fc1}")
        # print(f"fc2: {self.fc2}")
        # print(f"fc3: {self.fc3}")

    def forward(self, x):
        # print(f"\nInput shape: {x.shape}")
        
        # First conv + ReLU + pooling
        x = self.conv1(x)
        # print(f"After conv1 shape: {x.shape}")
        x = F.relu(x)
        # print(f"After ReLU1 shape: {x.shape}")
        x = self.pool(x)
        # print(f"After first pooling shape: {x.shape}")
        
        # Second conv + ReLU + pooling
        x = self.conv2(x)
        # print(f"After conv2 shape: {x.shape}")
        x = F.relu(x)
        # print(f"After ReLU2 shape: {x.shape}")
        x = self.pool(x)
        # print(f"After second pooling shape: {x.shape}")
        
        # Print the total number of elements before flattening
        # print(f"Total elements before flattening: {x.numel()}")
        # print(f"Expected elements after flattening: {16 * 6 * 6}")
        
        # Flatten the 2D feature maps into 1D vector
        try:
            x = x.view(-1, 16 * 6 * 6)
            # print(f"After flattening shape: {x.shape}")
        except RuntimeError as e:
            # print(f"Error during flattening: {str(e)}")
            # print(f"Current tensor shape: {x.shape}")
            # print(f"Attempting to reshape to: [-1, {16 * 6 * 6}]")
            raise
        
        '''
        First fully connected layer with ReLU
        -1 in view() automatically calculates the batch size dimension
        16 * 6 * 6 = 576 comes from:
          - Starting with 36x36 input image
          - After conv1 (5x5 kernel): 32x32x6 
          - After first pooling (2x2): 16x16x6
          - After conv2 (5x5 kernel): 12x12x16
          - After second pooling (2x2): 6x6x16
          - Flattened size = 16 channels * 6 height * 6 width = 576
        '''
        x = F.relu(self.fc1(x))
        # print(f"After fc1 shape: {x.shape}")
        
        # Second fully connected layer with ReLU
        x = F.relu(self.fc2(x))
        # print(f"After fc2 shape: {x.shape}")
        
        # Output layer (no activation - will be handled by loss function)
        x = self.fc3(x)
        # print(f"Final output shape: {x.shape}")
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        # Convolutional layers with BatchNorm and ReLU
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Dropout
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 256 * 2 * 2)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

