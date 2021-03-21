import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, patient_bs=6):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.patient_bs = patient_bs
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, 3)
        self.bn32 = nn.BatchNorm2d(16)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*3*3, 25)  # 6*6 from image dimension
        self.fc3 = nn.Linear(25, 10)
        self.fc4 = nn.Linear(patient_bs*10, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.bn1(x)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.bn2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn32(x)
        # print(x.shape)
        x = x.reshape(-1, 16*3*3)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        # print(x.shape)
        x = x.flatten()
        x = torch.sigmoid(self.fc4(x))
        return x
