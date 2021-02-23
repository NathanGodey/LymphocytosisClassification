import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, emb_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, dilation=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, dilation=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, dilation=2)
        self.linear1 = nn.Linear(64 * 50 * 50, emb_dim)
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = torch.sigmoid(self.linear1(x))
        return x

class Decoder(nn.Module):
    def __init__(self, emb_dim):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(emb_dim, 64 * 50 * 50)
        self.conv1 = nn.ConvTranspose2d(64, 32, 3, padding=1, dilation=2)
        self.conv2 = nn.ConvTranspose2d(32, 16, 3, padding=1, dilation=2)
        self.conv3 = nn.ConvTranspose2d(16, 3, 3, padding=1, dilation=2)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = x.reshape(x.shape[0], 64, 50, 50)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

class AutoEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(emb_dim)
        self.decoder = Decoder(emb_dim)

    def forward(self, x):
        emb = self.encoder(x)
        x_hat = self.decoder(emb)
        return x_hat
