import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, emb_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.max_pool = nn.MaxPool2d(2, return_indices=True)

        self.linear1 = nn.Linear(64 * 7 * 7, emb_dim*3)
        self.linear2 = nn.Linear(emb_dim * 3, emb_dim)

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x))
        x, indices1 = self.max_pool(x)
        x = F.leaky_relu(self.conv2(x))
        x, indices2 = self.max_pool(x)
        x = F.leaky_relu(self.conv3(x))
        x, indices3 = self.max_pool(x)

        x = x.reshape(x.shape[0], -1)
        x = F.leaky_relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x, [indices1, indices2, indices3]

class Decoder(nn.Module):
    def __init__(self, emb_dim):
        super(Decoder, self).__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(emb_dim, 3*emb_dim)
        self.linear2 = nn.Linear(3*emb_dim, 64*7*7)

        self.conv1 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 3, 3, padding=1)

        self.max_unpool = nn.MaxUnpool2d(2)

    def forward(self, x, pool_indices):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = x.reshape(x.shape[0], 64, 7, 7)

        x = self.max_unpool(x, pool_indices[2])
        x = F.leaky_relu(self.conv1(x))
        x = self.max_unpool(x, pool_indices[1])
        x = F.leaky_relu(self.conv2(x))
        x = self.max_unpool(x, pool_indices[0])
        x = torch.sigmoid(self.conv3(x))
        return x

class AutoEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(emb_dim)
        self.decoder = Decoder(emb_dim)

    def forward(self, x):
        emb, pool_indices = self.encoder(x)
        x_hat = self.decoder(emb, pool_indices)
        return x_hat
