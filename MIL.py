import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class BagWeightedMean(nn.Module):
    """
    Attention-based bag-level aggregator using basic MLP layers
    """
    def __init__(self, emb_size):
        """
        Args:
            emb_size (int): Size of the embedding dimension
        """
        super().__init__()
        self.emb_size = emb_size
        self.fc1 = nn.Linear(emb_size,1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, bag_embs):
        emb_w = F.leaky_relu(self.fc1(bag_embs))
        emb_w = F.leaky_relu(self.fc2(emb_w))
        emb_w = F.leaky_relu(self.fc3(emb_w))
        emb_w = F.leaky_relu(self.fc4(emb_w))
        emb_w = emb_w/emb_w.sum(axis = 0)

        weighted_bag_embs = bag_embs * emb_w
        s_weighted_bag_embs = weighted_bag_embs.sum(axis=0)

        return s_weighted_bag_embs


class FinalNet(nn.Module):
    """
    MLP network for the bag-level classifier
    """
    def __init__(self, emb_size):
        """
        Args:
            emb_size (int): Size of the embedding dimension
        """
        super(FinalNet, self).__init__()
        self.fc1 = nn.Linear(emb_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)

        # Threshold the sigmoid to avoid Nan values
        x = 1e-4 + (1-1e-4)*torch.sigmoid(x)
        return x


class MILModel(nn.Module):
    """
    Adaptable end-to-end MIL model combining all components
    """
    def __init__(self, encoder, bag_aggregator, bag_predictor):
        """
        Args:
            encoder (nn.Module): Torch model that computes visual features from images
            bag_aggregator (nn.Module or None): Torch model that aggregates the embeddings into a single bag embedding
            bag_predictor (nn.Module): Torch model that makes the final prediction using the bag_embedding
        """
        super(MILModel, self).__init__()
        self.encoder = encoder

        if bag_aggregator is None:
            # Identity instance predictor
            self.bag_aggregator = nn.Identity()
        else:
            self.bag_aggregator = bag_aggregator

        self.bag_predictor = bag_predictor

    def forward(self, x, add_feat):
        """
        inputs:
            - x : batch of images
            - add_feat : additional features to be incorporated in the model (age, lymphocite count, ...)
        """
        # Size : (batch size, channels, height, width)
        if hasattr(self.encoder, 'extract_features'):
            x = self.encoder.extract_features(x) # extract_features : specific method of EfficientNet, we can adapt it depending on the model we use
        else:
            x = self.encoder(x)

        if type(x)==tuple:
            x = x[0]

        # Size : (batch size, embedding dimension)
        p = self.bag_aggregator(x)
        # Size : (batch size, 1 or embedding dimension)
        if add_feat is not None:
            # Concatenate additional features
            p = torch.cat((p, add_feat), 0)
            # Size : (batch size, 1 + features dim or embedding dimension + features dim)
        out = self.bag_predictor(p)
        # Size : (1, num classes)
        return out

    def make_bags(self, dataloader, mode='emb'):
        """
        Generates bags out of a dataloader
        Args:
        - dataloader (torch.DataLoader): Image dataloader
        - mode ('emb' or 'raw'): 'emb' to get bags of embeddings, 'raw' to get bags of images
        Outputs :
        - bags (np.array(n_bags x n_samples_bag_i x dim_emb))
        - bags_labels (np.array(n_bags))
        - bags_ids (np.array(n_bags))
        - bags_lymph_count (np.array(n_bags))
        - bags_dobs (np.array(n_bags))
        """
        for i, (imgs, pat_ids, labs, lymph_count, dob) in tqdm(enumerate(dataloader)):

            if mode == 'raw':
                output = imgs.float()
            elif mode == 'emb':
                output = self.bag_aggregator(self.encoder(imgs.float()))
            else:
                raise Exception(f'Unknown mode {mode}')

            if i==0:
                embeddings = output.cpu().detach().numpy()
                ids = pat_ids.cpu().detach().numpy().flatten()
                labels = labs.cpu().detach().numpy().flatten()
                lymph_counts = lymph_count.cpu().detach().numpy().flatten()
                dobs = dob.cpu().detach().numpy().flatten()
            else:
                embeddings = np.vstack((embeddings, output.cpu().detach().numpy()))
                ids = np.append(ids, pat_ids.cpu().detach().numpy().flatten())
                labels = np.append(labels, labs.cpu().detach().numpy().flatten())
                lymph_counts = np.append(lymph_counts, lymph_count.cpu().detach().numpy().flatten())
                dobs = np.append(dobs, dob.cpu().detach().numpy().flatten())

        bags = []
        bags_labels = []
        bags_ids = []
        bags_lymph_count = []
        bags_dobs = []
        for id in np.unique(ids): #One bag per id
            bags_ids.append(id)
            indic = ids==id
            id_label = labels[indic][0]
            id_embeddings = embeddings[indic]
            id_lymph_count = lymph_counts[indic][0]
            id_dob = dobs[indic][0]
            bags.append(id_embeddings)
            bags_labels.append(id_label)
            bags_lymph_count.append(id_lymph_count)
            bags_dobs.append(id_dob)

        return bags, bags_labels, bags_ids, bags_lymph_count, bags_dobs

    def predict(self, dataloader):
        bags, bags_labels, bags_ids = self.make_bags(dataloader)
        pred_labels = self.bag_predictor.predict(bags)
        return np.vstack((bags_ids, pred_labels.flatten())).T
