import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class MILModel(nn.Module):

    def __init__(self, encoder, instance_predictor, bag_predictor):
        super(MILModel, self).__init__()
        self.encoder = encoder

        if instance_predictor is None:
            # Identity instance predictor
            self.instance_predictor = nn.Identity()
        else:
            self.instance_predictor = instance_predictor

        self.bag_predictor = bag_predictor

    def forward(self, x, add_feat):
        """
        inputs:
            - x : batch of images
            - add_feat : additional features to be incorporated in the model (age, lymphocite levels, ...)
        """
        # Size : (batch size, channels, height, width)
        if hasattr(self.encoder, extract_features):
            x = self.encoder.extract_features(x) # extract_features : specific method of EfficientNet, we can adapt it depending on the model we use
        else:
            x = self.encoder(x)
        # Size : (batch size, embedding dimension)
        p = self.instance_predictor(x)
        # Size : (batch size, 1 or embedding dimension)
        if add_feat is not None:
            p = torch.cat([p, add_feat], dim=1)
            # Size : (batch size, 1 + features dim or embedding dimension + features dim)
        out = self.bag_predictor(p)
        # Size : (1, num classes)
        return out

    def make_bags(self,dataloader):
        """
        Generates bags out of a dataloader
        Outputs :
        - bags (np.array(n_bags x n_samples_bag_i x dim_emb))
        - bags_labels (np.array(n_bags))
        """
        print(self.encoder)
        for i, (imgs, pat_ids, labs) in tqdm(enumerate(dataloader)):
          output = self.instance_predictor(self.encoder(imgs.float()))
          if i==0:
            embeddings = output.cpu().detach().numpy()
            ids = pat_ids.cpu().detach().numpy().flatten()
            labels = labs.cpu().detach().numpy().flatten()
          else:
            embeddings = np.vstack((embeddings, output.cpu().detach().numpy()))
            ids = np.append(ids, pat_ids.cpu().detach().numpy().flatten())
            labels = np.append(labels, labs.cpu().detach().numpy().flatten())
        bags = []
        bags_labels = []
        bags_ids = []
        for id in np.unique(ids):
            bags_ids.append(id)
            indic = ids==id
            id_label = labels[indic][0]
            id_embeddings = embeddings[indic]
            bags.append(id_embeddings)
            bags_labels.append(id_label)
        return bags, bags_labels, bags_ids

    def predict(self, dataloader):
        bags, bags_labels, bags_ids = self.make_bags(dataloader)
        pred_labels = self.bag_predictor.predict(bags)
        return np.vstack((bags_ids, pred_labels.flatten())).T


if __name__ == '__main__':
    """
    Example of use, absolutely not working
    """
    from efficientnet_pytorch import EfficientNet

    encoder = EfficientNet.from_pretrained('efficientnet-b0')
    attention = Attention()

    MILModel(encoder, None, attention)
