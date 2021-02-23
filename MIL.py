import torch
import torch.nn as nn
import torch.nn.functional as F



class MILModel(nn.Module):

    def __init__(self, encoder, instance_predictor, bag_predictor):

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


if __name__ == '__main__':
    """
    Example of use, absolutely not working
    """
    from efficientnet_pytorch import EfficientNet

    encoder = EfficientNet.from_pretrained('efficientnet-b0')
    attention = Attention()

    MILModel(encoder, None, attention)
