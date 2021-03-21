from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_curve, auc
from dataloader import LymphocytesFlat
from torch.utils.data import DataLoader
from MIL import MILModel
import torch
import matplotlib.pyplot as plt

dataset = LymphocytesFlat('trainset')
mil_model = torch.load('trained_models/mil_model_e2e_emb60_cli.ckpt').cuda()
dataloader = DataLoader(dataset, batch_size=15, shuffle=True, num_workers=3)
bags, bags_labels, bags_ids, bags_lymph_count, bags_dobs = mil_model.make_bags(dataloader, 'raw')

y_true = []
y_score = []
for i_bag in tqdm(range(len(bags))):
    bag_images = torch.Tensor(bags[i_bag]).cuda()
    bag_clinical_feat = torch.Tensor([bags_lymph_count[i_bag], bags_dobs[i_bag]]).cuda()
    output = mil_model(bag_images, add_feat = bag_clinical_feat)
    y_true.append(bags_labels[i_bag].item())
    y_score.append(output.item())

#Compute ROC curve and AUC score
fpr, tpr, threshold = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

#Print optimal threshold for the criterion max(TP - FP)
print('optimal threshold:', threshold[np.argmax(tpr - fpr)])

plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
