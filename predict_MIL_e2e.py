from tqdm import tqdm
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from dataloader import LymphocytesFlat
from torch.utils.data import DataLoader
from MIL import MILModel
import torch
import pandas as pd

model_path = 'trained_models/mil_model_e2e_emb60_cli.ckpt'
save_path = 'e2e_emb60_cli.csv'

dataset = LymphocytesFlat('testset')
mil_model = torch.load(model_path).cuda()
dataloader = DataLoader(dataset, batch_size=15, shuffle=True, num_workers=3)
bags, bags_labels, bags_ids, bags_lymph_count, bags_dobs = mil_model.make_bags(dataloader, 'raw')
threshold = 0.7
print('Threshold for prediction: ', threshold)

predictions = []
ids = []
outputs = []
for i_bag in tqdm(range(len(bags))):
    bag_images = torch.Tensor(bags[i_bag]).cuda()
    bag_clinical_feat = torch.Tensor([bags_lymph_count[i_bag], bags_dobs[i_bag]]).cuda()
    output = mil_model(bag_images, add_feat = bag_clinical_feat)
    outputs.append([output.item()])
    predictions.append(int(output.item()>threshold))
    ids.append(f'P{bags_ids[i_bag]}')

print('Rate of positive predictions: ', np.mean(predictions))

#Save results in a csv file
result_df = pd.DataFrame(zip(ids, predictions), columns = ['ID', 'Predicted'])
result_df.to_csv(save_path, index=False)
