from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from dataloader import LymphocytesFlat
from torch.utils.data import DataLoader
from MIL import MILModel, FinalNet, BagWeightedMean
import torch
import torch.nn as nn
import torch.optim as optim

embedding_dim = 60
nb_add_features = 2
save_model_path = 'trained_models/mil_model_e2e_emb60_cli.ckpt'

# Load three components and create end-to-end model
autoencoder_feature_extractor = torch.load('trained_models/autoencoder60.ckpt').encoder.cuda()
bwm = BagWeightedMean(embedding_dim).cuda()
final_net = FinalNet(embedding_dim + nb_add_features).cuda()
mil_model = MILModel(autoencoder_feature_extractor, bwm, final_net)

# Load bags of images and annotations
dataset = LymphocytesFlat('trainset')
dataloader = DataLoader(dataset, batch_size=15, shuffle=True, num_workers=3)
bags, bags_labels, bags_ids, bags_lymph_count, bags_dobs = mil_model.make_bags(dataloader, 'raw')

# Split (using stratify) in train and val sets
train_bags, val_bags, train_bags_labels, val_bags_labels, \
train_bags_ids, val_bags_ids, train_bags_lymph_count, val_bags_lymph_count, \
train_bags_dobs, val_bags_dobs = \
train_test_split(bags, bags_labels, bags_ids, bags_lymph_count, bags_dobs, test_size=0.22, stratify=bags_labels)

# Set threshold to positive class proportion
threshold = np.mean(train_bags_labels)
print('Threshold for prediction: ', threshold)

# One optimizer per component

# Fine-tuning visual feature extractor
encoder_opt = optim.Adam(autoencoder_feature_extractor.parameters(), lr=1e-6)
# Small learning rate Adam
bwm_opt = optim.Adam(bwm.parameters(), lr=1e-5)
final_opt = optim.Adam(final_net.parameters(), lr=1e-5)

criterion = nn.BCELoss()
best_val_loss = 2000
best_f1 = 0
best_bal_acc = 0
N_epochs=200
batch_size = 4

for i_epoch in range(N_epochs):
    train_loss_avg = 0
    train_predictions = []
    train_probas = []
    print(f'Epoch {i_epoch}/{N_epochs}')
    loss = 0
    for i_bag in tqdm(range(len(train_bags))):
        bag_images = torch.Tensor(train_bags[i_bag]).cuda()
        bag_label = torch.Tensor([train_bags_labels[i_bag]]).cuda()
        bag_clinical_feat = torch.Tensor([train_bags_lymph_count[i_bag], train_bags_dobs[i_bag]]).cuda()
        output = mil_model(bag_images, add_feat = bag_clinical_feat)
        loss += criterion(output, bag_label)/batch_size
        train_probas.append((output.item(), bag_label))
        train_predictions.append(int(output.item()>threshold))

        # Backpropagate once every batch_size iterations
        if (i_bag%batch_size) == batch_size - 1:
            train_loss_avg += batch_size * loss.item()/len(train_bags)
            loss.backward()
            final_opt.step()
            bwm_opt.step()
            encoder_opt.step()
            loss = 0

    balanced_accu = balanced_accuracy_score(train_bags_labels, train_predictions)
    f1 = f1_score(train_bags_labels, train_predictions)
    auc = roc_auc_score(train_bags_labels, train_predictions)
    print(f'Training : (loss : {train_loss_avg}, bal. acc. : {balanced_accu}, f1 : {f1}, auc : {auc})')

    P = [t[0] for t in train_probas]
    print('Mean probability:', np.mean(P), ', Std dev.:', np.std(P))

    val_loss_avg = 0
    val_predictions = []
    for i_bag in tqdm(range(len(val_bags))):
        bag_images = torch.Tensor(val_bags[i_bag]).cuda()
        bag_label = torch.Tensor([val_bags_labels[i_bag]]).cuda()
        bag_clinical_feat = torch.Tensor([val_bags_lymph_count[i_bag], val_bags_dobs[i_bag]]).cuda()
        output = mil_model(bag_images, add_feat = bag_clinical_feat)
        val_predictions.append(int(output.item()>threshold))
        loss = criterion(output, bag_label)
        val_loss_avg += loss.item()/len(val_bags)

    balanced_accu = balanced_accuracy_score(val_bags_labels, val_predictions)
    f1 = f1_score(val_bags_labels, val_predictions)
    auc = roc_auc_score(val_bags_labels, val_predictions)
    print(f'Validation : (loss : {val_loss_avg}, bal. acc. : {balanced_accu}, f1 : {f1}, auc : {auc})')

    if val_loss_avg<best_val_loss:
        best_val_loss = val_loss_avg
        best_bal_acc = balanced_accu
        best_f1 = f1
        torch.save(mil_model, save_model_path)

print({"Best validation loss": best_val_loss, "Best F1 score": best_f1, "Best bal. acc.": best_bal_acc})
