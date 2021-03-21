from dataloader import LymphocytesFlat
import matplotlib.pyplot as plt
from autoencoder import AutoEncoder
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
import random
import os

save_model_path = 'trained_models/autoencoder60.ckpt'

# Retrieve patients in folder and make train and val sets
train_rate = 0.75
patients = os.listdir('trainset')
random.shuffle(patients)
pat_train_size = int(train_rate*len(patients))
pat_val_size = len(patients) - pat_train_size

train_patients = patients[:pat_train_size]
val_patients = patients[pat_train_size:]

# Create AutoEncoder model with specifier embedding dimension
model = AutoEncoder(emb_dim=60).cuda()
print('Number of parameters: ', sum(p.numel() for p in model.parameters()))

# Load train_set and val_set images and annotations
train_set = LymphocytesFlat('trainset', train_patients)
val_set = LymphocytesFlat('trainset', val_patients)
print(f'Train size: {len(train_set)}; Val size: {len(val_set)}')
train_loader = DataLoader(train_set, batch_size=5, shuffle=True, num_workers=3)
val_loader = DataLoader(val_set, batch_size=5, shuffle=True, num_workers=3)

n_epochs = 200
n_iters = 300
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

min_val_loss = 10000
for epoch in range(n_epochs):
    print(f"Epoch {epoch}/{n_epochs}")
    train_loss = 0
    t = tqdm(enumerate(train_loader))
    for iter, (img, pat_id, lab,_,_) in t:
        optimizer.zero_grad()
        input = img.float().cuda()
        output = model(input)
        loss = criterion(output, input)
        loss.backward()
        train_loss+=loss.item()
        t.set_description(f'Current loss : {train_loss/(iter+1)}')
        t.refresh()
        optimizer.step()
    print('Training loss: ', train_loss/len(train_set))

    val_loss = 0
    predictions = []
    gt = []
    for iter, (img, pat_id, lab,_,_) in tqdm(enumerate(val_loader)):
        input = img.float().cuda()
        output = model(input)
        loss = criterion(output, input)
        val_loss += loss.item()
    print('Validation Loss :',val_loss/len(val_set))
    if val_loss <= min_val_loss:
        min_val_loss = val_loss
        torch.save(model, save_model_path)
    else:
        # Print AutoEncoder output for last model
        batch = val_set[0][0].float()
        input = batch.unsqueeze(0).cuda()
        output = model(input)
        plt.imshow(input.cpu().permute(2,3,1,0).detach().numpy()[:,:,:,0])
        plt.show()
        plt.imshow(output.cpu().permute(2,3,1,0).detach().numpy()[:,:,:,0])
        plt.show()
        print("Early stopping")
        break

    # batch = val_set[0][0].float()
    # input = batch.unsqueeze(0).cuda()
    # output = model(input)
    # plt.imshow(input.cpu().permute(2,3,1,0).detach().numpy()[:,:,:,0])
    # plt.show()
    # plt.imshow(output.cpu().permute(2,3,1,0).detach().numpy()[:,:,:,0])
    # plt.show()
