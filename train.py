from dataloader import Lymphocytes
import matplotlib.pyplot as plt
from model import CNN
import torch.optim as optim
from torch.utils.data import random_split
import torch.nn as nn
from tqdm import tqdm
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

dataset = Lymphocytes('trainset', patient_bs=20)
model = CNN(patient_bs=20).cuda()
print('Number of parameters: ', sum(p.numel() for p in model.parameters()))
train_size, val_size = int(0.75*len(dataset)), len(dataset) - int(0.75*len(dataset))
n_epochs = 200

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()
train_set, val_set = random_split(dataset, (train_size, val_size))

for epoch in range(n_epochs):
    print(f"Epoch {epoch}/{n_epochs}")
    train_loss = 0
    for iter, patient in tqdm(enumerate(train_set)):
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # f = r-a
        # print(f)
        optimizer.zero_grad()
        input = patient['images'].float().cuda()
        target = torch.Tensor([patient['annotations']['LABEL']]).cuda()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        train_loss+=loss
        optimizer.step()
    print('Training loss: ', train_loss.item()/train_size)

    val_loss = 0
    predictions = []
    gt = []
    for iter, patient in tqdm(enumerate(val_set)):
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # f = r-a
        # print(f)
        input = patient['images'].float().cuda()
        target = torch.Tensor([patient['annotations']['LABEL']]).cuda()
        output = model(input)
        loss = criterion(output, target)
        val_loss += loss

        predictions.append(int(output.item()>0.5))
        gt.append(int(target))
    bas = balanced_accuracy_score(gt, predictions)
    print('Validation Balanced Accuracy :',bas)
    print('Validation F1-Score :',f1_score(gt, predictions))
    print('Validation Loss :',val_loss.item()/val_size)
    torch.save(model, 'model.ckpt')
