import torch
import pickle
from dataloader import LymphocytesFlat
from torch.utils.data import DataLoader
from utils import load_attention_mil
import numpy as np
from MIL import MILModel
import pandas as pd

device = torch.device('cpu')
dataset_train = LymphocytesFlat('trainset')
dataset_test = LymphocytesFlat('testset')

dataloader_train = DataLoader(dataset_train, batch_size=3, shuffle=True, num_workers=3)
dataloader_test = DataLoader(dataset_test, batch_size=3, shuffle=True, num_workers=3)

autoencoder_feature_extractor = torch.load('trained_models/autoencoder.ckpt', map_location=device).encoder
instance_predictor = None
bag_predictor = load_attention_mil('trained_models/attention_deep_pooling_mil_1.h5', 25, 198)

mil_model = MILModel(autoencoder_feature_extractor, None, bag_predictor)
prediction_train = mil_model.predict(dataloader_train)
prediction_test = mil_model.predict(dataloader_test)


df_train = pd.DataFrame(prediction_train, columns = ['ID', 'Proba'])
df_test = pd.DataFrame(prediction_test, columns = ['ID', 'Proba'])
df_all = df_train.append(df_test)
df_all['ID'] = df_all['ID'].apply(lambda x: f'P{int(x)}')
annotation_df = pd.read_csv('clinical_annotation.csv')
merged_df = annotation_df.merge(df_all, on='ID')
print(merged_df)
merged_df.to_csv('clinical_and_proba.csv', index= False)
