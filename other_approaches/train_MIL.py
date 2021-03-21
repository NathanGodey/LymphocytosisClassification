from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from mil.models import AttentionDeepPoolingMil, MILES
from mil.utils.utils import get_samples_weight
from mil.utils.padding import Padding
from mil.metrics import AUC, BinaryAccuracy
from mil.validators import StratifiedKFold, LeavePOut
from mil.trainer.trainer import Trainer
from mil.preprocessing import StandarizerBagsList
import pickle
from dataloader import LymphocytesFlat
from torch.utils.data import DataLoader
from MIL import MILModel
import torch
import tensorflow as tf
#
# autoencoder_feature_extractor = torch.load('trained_models/autoencoder80.ckpt').encoder.cpu()
# dataset = LymphocytesFlat('trainset')
# mil_model = MILModel(autoencoder_feature_extractor, None, None)
# dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=3)
#
# bags, bags_labels, bags_ids = mil_model.make_bags(dataloader)
# pickle.dump(bags, open('bags.npy', 'wb'))
# pickle.dump(bags_labels, open('bags_labels.npy', 'wb'))


bags = pickle.load(open('bags.npy', 'rb'))
bags_labels = pickle.load(open('bags_labels.npy', 'rb'))

train_bags, val_bags, train_bags_labels, val_bags_labels = train_test_split(bags, bags_labels, test_size=0.2)

max_len_train = np.max([len(bag) for bag in train_bags])
max_len_val = np.max([len(bag) for bag in val_bags])

max_ = np.max([max_len_train, max_len_val])
trainer = Trainer()

metrics = [AUC, BinaryAccuracy]
model = AttentionDeepPoolingMil(gated=False, threshold=0.5, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
pipeline = [('padding', Padding(max_len=max_))]

trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)

valid = StratifiedKFold(n_splits=10, shuffle=True)

history = trainer.fit(train_bags, train_bags_labels, validation_strategy=valid, sample_weights='balanced',
                      verbose=1, model__epochs=50, model__batch_size=2, model__verbose=1)


print(trainer.predict_metrics(val_bags, val_bags_labels))
trainer.model.model.save_weights('trained_models/attention_deep_pooling_mil.h5')
