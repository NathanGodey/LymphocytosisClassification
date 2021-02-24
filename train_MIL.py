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
import dill

embeddings = pickle.load(open('embeddings.npy', 'rb'))
ids = pickle.load(open('ids.npy', 'rb'))
labels = pickle.load(open('labels.npy', 'rb'))
train_embeddings, val_embeddings, train_ids, val_ids, train_labels, val_labels = train_test_split(embeddings, ids, labels, test_size=0.25, stratify=labels)

bags = []
bags_labels = []
for id in np.unique(ids):
    indic = ids==id
    id_label = labels[indic][0]
    id_embeddings = embeddings[indic]
    bags.append(id_embeddings)
    bags_labels.append(id_label)

train_bags, val_bags, train_bags_labels, val_bags_labels = train_test_split(bags, bags_labels, test_size=0.2)

max_len_train = np.max([len(bag) for bag in train_bags])
max_len_val = np.max([len(bag) for bag in val_bags])

max_ = np.max([max_len_train, max_len_val])
trainer = Trainer()

metrics = [AUC, BinaryAccuracy]
model = AttentionDeepPoolingMil(gated=True, threshold=0.5)
pipeline = [('padding', Padding(max_len=max_))]

trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)

valid = StratifiedKFold(n_splits=5, shuffle=True)

history = trainer.fit(train_bags, train_bags_labels, validation_strategy=valid, sample_weights='balanced',
                      verbose=1, model__epochs=20, model__batch_size=3, model__verbose=0)

trainer.model.model.save_weights('trained_models/attention_deep_pooling_mil_1.h5')
print(trainer.predict_metrics(val_bags, val_bags_labels))
