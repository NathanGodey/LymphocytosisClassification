import numpy as np
from mil.models import AttentionDeepPoolingMil
from mil.utils.padding import Padding
from mil.metrics import AUC, BinaryAccuracy
from mil.validators import KFold
from mil.trainer.trainer import Trainer
from torch import nn
import torch.nn.functional as F
import torch

def load_attention_mil(path, dim_emb=25, pad_size=50):

    trainer = Trainer()

    metrics = [AUC, BinaryAccuracy]
    model = AttentionDeepPoolingMil(gated=False, threshold=0.5)
    pipeline = [('padding', Padding(max_len=pad_size))]

    trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)

    valid = KFold(n_splits=2, shuffle=True)

    #Initialize the trainer with dummy data before loading weigths
    history = trainer.fit([[[4]*dim_emb], [[5]*dim_emb]], [0,0], validation_strategy=valid, sample_weights='balanced',
                          verbose=0, model__epochs=1, model__batch_size=4, model__verbose=0)

    trainer.model.model.load_weights(path)
    print(trainer.predict([[[4]*dim_emb], [[5]*dim_emb]]))
    return trainer
