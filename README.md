# LymphocytosisClassification

This project was the main assignment of the CentraleSupelec course named <i>Deep Learning for Medical Imaging</i>, led by Olivier Colliot and Maria Vakalopolou.
The report can be found <a href=https://github.com/NathanGodey/LymphocytosisClassification/blob/main/LymphocytosisClassification.pdf>here</a>.


## How to reproduce the results ? (using pretrained models)
### 1 - Create a virtual environment
```
python3 -m venv venv
```
```
source venv/bin/activate
```
### 2 - Install requirements
```
pip install -r requirements.txt
```
### 3 - Infer on test set using pretrained MIL model (find in trained_models)
```
python predict_MIL_e2e.py
```

## How to train the models ?
### 1 - Create a virtual environment
```
python3 -m venv venv
```
```
source venv/bin/activate
```
### 2 - Install requirements
```
pip install -r requirements.txt
```
### 3 - Train autoencoder
```
python train_autoencoder.py
```
### 4 - Train MIL model
```
python train_MIL_e2e.py
```
### 5 - Infer on test set using trained MIL model
```
python predict_MIL_e2e.py
```
### (optional) To get ROC curve
```
python get_roc_curve.py
```
