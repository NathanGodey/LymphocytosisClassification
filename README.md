# LymphocytosisClassification

## How to reproduce the results ?
### 1 - Infer using pretrained models
```
python predict_MIL_e2e.py
```
## OR
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
