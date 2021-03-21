# LymphocytosisClassification

## How to reproduce the results ?
### Create a virtual environment
```
python3 -m venv venv
```
```
source venv/bin/activate
```
### Install requirements
```
pip install -r requirements.txt
```
### Train autoencoder
```
python train_autoencoder.py
```
### Train MIL model
```
python train_MIL_e2e.py
```
### Infer on test set using trained MIL model
```
python predict_MIL_e2e.py
```
### To get ROC curve
```
python get_roc_curve.py
```
