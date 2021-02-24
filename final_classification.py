import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

patient_df = pd.read_csv('clinical_and_proba.csv')
def birthdate_to_age(birthdate):
    try:
        month, day, year = birthdate.split('/')
    except:
        month, day, year = birthdate.split('-')
    month, day, year = int(month), int(day), int(year)
    return 365*year + 30*month + day

patient_df['DOB'] = patient_df['DOB'].apply(birthdate_to_age)

train_df = patient_df[patient_df['LABEL']>=0]
test_df = patient_df[patient_df['LABEL']<0]
ids = train_df['ID'].to_numpy()
X = train_df[['LYMPH_COUNT', 'DOB', 'Proba']].to_numpy()
y = train_df['LABEL'].to_numpy()

ids_test = test_df['ID'].to_numpy()
X_test = test_df[['LYMPH_COUNT', 'DOB', 'Proba']].to_numpy()
acc_mean = 0
for i in range(10):
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(X, y, ids, test_size=0.25, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes = [10,8,5], max_iter=1750).fit(X_train, y_train)
    acc = balanced_accuracy_score(y_val, clf.predict(X_val))
    acc_mean+=acc
    print(acc)
print('-->', acc_mean/10)
y_pred = clf.predict(X_test)
result_df = pd.DataFrame(zip(ids_test, y_pred), columns = ['ID', 'Predicted'])
# print(result_df)
result_df.to_csv('prediction_w_annot.csv', index=False)
