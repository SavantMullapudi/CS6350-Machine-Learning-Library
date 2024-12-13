from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

train_data = pd.read_csv("dataset/ml-2024-f/train_final.csv")
test_data = pd.read_csv("dataset/ml-2024-f/test_final.csv")

con_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
cat_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

X = train_data.drop(['income>50K'], axis=1)
y = train_data['income>50K']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_num = X_train[con_features]
X_train_cat = X_train[cat_features]
X_val_num = X_val[con_features]
X_val_cat = X_val[cat_features]

imputer_num = SimpleImputer(strategy='median')
X_train_num_imputed = imputer_num.fit_transform(X_train_num)
X_val_num_imputed = imputer_num.transform(X_val_num)

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num_imputed)
X_val_num_scaled = scaler.transform(X_val_num_imputed)

imputer_cat = SimpleImputer(strategy='most_frequent')
X_train_cat_imputed = imputer_cat.fit_transform(X_train_cat)
X_val_cat_imputed = imputer_cat.transform(X_val_cat)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat_encoded = encoder.fit_transform(X_train_cat_imputed)
X_val_cat_encoded = encoder.transform(X_val_cat_imputed)

X_train_preprocessed = np.hstack([X_train_num_scaled, X_train_cat_encoded])
X_val_preprocessed = np.hstack([X_val_num_scaled, X_val_cat_encoded])

classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train_preprocessed, y_train)

y_val_pred = classifier.predict(X_val_preprocessed)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

X_test_num = scaler.transform(imputer_num.transform(test_data[con_features]))
X_test_cat = encoder.transform(imputer_cat.transform(test_data[cat_features]))
X_test_preprocessed = np.hstack([X_test_num, X_test_cat])

y_test_proba = classifier.predict_proba(X_test_preprocessed)[:, 1]

submission_df = pd.DataFrame({
    'ID': test_data['ID'],  
    'Prediction': y_test_proba
})

submission_df.to_csv('submission_logistic_regression.csv', index=False)

