import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


data = pd.read_csv("data2.csv")
X = data.iloc[:,:9]
y = data.iloc[:, 9]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
rf = RandomForestClassifier(n_estimators=100, random_state = 22)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

joblib.dump(rf, "rf_model2.joblib")