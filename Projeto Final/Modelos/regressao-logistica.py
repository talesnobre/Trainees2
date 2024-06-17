import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from joblib import dump, load
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


df = pd.read_csv("Projeto Final/data/quest.csv", encoding='ISO-8859-1')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')
classifier.fit(X_train, y_train)

dump(classifier, 'logistic-regression.joblib')

classifier = load('logistic-regression.joblib')

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

precision = precision_score(y_test, y_pred, average='macro') 
print(precision)

f1 = f1_score(y_test, y_pred, average='macro') 
print(f1)
