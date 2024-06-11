import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from joblib import dump, load

df = pd.read_csv("./data/quest.csv", encoding='ISO-8859-1')

# Separando as colunas entre X e y
X = df.iloc[:, 0:60]
y = df[60]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs')
classifier.fit(X_train, y_train)

# Salvar o Modelo
dump(classifier, 'logistisc-regression.joblib')

# Chamar o Modelo
classifier = load('random_forest_model.joblib')


