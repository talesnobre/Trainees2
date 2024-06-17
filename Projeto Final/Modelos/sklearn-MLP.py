import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.metrics import accuracy_score

df = pd.read_csv("final project/Trainees2/Projeto Final/data/quest.csv", encoding='ISO-8859-1')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=42)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

dump(classifier, 'sklearn-MLP.joblib')

classifier = load('sklearn-MLP.joblib')

accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')
