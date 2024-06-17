import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


df = pd.read_csv("Projeto Final/data/quest.csv", encoding='ISO-8859-1')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

# salvar o modelo
dump(classifier, 'random_forest_model.joblib')

#chamar o modelo
classifier = load('random_forest_model.joblib')

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

precision = precision_score(y_test, y_pred, average='macro') 
print(precision)
f1 = f1_score(y_test, y_pred, average='macro') 
print(f1)
recall = recall_score(y_test, y_pred, average='macro')
print(recall)
