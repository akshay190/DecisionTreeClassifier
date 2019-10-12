import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('breast-cancer.txt', header=None)
data.columns = ["CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
               "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses",
               "CancerType"]

print(data.head())
print(data.describe())
data['BareNuclei'].value_counts()

data = data[data['BareNuclei'] != '?']

print(data.dtypes)

data['BareNuclei'] = pd.to_numeric(data['BareNuclei'])

print(data.head())

data.corr()['CancerType']
print(sns.heatmap(data.corr()))

X = data[["ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion","SingleEpithelialCellSize", 
         "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses"]]
y = data["CancerType"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=1)
clf_gini = DecisionTreeClassifier(criterion = 'gini', random_state = 1, max_depth =2, min_samples_leaf = 5)
clf_gini.fit(X_train, y_train)

predictions = clf_gini.predict(X_test)

print('test_accuracy:', accuracy_score(predictions, y_test))
print('training_accuracy:', accuracy_score(clf_gini.predict(X_train), y_train))


clf_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth=3, min_samples_leaf = 5)
clf_entropy.fit(X_train, y_train)
predictions = clf_entropy.predict(X_test)
print('test_accuracy:', accuracy_score(predictions, y_test))
print('training_accuracy:', accuracy_score(clf_entropy.predict(X_train), y_train))



clf_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth=5, min_samples_leaf = 5)
clf_entropy.fit(X_train, y_train)
predictions = clf_entropy.predict(X_test)
print('test_accuracy:', accuracy_score(predictions, y_test))
print('training_accuracy:', accuracy_score(clf_entropy.predict(X_train), y_train))
