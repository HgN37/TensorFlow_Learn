from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Chia feature va label ra lam train va test
# 50% dung de train, 50% dung de test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

from sklearn import tree
classifier_one = tree.DecisionTreeClassifier()
classifier_one.fit(X_train, y_train)
predictions = classifier_one.predict(X_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))