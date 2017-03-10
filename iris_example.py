# Iris nhan dang 3 loai hoa dua tren 4 thong so
# setosa/versicolor/virginica
# sepal length/width + petal length/width (cm)
# data duoc cung cap san trong sklearn
# sau khi training ta cung cap 4 thong so va Iris se doan ten

import numpy as np
# Import iris
from sklearn.datasets import load_iris
from sklearn import tree

# Nap Iris va lay mot vai mau ra lam "test"
iris = load_iris()
test = [0,50,100] # lay mot mau tu moi loai hoa 

# In ra cac thong so va cac ten hoa
print (iris.feature_names)
print (iris.target_names)

# In ra ten/thong so cua mau du lieu co san 
print (iris.data[5])
print (iris.target[5])

# Training data = Iris - test
train_target = np.delete(iris.target, test)
train_data = np.delete(iris.data, test, axis = 0)
# Testing data
test_target = iris.target[test]
test_data = iris.data[test]

# Training
clf = tree.DecisionTreeClassifier() 
clf = clf.fit(train_data, train_target)

# Checking
print (test_target)
print (clf.predict(test_data))

