# Tao mot classifier moi theo thuat toan KNN (K Nearest Neightbor)
# Mot sample se duoc tinh toan khoang cach toi cac diem training
# Sample se duoc xep cung loai voi training point gan nhat

# Dung de tinh khoang cach
# khoang cach c = sqrt ( (a1-b1)^2 + (a2-b2)^2 + (a3-b3)^2 + ....) 
from scipy.spatial import distance
def euc(a, b):
    return distance.euclidean(a, b)
    
# Classifier tu viet
# Mot classifier can ham "fit" de hoc va "predict" de du doan
class Test_KNN():
    # Ham fit de nhap du lieu training
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    # Moi sample trong X_test la mot row
    # tung row la mot sample    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    # Tinh khoang cach ngan nhat
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i 
        return self.y_train[best_index] # tra ve label dung nhat

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

# Chia feature va label ra lam train va test
# 50% dung de train, 50% dung de test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

my_classifier = Test_KNN() #TODO: make own classifier
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))