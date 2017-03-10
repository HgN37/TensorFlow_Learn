# Su dung thu vien sklearn
from sklearn import tree

# Cung cap "dac tinh" va "nhan"
features = [[100,100], [105,1], [150,0], [170,0],[12,11],[11,16],[111,123]]
labels = ["ee", "oo", "ee", "ee","eo","oe","oo"]

# Training
# tao classifier (chua co rule)
clf = tree.DecisionTreeClassifier()
#tao rule dua theo "feature" va "label" 
clf = clf.fit(features, labels)

# Du doan ket qua
print (clf.predict([[107,1]]))