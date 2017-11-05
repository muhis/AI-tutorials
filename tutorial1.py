from sklearn import datasets, tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)
FlowerClassifier = tree.DecisionTreeClassifier()
FlowerClassifier.fit(x_train, y_train)
predictions = FlowerClassifier.predict(x_test)
print(accuracy_score(y_test, predictions))