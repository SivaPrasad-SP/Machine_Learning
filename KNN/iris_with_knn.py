from knn import Knn

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X, y = iris.data, iris.target

print(type(iris), type(X), type(y), X.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

knn = Knn(k=5)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(f"predicted labels : {y_pred}")
print(f"Accuracy : {knn.accuracy(y_test, y_pred)}")