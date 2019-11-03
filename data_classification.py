from sklearn import tree
from sklearn import linear_model
from sklearn import neighbors

features_X = [[170,76,35],[173,70,40],[169,69,50],[154,59,29],[150,48,26],[158,55,36],[160,68,56],[171,85,34]]
labels_Y = ["m", "m", "m", "f", "f", "f", "f", "m"]

classifer_tree = tree.DecisionTreeClassifier()
classifer_tree = classifer_tree.fit(features_X, labels_Y)

classifer_logistic = linear_model.LogisticRegression()
classifer_logistic = classifer_logistic.fit(features_X, labels_Y)

classifer_kneighbours = neighbors.KNeighborsClassifier()
classifer_kneighbours = classifer_kneighbours.fit(features_X, labels_Y)

print classifer_tree.predict([[165,70,28]])
print classifer_logistic.predict([[165,70,28]])
print classifer_kneighbours.predict([[165,70,28]])
