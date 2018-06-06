import random
from scipy.spatial import distance 
from sklearn.metrics import accuracy_score

def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN : 

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):

        predictions = []
        
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i

        return self.y_train[best_index]
from sklearn import datasets 
iris = datasets.load_iris()

X = iris.data
y = iris.target 

from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


classifier = ScrappyKNN()

#this is the several machine learning algorithms if you want to test it just uncomment it one by one

#from sklearn import tree 
#classifier_1 = tree.DecisionTreeClassifier()

#from sklearn.neighbors import KNeighborsClassifier
#classifier_2 = KNeighborsClassifier()

#from sklearn.naive_bayes import GaussianNB
#classifier_3 = GaussianNB()

#classifier_1.fit(X_train, y_train)
#classifier_2.fit(X_train, y_train)
#classifier_3.fit(X_train, y_train)

#predictions_1 = classifier_1.predict(X_test)
#predictions_2 = classifier_2.predict(X_test)
#predictions_3 = classifier_3.predict(X_test)
#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, predictions_1))
#print(accuracy_score(y_test, predictions_2))
#print(accuracy_score(y_test, predictions_3))

classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(accuracy_score(y_test, predictions))
