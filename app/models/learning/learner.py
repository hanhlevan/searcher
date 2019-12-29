import pickle
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate

class MachineLearner:

    def __init__(self):
        self.clf = LinearSVC()

    def setData(self, X, y):
        self.X, self.y = X, y

    def setDataSeparate(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train, self.X_test, self.y_test = \
            X_train, y_train, X_test, y_test

    def train(self):
        self.clf.fit(self.X_train, self.y_train)
        self.save()

    def save(self):
        pickle.dump(self.clf, open(self.fileName, 'wb'))

    def load(self, modelFile):
        self.clf = pickle.load(open(modelFile, 'rb'))

    def loadVectorizer(self, vectorFile):
        self.vectorizer = pickle.load(open(vectorFile, "rb"))

    def test(self):
        print("testing")
        scores = self.clf.score(self.X_test, self.y_test)    
        print(scores)

    def crossValidate(self, cv=10):
        cv_results = cross_validate(self.clf, self.X, self.y, cv=cv)
        print(sorted(cv_results.keys()))
        print(cv_results)
    
    def run(self):
        self.load()
        # self.train()
        self.test()
    
    def predict(self, X):
        X = self.vectorizer.transform(X)
        return self.clf.predict(X)