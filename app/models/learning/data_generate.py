import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import ClusterCentroids

class DataGenerator:

    fileName = "vectorizer.sav"

    def __init__(self, separation=True, testSize=0.33):
        self.separation = separation
        self.testSize = testSize

    def setData(self, data):
        self.data = data

    def groupByCategory(self):
        self.categoryData = dict()
        for doc in self.data:
            category = doc["category"]
            text = "%s %s" % (doc["title"], doc["content"])
            if category not in self.categoryData:
                self.categoryData[category] = []
            self.categoryData[category].append(text)

    def generateData(self):
        self.X, self.y = [], []
        for category, docs in self.categoryData.items():
            nsize = len(docs)
            self.X.extend(docs)
            self.y.extend([category] * nsize)
        # self.vectorizer = TfidfVectorizer()
        # self.X = self.vectorizer.fit_transform(self.X)
        # self.saveVectorizer()
        self.loadVectorizer()
        self.X = self.vectorizer.transform(self.X)

    def saveVectorizer(self):
        pickle.dump(self.vectorizer, open(self.fileName, 'wb'))

    def loadVectorizer(self):
        self.vectorizer = pickle.load(open(self.fileName, 'rb'))

    def transform(self, X):
        return self.vectorizer.transform(X)

    def balanceData(self):
        cc = ClusterCentroids(random_state=0)
        self.X, self.y = cc.fit_resample(self.X, self.y)

    def run(self):
        self.groupByCategory()
        self.generateData()
        self.balanceData()
        if self.separation:
            self.trainTestSplit()
    
    def trainTestSplit(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.testSize, random_state=42)