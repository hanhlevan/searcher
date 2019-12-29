from app.models.correct_word.corrector import WordCorrector
from lib.database.connector import AccessDatabase
from app.models.correct_sentence.add_accents import CorrectVietnameseSentence
from app.models.learning.learner import MachineLearner
from pyvi import ViTokenizer
import pickle

class QueryParser:

    defaultResult = {
        "queries" : [],
        "category" : "",
        "extension" : []
    }

    def __init__(self, dbHost, dbPort, dbName):
        self.accessor = AccessDatabase(dbHost, int(dbPort), dbName)
    
    def fit(self):
        # Word Corrector
        data = self.accessor.find("index", {}, {"term" : True, "_id" : False})
        words = [w["term"] for w in data]
        self.wordCorrect = WordCorrector()
        self.wordCorrect.setWords(words)
        self.wordCorrect.fit()
        # Sentence Corrector
        data = self.accessor.find("prepost", {}, {"title" : True, "_id" : False})
        data = [w["title"].replace("_", " ") for w in data]
        self.corrector = CorrectVietnameseSentence()
        self.corrector.fit(data)
        # Load Synonym
        self.loadSynonym()

    def loadSynonym(self):
        data = self.accessor.find("synonym", {}, {"term" : True, "words" : True, "_id" : False})
        self.synonym = dict()
        for item in data:
            self.synonym[item["term"]] = item["words"]
    
    def saveWordCorrect(self, fileName):
        pickle.dump(self.wordCorrect, open(fileName, 'wb'))
    
    def loadWordCorrect(self, fileName):
        self.wordCorrect = pickle.load(open(fileName, 'rb'))

    def saveSentenceCorrect(self, fileName):
        pickle.dump(self.corrector, open(fileName, 'wb'))

    def loadSentenceCorrect(self, fileName):
        self.corrector = pickle.load(open(fileName, 'rb'))

    def loadModelCategory(self, vectorFile, modelFile):
        # Load model
        self.machineLearner = MachineLearner()
        self.machineLearner.loadVectorizer(vectorFile)
        self.machineLearner.load(modelFile)

    def predict(self, raw_query, history=[""]):
        # Get Queries
        query = ViTokenizer.tokenize(raw_query).replace("_", " ")
        tokens = query.split()
        if len(tokens) == 0:  return self.defaultResults
        query = ' '.join([self.wordCorrect.predict(w) for w in tokens])
        queries = self.corrector.predict([query])[0]
        queries = [ViTokenizer.tokenize(word) for word in queries]
        # Get Categories
        firstQuery = queries[0]
        itemGuess = []
        for historyItem in history:
            itemGuess.append("%s %s" % (historyItem, firstQuery))
        categories = self.machineLearner.predict(itemGuess)
        # Get Synonym
        extensionQuery = "%s %s" % (firstQuery, raw_query)
        extension = set(extensionQuery.split())
        for word in firstQuery.split():
            if word in self.synonym:
                extension.update(self.synonym[word])
        return {
            "queries" : queries, 
            "categories" : categories,
            "extension" : extension
        }