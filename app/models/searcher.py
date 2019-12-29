from app.models.config_reader import ConfigReader
from app.models.indexer import Indexer
from app.models.ranker import Ranker
from app.models.evaluator import Evaluator
from app.models.parser import QueryParser
from lib.database.connector import AccessDatabase
from pyvi import ViTokenizer
import time

class Searcher:

    def __init__(self, configFile):
        self.__configer = ConfigReader(configFile)
        self.__configer.parseFile()

        self.__parser = QueryParser(
            self.__configer.config["dbHost"], 
            self.__configer.config["dbPort"], 
            self.__configer.config["dbName"])
        self.__parser.loadModelCategory(
            self.__configer.config["vectorFile"], 
            self.__configer.config["learnerModelFile"])
        self.__parser.loadWordCorrect(self.__configer.config["wordCorrectFile"])
        self.__parser.loadSentenceCorrect(self.__configer.config["sentenceCorrectFile"])
        self.__parser.loadSynonym()
        
        self.__ranker = Ranker()
        self.__getFields()
        self.__accessor = AccessDatabase(self.__configer.config["dbHost"], 
            self.__configer.config["dbPort"], 
            self.__configer.config["dbName"])
        self.__retrievalMinScore = float(self.__configer.config["retrievalMinScore"])
        
        self.__indexer = Indexer(self.__accessor, self.__fields, self.__retrievalMinScore)

        self.__evaluator = Evaluator(self.__configer.config["dbHost"], 
            self.__configer.config["dbPort"], 
            self.__configer.config["dbName"])

    def __getFields(self):
        fields = self.__configer.config["fields"]
        self.__fields = dict()
        try:
            for item in fields.split(","):
                left, right = item.split(":")
                self.__fields[left] = float(right)
            return True
        except:
            print("Error when try parsing configuration file!")
            return False

    def run(self, raw_query, histories):
        print(histories);
        start = time.time()
        dataQuery = self.__parser.predict(raw_query, histories)
        firstQuery = dataQuery["queries"][0]
        rData = self.__indexer.retrieval(firstQuery, dataQuery["extension"], ["Người ngoài hành tinh", "1001 bí ẩn", "Ngày tận thế", "chinh phục sao Hỏa"])#dataQuery["categories"])
        self.__ranker.setData(dataQuery, rData, self.__fields)
        results = self.__ranker.getResult()
        timeSpent = time.time() - start
        dataQuery["categories"] = list(dataQuery["categories"])
        dataQuery["extension"] = list(dataQuery["extension"])
        return {
            "time" : timeSpent,
            "results" : results,
            "dataQuery" : dataQuery
        }
        
    def evaluate(self):
        valuationItems = self.__accessor.find("evaluation")
        queries = [item["query"] for item in valuationItems]
        listRetrievalIDs = []
        for query in queries:
            searchItems = self.run(query, [""])
            retrievalIDs = [item["_id"] for item in searchItems]
            listRetrievalIDs.append(retrievalIDs)
            self.__evaluator.setData(query, retrievalIDs)
            print("Processing at query '%s'" % query)
            evaluation = self.__evaluator.evaluate()
            for key, value in evaluation.items():
                print("%s : %.2f" % (key, value))
        print(self.__evaluator.MAP(queries, listRetrievalIDs))