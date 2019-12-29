from lib.database.connector import AccessDatabase
from statistics import mean 

class Evaluator:

    __colName = "evaluation"

    def __init__(self, dbHost, dbPort, dbName):
        self.__accessor = AccessDatabase(dbHost, dbPort, dbName)

    def setData(self, query, listRetrievals):
        self.__query = query
        self.__listRetrievals = listRetrievals
        item = self.__accessor.find_one(self.__colName, {
            "query" : query
        })
        if item is None: return False
        self.__listIDs = item["listIDs"]
        return True

    def __precision(self):
        if len(self.__listRetrievals) == 0: return 0
        intersectSize = len(set(self.__listRetrievals).intersection(set(self.__listIDs)))
        return intersectSize / len(self.__listRetrievals)

    def __recall(self):
        intersectSize = len(set(self.__listRetrievals).intersection(set(self.__listIDs)))
        return intersectSize / len(self.__listIDs)

    def __f1_score(self, precision, recall):
        if (precision == 0) and (recall == 0): return 0
        return (2 * precision * recall) / (precision + recall)

    def evaluate(self):
        precision = self.__precision()
        recall = self.__recall()
        f1_score = self.__f1_score(precision, recall)
        return {
            "precision" : precision,
            "recall" : recall,
            "f1-score" : f1_score
        }

    def precisionAtK(self, k=10):
        if len(self.__listRetrievals) == 0: return 0
        if k == 0: return None
        intersectSize = len(set(self.__listRetrievals[:k]).intersection(set(self.__listIDs)))
        return intersectSize / k

    def AP(self):
        if len(self.__listRetrievals) == 0: return 0
        results = 0
        for k in range(1, len(self.__listRetrievals) + 1):
            results += self.precisionAtK(k)
        return results / len(self.__listRetrievals)

    def MAP(self, multiQueries, listRetrievals):
        if len(listRetrievals) == 0: return 0
        results = []
        for query, retrievals in zip(multiQueries, listRetrievals):
            self.setData(query, retrievals)
            results.append(self.AP())
        return mean(results)
