from lib.algorithms.matching import NGram, EditDistance
import lib.string.stringman as sman
import operator

class Ranker:

    coefficient = {
        "ngram" : 1,
        "cosine" : 1,
        "edit" : 1
    }

    def __init__(self):
        self.ngramMatcher = NGram()
        self.editMatcher = EditDistance()
    
    def setData(self, dataQuery, data, fields):
        self.__query = dataQuery["queries"][0]
        self.__extensions = dataQuery["extension"]
        self.__fields = fields
        self.__score = dict()
        self.__score["cosine"] = [item["score"] for item in data]
        self.__data = []
        self.__nszie = len(data)
        self.__ids = []
        self.__results = []
        for item in data:
            item = item["data"]
            dataItem = dict()
            for field in self.__fields:
                dataItem[field] = item[field]
            self.__data.append(dataItem)
            self.__ids.append({
                "_id" : item["_id"],
                "rootId" : item["rootId"]
            })

    def __scoring(self):
        ngramScore = []
        for i, item in enumerate(self.__data):
            score = 0
            for field, radix in self.__fields.items():
                score += radix * self.ngramMatcher.exec(self.__query, self.__extensions, item[field])
            ngramScore.append(score)
        self.__score["ngram"] = ngramScore
        editScore = []
        for i, item in enumerate(self.__data):
            score = 0
            for field, radix in self.__fields.items():
                score += radix * self.editMatcher.exec(self.__query, item[field])
            editScore.append(score)
        self.__score["edit"] = editScore

    def __ranking(self):
        score = []
        for i in range(self.__nszie):
            currentScore = 0
            for key in self.__score:
                currentScore += self.__score[key][i] * self.coefficient[key]
            score.append(currentScore)
        score = dict((x, y) for x, y in enumerate(score))
        self.__score = sorted(score.items(), key=operator.itemgetter(1))[::-1]
        for pos, score in self.__score:
            self.__results.append({
                "_id" : self.__ids[pos]["_id"],
                "rootId" : self.__ids[pos]["rootId"]
            })

    def __show(self):
        for pos, score in self.__score:
            print("%s|\t%f" % (sman.adjustString(self.__data[pos]["title"], 80), score))
            print("---")

    def getResult(self):
        self.__scoring()
        self.__ranking()
        self.__show()
        return self.__results
