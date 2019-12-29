from collections import Counter
import numpy as np
import operator, time, threading

class Indexer:


    def __init__(self, dbAccessor, fields, minScore=0.5):
        self.db = dbAccessor
        self.minScore = minScore
        self.setFields(fields)
        self.__loadStopwords()
        self.__loadIndexes()

    def setFields(self, fields):
        self.fields = fields

    def retrieval(self, query, extension, categories):
        start = time.time()
        self.query = query
        self.categories = set(categories)
        self.__filterExtension(extension)
        print("###Simple with %.2f" % (time.time() - start))
        start = time.time()
        self.__retrievalDocs()
        print("###Retrieval documents with %.2f" % (time.time() - start))
        # self.__genVectorDocs() # Only use when change vector algorithm
        start = time.time()
        self.__genVectorQuery()
        print("###Generate vector query with %.2f" % (time.time() - start))
        start = time.time()
        self.__scoring()
        print("###Scoring query with %.2f" % (time.time() - start))
        return self.__results

    def __filterExtension(self, extension):
        # Remove stopwords
        self.extension = set()
        for word in set(extension):
            if word not in self.stopwords:
                self.extension.add(word)

    def __loadIndexes(self):
        data = self.db.find("index")
        self.indexes = dict()
        for index in data:
            keyGen = tuple([index["category"], index["field"], index["term"]])
            self.indexes[keyGen] = index

    def __loadStopwords(self):
        items = self.db.find("stopwords", {}, {"_id" : False, "term" : True})
        self.stopwords = set([item["term"] for item in items])

    def __getIndex(self, term, category, field):
        return self.db.find_one("index", {
            "term" : term,
            "category" : category, 
            "field" : field
        })

    def __TfIdf(self, tf, df):
        idf = 1 / df
        return tf * idf

    def __genVectorDocs(self):
        # Term Freq of docs
        cur, cnt = 1, len(self.docs)
        for doc in self.docs:
            percent = (cur / cnt) * 100
            print("Processing at (%d/%d - %.2f%%)" % (cur, cnt, percent))
            cur += 1

            category = doc["category"]
            vectorDoc = dict()
            for field in self.fields:
                vectorField = []
                for term, freq in doc["tFreq"][field]:
                    try:
                        df = self.__getIndex(term, category, field)["dFreq"]
                    except:
                        print("Not found: %s, %s, %s" % (term, category, field))
                        df = 1
                    vectorField.append([term, self.__TfIdf(tf=freq, df=df)])
                vectorDoc[field] = vectorField
            print(vectorDoc)
            self.db.update("prepost", {
                "_id" : doc["_id"]
            }, {
                "vectors" : vectorDoc
            })

    def __sim(self, category, field, vector, norm):
        if len(vector) == 0: return 0
        keyGen = tuple([category, field])
        if keyGen not in self.qNorm: return 0
        vectorQ = self.qVectors[keyGen]
        vector = dict((x, y) for x, y in vector)
        score = 0
        for item in vectorQ:
            try: score += vectorQ[item] * vector[item]
            except: pass
        return score / (self.qNorm[keyGen] * norm)

    def __similarity(self, data):
        score = 0
        category, vectors, norms = data["category"], data["vectors"], data["normL2"]
        for field, vector in vectors.items():
            score += self.fields[field] * self.__sim(category, field, vector, norms[field])
        return score

    def __scoring(self):
        self.__score = dict()
        start = time.time()
        # self.__scoringThreads()
        for _id, dataDoc in self.listDocs.items():
            currentScore = self.__similarity(dataDoc)
            if currentScore >= self.minScore:
                self.__score[_id] = currentScore
        print("Sim------------ %.2f" % (time.time() - start))
        start = time.time()
        sorted_score = sorted(self.__score.items(), key=operator.itemgetter(1))[::-1]
        print("Sorting-------- %.2f" % (time.time() - start))
        self.__results = []
        for _id, score in sorted_score:
            self.__results.append({
                "score" : score,
                "data" : self.listDocs[_id]
            })
            # print("%s ==> %s" % (_id, score))
            # print(self.listDocs[_id]["title"])
            # print("---")

    def __scoringThread(self, left, right):
        for _id in self.listKeys[left : right]:
            score = self.__similarity(self.listDocs[_id])
            if score >= self.minScore:
                self.__score[_id] = score

    def __scoringThreads(self, nThreads=None, nItems=400):
        self.listKeys = list(self.listDocs.keys())
        nsize = len(self.listKeys)
        if nThreads is None:
            nThreads = nsize // nItems + 1
        startIndexs = [0] * (nThreads + 1)
        segmentSize = nsize // nThreads
        for i in range(1, nThreads):
            startIndexs[i] = startIndexs[i - 1] + segmentSize
        startIndexs[nThreads] = len(self.listDocs)
        threads = []
        for i in range(nThreads):
            thread = threading.Thread(target=self.__scoringThread, args=(startIndexs[i],startIndexs[i + 1]))
            thread.start()
            threads.append(thread)
        print("Started %d threads to scoring!" % len(threads))
        for i in range(nThreads): threads[i].join()

    def __genVectorQuery(self):
        self.qVectors = dict()
        self.qNorm = dict()
        tFreq = dict(Counter(self.query.split()))
        for category in self.categories:
            for field in self.fields:
                vectorField = dict()
                for term, freq in tFreq.items():
                    try:
                        keyGen = tuple([category, field, term])
                        df = self.indexes[keyGen]["dFreq"]
                        vectorField[term] = self.__TfIdf(tf=freq, df=df)
                    except:
                        # print("Not found: '%s', '%s', '%s'" % (term, category, field))
                        pass
                keyGen = tuple([category, field])
                self.qVectors[keyGen] = vectorField
                values = list(vectorField.values())
                if len(values) > 0: self.qNorm [keyGen]= np.linalg.norm(np.array(values))
        # print(self.qVectors)
        # print(self.qNorm)

    def __retrievalDocs(self):
        self.listDocs = set()
        for term in self.extension:
            for category in self.categories:
                for field in self.fields:
                    keyGen = tuple([category, field, term])
                    try:
                        values = self.indexes[keyGen]["listIDs"]
                        self.listDocs.update(values)
                    except Exception as Error:
                        # print("Error: ", Error)
                        pass
        start = time.time()
        data = self.db.find("prepost", {"_id" : {"$in" : list(self.listDocs)}}, { "tFreq" : False, "biGrams" : False})
        print("-------------Find time: %.2f" % (time.time() - start))
        self.listDocs = dict()
        for item in data: self.listDocs[item["_id"]] = item
        print("Retrieval %d document(s)!" % len(self.listDocs))

    # def __retrievalDocs(self):
    #     self.listDocs = set()
    #     for term in self.extension:
    #         for category in self.categories:
    #             for field in self.fields:
    #                 keyGen = tuple([category, field, term])
    #                 try:
    #                     values = self.indexes[keyGen]["listIDs"]
    #                     self.listDocs.update(values)
    #                 except Exception as Error:
    #                     print(Error)
    #     start = time.time()
    #     self.listDocs = list(self.listDocs)
    #     self.__threadRetrieval()
    #     print("-------------Find time: %.2f" % (time.time() - start))
    #     self.listDocs = dict()
    #     for item in self.__data: self.listDocs[item["_id"]] = item
    #     print("Retrieval %d document(s)!" % len(self.listDocs))
        
    