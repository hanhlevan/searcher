import nltk, editdistance, operator
from lib.datastructures.sentence import Sentence
import json

class WordCorrector:

    __LowerCode = "àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    __mapFile = "./resource/map.json"
    __dictionary  = "./resource/Viet74K.txt"

    def __init__(self):
        self.__loadMap()

    def __loadMap(self):
        self.tokens = dict()
        with open(self.__mapFile) as fdata:
            self.__mapChar = json.load(fdata)

    def setWords(self, words):
        """Could be list of raw words"""
        # self.words = words
        self.words = open(self.__dictionary).read().split()
        self.words = set([word.lower() for word in self.words])

    def __convertToTelex(self, word):
        result = ''
        for c in word:
            nextChar = c
            if c in self.__mapChar:
                nextChar = self.__mapChar[c]
            result += nextChar
        return result

    def __getAllTokens(self, word, delimiter="_"):
        tokens = dict()
        for token in word.split(delimiter):
            if token.isalnum():
                value = Sentence(token).remove_accents()
                keyName = self.__convertToTelex(token)
                tokens[keyName] = value
        return tokens

    def fit(self):
        for word in self.words:
            self.tokens.update(self.__getAllTokens(word))

    def save(self):
        self.tokens = sorted(self.tokens)
        with open(self.fileName, "w") as f:
            f.write(" ".join(self.tokens))
            f.close()

    def load(self, fileName=None):
        if fileName is None:
            fileName = self.fileName
        self.tokens = open(fileName).read().split()
        self.getIndex()
        print(len(self.tokens))

    def predict(self, utoken):
        nIntersect = len(set(utoken.lower()).intersection(set(self.__LowerCode)))
        if nIntersect > 0: return utoken
        if utoken in self.tokens: return self.tokens[utoken]
        if utoken in self.tokens.values(): return utoken
        if not utoken.isalnum(): return utoken
        results = dict()
        for token, value in self.tokens.items():
            results[value] = editdistance.eval(utoken, token)
        sortedDict = sorted(results.items(), key=operator.itemgetter(1))
        return sortedDict[0][0]
