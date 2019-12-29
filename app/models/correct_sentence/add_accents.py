import pickle
from app.models.correct_sentence.hmm import HiddenMarkovModel

class CorrectVietnameseSentence():
	
	def __init__(self, list_ngrams=[2, 3, 4], eta=0.000001):
		self.list_ngrams = list_ngrams
		self.eta = eta
		self.model = HiddenMarkovModel(self.list_ngrams, self.eta)
	
	def fit(self, data):
		self.model.setData(data)
		self.model.fit()
	
	def predict(self, testcase, lim_per_index=[5], output_size=1):
		return self.model.fast_predict(testcase, lim_per_index, output_size)
	
	def score(self, inp_list, label_list, list_of_indices=[5]):
		return self.model.score(inp_list, label_list, list_of_indices)