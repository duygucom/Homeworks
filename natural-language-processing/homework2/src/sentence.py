# -*- coding:utf-8 -*-
'''
Created on Apr 12, 2018
@author: burak
'''

import logging

LOG_LEVEL = logging.INFO


class Sentence():
	'''
	Read a single sentence
	'''

	def __init__(self, sentence=None, type='list', log_level=LOG_LEVEL):
		'''
		type: list or str
		'''

		self.words = []
		self.tags = []			#second field
		self.head = []		#third field
		
		self.type = type

		self.logger = None
		self.init_logging(LOG_LEVEL)
		self.logger.info('processing init')
		
		if self.type == 'str':
			sentence = transform(sentence)

		self.sentence = sentence	#whole

		print(self.sentence)
		print()
		self.read_graph(sentence)

	def transform(self, sentence):

		sent = []

		a = sentence.strip().split('\n')

		for line in a:
			if line.split() == '\t':
				if sent == []:
					break
				sent = []
			else:
				sent.append(line.strip().split('\t'))
		return sent
			
	def read_graph(self, sentence):
		'''
		'''
		lengthSent = len(sentence)
		lengthField = len(sentence[0]) 

		for i in range(lengthSent):
			self.words.append(self.sentence[i][0])
			self.tags.append(self.sentence[i][1])
			if(lengthField == 4):
				self.head.append(self.sentence[i][2])

	
	def extract_features(self):
		'''
		add a dummy root node and extract featuers for all possible edges
		'''
		"""
		{
		'word': self.sentence[index]
		'address': index,
		'ctag': self.tag[index]
		'head': self.head[index]
		}
		"""




	def save(self, save_path):
		'''
		'''

	def __repr__(self):
		'''
		'''

	def init_logging(self, log_level):
		'''
		logging config and init
		'''
		if not self.logger:
			logging.basicConfig(
				format='%(asctime)s-|%(name)20s:%(funcName)12s|'
					   '-%(levelname)8s-> %(message)s')
			self.logger = logging.getLogger(self.__class__.__name__)
			self.logger.setLevel(log_level)


if __name__ == '__main__':

	sentence = str(
		'From\tIN \t3\tcase\n'
		'the\tDT\t3\tdet\n'
		'AP\tNNP\t4\tobl\n'
		'comes\tVBZ\t0\troot\n'
		'this\tDT\t6\tdet\n'
		'story\tNN\t4\tnsubj\n'
		':\t:\t4\tpunct\n'
	)

	Sentence(sentence, type='str')
	#print(deps)
	sentence = [
		['From', 'IN', '3', 'case'],
		['the', 'DT', '3', 'det'],
		['AP', 'NNP', '4', 'obl'],
		['comes', 'VBZ', '0', 'root'],
		['this', 'DT', '6', 'det'],
		['story', 'NN', '4', 'nsubj'],
		[':', ':', '4', 'punct']
	]

	Sentence(sentence, type='list')
	#print(deps)

	sentence = [
		['From', 'IN'],
		['the', 'DT'],
		['AP', 'NNP'],
		['comes', 'VBZ'],
		['this', 'DT'],
		['story', 'NN'],
		[':', ':']
	]

	deps = Sentence(sentence, type='list')
	print(deps)

