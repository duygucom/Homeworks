# -*- coding:utf-8 -*-
'''
Created on Apr 12, 2018
@author: burak
'''

import logging

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC

from data import Data
from sentence import Sentence
from mst import mst

LOG_LEVEL = logging.INFO


class Parser(object):
    '''
    docstring for Parser
    '''

    def __init__(self, params=None):
        '''
        initialize the parser models (and the vectorizers if necessary)
        params is a dict of arguments that can be passed to models
        '''
        self.vectorizer = DictVectorizer()
        self.model = SVC()

    def train(self, file_name):
        '''
        extract the features and train the models
        file_name shows the path to the training file with task specific format
        '''

    def _learn_edges(self, X, y):
        '''
        '''
        

    def _learn_labels(self, X, y):
        '''
        '''

    def test(self, in_file_name, out_file_name=None):
        '''
        test the model, extract features and parse the sentences
        file_name shows the path to the test file with task specific format
        return uas and las
        '''

    def parse_sentence(self, sentence):
        '''
        tag a single tokenized and pos tagged sentence
        sentence: [[tkn1, tag1], [tkn2, tag2], ...]
        return the parsed with the same data format
            4 tab separated fields per line
        '''

    def parse(self, sentences):
        '''
        tag a list of sentences
        return a list of parsed sentences with same data format
        '''

    def _mst(self, scores):
        '''
        '''

    def save(self, file_name):
        '''
        save the trained models to file_name
        '''

    def load(self, file_name):
        '''
        load the trained models from file_name
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

    parser = Parser()
    parser.train('data_mini/en-ud-train.conllu')
    uas, las = parser.test('data_mini/en-ud-dev.conllu')
    print(uas, las)

    sentence = [
        ['From', 'IN'],
        ['the', 'DT'],
        ['AP', 'NNP'],
        ['comes', 'VBZ'],
        ['this', 'DT'],
        ['story', 'NN'],
        [':', ':']
    ]
    print(parser.parse_sentence(sentence))

    parser.save('models/dep_parser_1.pickle')

    parser2 = Parser()
    parser2.load('models/dep_parser_1.pickle')
    print(parser2.parse_sentence(sentence))
