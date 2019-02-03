# -*- coding: utf-8 -*-
'''
Created on 5 Mar 2018

@author: duygu
'''

import pickle
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn_crfsuite import CRF
from sklearn_crfsuite.utils import flatten
from tagger import Tagger


class TaggerCRF(Tagger):
    '''
    '''
    def __init__(self):
        self.model = CRF()
        self.data = []
        self.sentences = []
        self.tags = []
        self.task = None
        self.field = None
        self.method = 'crf'
        self.features = None
        self.mapped = []
        self.X_train, self.y_train = [], []

    def setTask(self,task):
    	self.task = task

    def getTask(self):
    	return self.task

    def setField(self,field):
    	self.field = field

    def getField(self):
    	return self.field
