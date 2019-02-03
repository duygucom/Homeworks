# -*- coding: utf-8 -*-
'''
Created on 5 Mar 2018

@author: duygu
'''

import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from tagger import Tagger


class TaggerLR(Tagger):
    '''
    '''

    def __init__(self):
        self.model = Pipeline([
                            ('vectorizer', DictVectorizer(sparse=True)),
                            ('classifier', LogisticRegression())
                            ])
        self.data = []
        self.sentences = []
        self.tags = []
        self.task = None
        self.field = None
        self.method = 'lr'
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
