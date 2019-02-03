# -*- coding: utf-8 -*-
'''
Created on 5 Mar 2018

@author: duygu
'''
import pickle
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn_crfsuite import metrics
from sklearn_crfsuite.utils import flatten
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import seaborn as sns
from nltk import pos_tag
from nltk import chunk
import logging

LOG_LEVEL = logging.INFO


class Tagger():
    '''
    base class for all taggers
    '''
    def __init__(self, params=None):
        '''
        initialize the tagger (and the vectorizer if necessary)
        params is a dict of tagger arguments that can be passed to models
        '''

        self.logger = None
        self.init_logging(LOG_LEVEL)

    def parser(self, file_name):

        tmpSentences = []
        tmpTags = []
        tmpData = []

        data, sentences, tags = [], [], []

        with open(file_name) as f:                      # get data [[]]
            for line in f:
                if(line != '\n'):
                    tmpData.append(line)
                else:
                    data.append(tmpData)
                    tmpData = []

        for sentence in data:
            for index in range(len(sentence)):
                line = str(sentence[index].split('\t'))
                tmpSentences.append(sentence[index].split('\t')[0])
                tmpTags.append((sentence[index].split('\t')[-1]).split('\n')[0])

            sentences.append(tmpSentences)
            tags.append(tmpTags)
            tmpSentences = []
            tmpTags = []
        f.close()
        return data, sentences, tags

    def mapFields(self, sentences, tags):
        tmpMapped = []
        mapped = []
        for sentence, tag in zip(sentences, tags):
            for i in range(len(sentence)):
                tup1 = (sentence[i], tag[i])
                tmpMapped.append(tup1)
            mapped.append(tmpMapped)
            tmpMapped = []
        return mapped

    def getFeatures(self, sentence, index):
        return{
            'word': sentence[index],
            'index': index if self.task != 'chunk' else 'None',
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'is_capitalized': sentence[index][0].upper() == sentence[index][0],
            'is_all_caps': sentence[index].upper() == sentence[index],
            'is_all_lower': sentence[index].lower() == sentence[index],
            'has_hyphen': '-' in sentence[index],
            'is_numeric': sentence[index].isdigit(),
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:],
            'prefix-1': sentence[index][0],
            'prefix-2': sentence[index][:2],
            'prefix-3': sentence[index][:3],
            'suffix-1': sentence[index][-1] if self.task != 'chunk' else 'None',
            'suffix-2': sentence[index][-2:],
            'suffix-3': sentence[index][-3:],
            'prev_word': '' if index == 0 else sentence[index - 1],
            'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
            'prev_tag': '' if index == 0 or self.method == 'lr' else pos_tag(sentence[index-1])[0][1],
            'next_tag': '' if index == len(sentence) - 1 or self.method == 'lr' else pos_tag(sentence[index+1])[0][1],
            'pos': pos_tag(sentence[index])[0][1] if self.task != 'pos' else 'None'
        }


    def getXY(self, sentences, mapped):

        X, y = [], []
        for sentence, mapped in zip(sentences,mapped):                                #mappingden feature dön
            if(self.method == 'lr'):
                for index in range(len(sentence)):
                    X.append(self.getFeatures(sentence, index))
                    y.append(mapped[index][-1])
            elif(self.method == 'crf'):
                X.append([self.getFeatures(sentence, index) for index in range(len(sentence))])
                y.append([tag for _, tag in mapped])
        return X,y

    def train(self, file_name):
        '''
        extract the features and train the model
        file_name shows the path to the file with task specific format
        '''
        self.data, self.sentences, self.tags = self.parser(file_name)          #get sentences, additional fields and tag
        self.mapped = self.mapFields(self.sentences, self.tags)                #sentence-field-tag mapping
        self.X_train, self.y_train = self.getXY(self.sentences, self.mapped)
        self.model.fit(self.X_train[:10000],self.y_train[:10000])

    def test(self, file_name, labels_to_remove=[]):
        '''
        test the model, extract features and predict tags
        return metrics, confusion matrix and tagged data in the order below
        precision
        recall
        f1
        accuracy
        confusion matrix
        tagged data
        file_name shows the path to the file with task specific format
        labels_to_remove show classes to ignore
            while calculating precision, recall and f1 scores
        '''
        testData, testSentences, testTags = self.parser(file_name)          #get sentences, additional fields and tag

        mappedTest = self.mapFields(testSentences, testTags)                #sentence-field-tag mapping


        labels = list(self.model.classes_)
        if self.task == 'ner':
            labels.remove('O')

        X_test, y_test = self.getXY(testSentences, mappedTest)

        y_pred = self.model.predict(X_test)

        if(self.method == 'crf'):
            y_test = flatten(y_test)
            y_pred = flatten(y_pred)


        precision = precision_score(y_test, y_pred, average = 'micro',labels = labels)
        recall = recall_score(y_test, y_pred, average = 'micro', labels = labels)
        f1 = f1_score(y_test, y_pred, average = 'micro', labels = labels)
        accuracy =  accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        tagged_sents = self.tag_sents(testSentences)


        #HEAT MAP

        """
        plt.figure(figsize=(9,9))
        sns.heatmap(confusion, annot=False, fmt=".0f", linewidths=.5, square = True, cmap = 'Reds_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
        plt.title(all_sample_title, size = 15);
        plt.show()
        """

        """
        #Learning Curve
        X1 = np.asarray(self.X_train)
        Y1 = np.asarray(self.y_train)

        X, y = X1, Y1


        title = "Learning Curves (Naive Bayes)"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

        estimator = GaussianNB()
        plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

        title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        # SVC is more expensive so we do a lower number of CV iterations:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        estimator = SVC(gamma=0.001)
        plt = plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

        plt.show()
        """

        return (precision, recall, f1, accuracy, confusion, tagged_sents)
    """
    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt
    """
    def crossValidation(self):


        kfold = model_selection.KFold(n_splits=10, random_state=7)
        results = model_selection.cross_val_score(self.model, self.X_train, self.y_train, cv=kfold, scoring='accuracy')
        accuracy = results.mean()

        return accuracy

    def evaluate(self, file_name, labels_to_remove=[]):
        '''
        return accuracy
        to be used for validation, learning curve, parameter optimization etc.
        '''
        if self.task == 'chunk' and self.method == 'lr':
            accuracy = self.crossValidation()
        else:
            accuracy = self.test(file_name, labels_to_remove)[3]

        return accuracy

    def tag(self, sentence):
        '''
        tag a single tokenized sentence
        sentence: [tkn1, tkn2, ... tknN]
        return the tagged sentence as list of fields
            as given in training/test files
        tagged sentence: [[tkn1, ..., tag1], [tkn2, ..., tag2], ...]
        put None for any field between token and tag that is not predicted
        '''
        X_tag = []
        taggedSent = []

        for index in range(len(sentence)):
            X_tag.append(self.getFeatures(sentence, index))


        if(self.method == 'lr'):
            y_pred = self.model.predict(X_tag)
        elif(self.method == 'crf'):
            y_pred = self.model.predict_single(X_tag)

        for i in range(len(sentence)):
            if self.task != 'ner':
                tmpList = [sentence[i], 'NONE', y_pred[i]]
            else:
                tmpList = [sentence[i], 'NONE', 'NONE', y_pred[i]]
            taggedSent.append(tmpList)

        return taggedSent


    def tag_sents(self, sentences):

        taggedSents = []

        for sentence in sentences:
            tmpList = self.tag(sentence)
            taggedSents.append(tmpList)

        return taggedSents

        '''
        tag a list of tokenized sentences
        sentences: [[tkn11, tkn12, ... tkn1N], [tkn21, tkn22, ... tkn2M] ...]
        return the tagged sentence as a list of fields
            as given in training/test files
        tagged sentences: [[[tkn11, ..., tag11], ...], ...]
        put None for any field between token and tag that is not predicted
        '''

    def save(self, file_name):
        '''
        save the trained models for tagger to file_name
        '''
        f = open(file_name, 'wb')
        pickle.dump(self.model,f)
        f.close()


    def load(self, file_name):
        '''
        load the trained models for tagger from file_name
        '''
        self.model = pickle.load(open(file_name, 'rb'))
        return self.model

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
