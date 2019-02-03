# -*- coding: utf-8 -*-
'''
Created on 5 Mar 2018

@author: duygu
'''

from tagger_lr import TaggerLR


class TaggerLR_NER(TaggerLR):
    '''
    '''
    def __init__(self):
        TaggerLR.__init__(self)
        TaggerLR.setTask(self, 'pos')
        TaggerLR.setField(self, 4)



if __name__ == '__main__':

    #data_dir = 'data_mini/ner/'
    data_dir = 'data/ner/'

    tagger = TaggerLR_NER()

    tagger.train(data_dir + 'eng.train.txt')
    """
    print('eval acc:', tagger.evaluate(data_dir + 'eng.testa.txt'))


    precision, recall, f1, accuracy, confusion, tagged_sents = \
        tagger.test(data_dir + 'eng.testb.txt')

    print('test pre:', precision)
    print('test rec:', recall)
    print('test f1: ', f1)
    print('test acc:', accuracy)
    print('test con:', confusion)
    print()
    """

    tagger.save('models/tagger_lr_ner.pickle')
    """
    del tagger
    try:
        print(tagger)
    except Exception as e:
        print(e)
    """
    '''
    --    :    B-NP    O
    Brussels    NNP    I-NP    B-ORG
    Newsroom    NNP    I-NP    I-ORG
    32    CD    I-NP    O
    2    CD    I-NP    O
    287    CD    I-NP    O
    6800    CD    I-NP    O

    There    EX    B-NP    O
    was    VBD    B-VP    O
    no    DT    B-NP    O
    Bundesbank    NNP    I-NP    B-ORG
    intervention    NN    I-NP    O
    .    .    O    O
    '''
    """
    tagger = TaggerLR_NER()
    tagger.load('models/tagger_lr_ner.pickle')
    print()
    print(tagger.tag(['--', 'Brussels', 'Newsroom', '32', '2', '287', '6800']))
    print()
    print(tagger.tag_sents([
        ['--', 'Brussels', 'Newsroom', '32', '2', '287', '6800'],
        ['There', 'was', 'no', 'Bundesbank', 'intervention', '.']]))
    """
