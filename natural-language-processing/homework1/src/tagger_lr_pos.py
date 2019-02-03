# -*- coding: utf-8 -*-
'''
Created on 5 Mar 2018

@author: duygu
'''

from tagger_lr import TaggerLR


class TaggerLR_POS(TaggerLR):
    '''
    '''
    def __init__(self):
        TaggerLR.__init__(self)
        TaggerLR.setTask(self, 'pos')
        TaggerLR.setField(self, 3)



if __name__ == '__main__':

    data_dir = 'data_mini/pos/'
    #data_dir = 'data/pos/'
    tagger = TaggerLR_POS()

    tagger.train(data_dir + 'en-ud-train.conllu')


    print('eval acc:', tagger.evaluate(data_dir + 'en-ud-dev.conllu'))

    precision, recall, f1, accuracy, confusion, tagged_sents = \
        tagger.test(data_dir + 'en-ud-test.conllu')

    print('test pre:', precision)
    print('test rec:', recall)
    print('test f1: ', f1)
    print('test acc:', accuracy)
    print('test con:', confusion)
    print()

    #tagger.save('models/tagger_lr_pos.pickle')
    """
    del tagger
    try:
        print(tagger)
    except Exception as e:
        print(e)
    """
    '''
    I   PRON    PRP
    do  AUX VBP
    n't PART    RB
    think   VERB    VB
    it  PRON    PRP
    matters VERB    VBZ

    Gets    VERB    VBZ
    the DET DT
    Job NOUN    NN
    Done    ADJ JJ
    '''
    """
    tagger = TaggerLR_POS()
    tagger.load('models/tagger_lr_pos.pickle')
    print()
    print(tagger.tag(['I', 'do', 'n\'t', 'think', 'it', 'matters']))
    print()
    print(tagger.tag_sents([
        ['I', 'do', 'n\'t', 'think', 'it', 'matters'],
        ['Gets', 'the', 'Job', 'Done']]))
    """
