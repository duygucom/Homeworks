# -*- coding:utf-8 -*-
'''
Created on Apr 12, 2018
@author: burak
'''

import logging
from sentence import Sentence

LOG_LEVEL = logging.INFO


class Data():
    '''
    Read data from a file
    '''

    def __init__(self, file_name, log_level=LOG_LEVEL):
        '''
        Constructor
        '''
        self.file_name = file_name
        self.sent = []
        self.logger = None
        self.init_logging(LOG_LEVEL)
        self.logger.info('processing' + self.file_name)
        self.read_graphs(file_name)

    def read_graphs(self, file_name):
        '''
        '''

        #sentence.transform ve read_graphı çağır
        
        with open(self.file_name, 'r') as in_file:
        	sent = []
        	for line in in_file:
        		if line.split() == '':
        			if sent == []:
        				break
        			self.data.append(sent)
        			sent = []
        		else:
        			sent.append(line.strip().split('\t'))
       	self.logger.info('read data')
       	self.sent = sent
       	self.save('data_mini/parsed_data')
       	#print(self.sent)


    def save(self, save_path):
        '''
        '''
        file = open(save_path,'w') 
		 
        for item in self.sent:
            file.write("%s\n" % item)
		 
        file.close() 


    def __repr__(self):
        '''
        '''

        return str(self.sent)
    
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

    deps = Data('data_mini/en-ud-dev.conllu')

    print(deps)
