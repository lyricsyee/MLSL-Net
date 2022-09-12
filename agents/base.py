'''
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
'''

import logging

class BaseAgent:
    '''
    This base class will contain the base functions to be overloaded.
    '''

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def load_checkpoint(self, file_name):
        raise NotImplementedError


    def save_checkpoint(self, file_name='checkpoint.pth.tar', is_best=0):
        raise NotImplementedError


    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


    def train_one_epoch(self):
        raise NotImplementedError


    def validate(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError



