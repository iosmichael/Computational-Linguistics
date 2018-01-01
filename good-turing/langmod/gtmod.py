from sgtmod import *
'''
Created on Oct 10, 2017

@author: michael liu
'''
class UnigramGTModel :
    def __init__(self, lm_data):
        self.lm_data = lm_data
        self.gts = SimpleGTModel(lm_data)
        
    def p(self, w, h):
        return self.gts.prob(w)
    
    def kind_of_model(self):
        return "unigram with good turing smoothing"

class BigramGTModel:
    def __init__(self, lm_data):
        self.lm_data = lm_data
        self.gts = GTModel(lm_data)
        
    def p(self, w, h):
        return 0
    
    def kind_of_model(self):
        return "bigram with good turing smoothing"

class TrigramGTModel:

    def __init__(self, lm_data):
        self.lm_data = lm_data
        self.gts = GTModel(lm_data)
        
    def p(self, w, h):
        return 0

    def kind_of_model(self):
        return "trigram with good turing smoothing"

'''
GT-Katz Cutoff Model
'''
class UnigramCutoffModel :
    def __init__(self, lm_data):
        self.lm_data = lm_data
        self.gts = KatzCutoffModel(lm_data)
        
    def p(self, w, h):
        return self.gts.prob(5, w)
    
    def kind_of_model(self):
        return "unigram with cutoff model"