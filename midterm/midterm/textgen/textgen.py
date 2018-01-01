'''
Created on Oct 26, 2017

@author: tvandrun
'''

################
#
# Usage:
#      python textgen.py [training text file]
#
# This will produce a few sentences (or one paragraph) of
# text "imitating" the style of the training text using
# the language models.
#
################


from nltk import FreqDist
import re
import math
import sys
import random

class LanguageModelData:
    
    def __init__(self, text):
        self.fd_unigrams = FreqDist(text)
        self.fd_bigrams = FreqDist([tuple(text[i:i+2]) for i in range(len(text) - 1)])
        self.fd_trigrams = FreqDist([tuple(text[i:i+3]) for i in range(len(text) - 2)])
        self.N = len(text)
        self.vocab = self.fd_unigrams.keys()

    def unigram_count(self, w):
        return float(self.fd_unigrams[w])
    
    def bigram_count(self, w1, w2):
        return float(self.fd_bigrams[(w1, w2)])
        
    def trigram_count(self, w1, w2, w3):
        return float(self.fd_trigrams[(w1, w2, w3)])

file_name = sys.argv[1]

raw_words = re.findall(r'[a-z][a-z\']*|\d+|[!$%*()\-:;\"\',.?]', 
file(file_name).read().lower())
alphabet = "abcdefghijklmnopqrstuvwxyz'"
def transform(w):
    if w.isdigit():
        return "NUM"    
    elif all(c in alphabet for c in w) :
        return w
    else :
        return  "PNCT"

lm_data = LanguageModelData([transform(w) for w in raw_words])        

class UnigramLanguageModel :
    
    def __init__(self, lm_data):
        self.lm_data = lm_data
        
    def p(self, w, h):
        return float(self.lm_data.fd_unigrams[w]) / self.lm_data.N
    
    def kind_of_model(self):
        return "unigram"

class BigramLanguageModel :
    
    def __init__(self, lm_data):
        self.lm_data = lm_data
        
    def p(self, w, h):
        try :
            if len(h) == 0 :
                return float(self.lm_data.fd_unigrams[w]) / self.lm_data.N
            else :
                return float(self.lm_data.fd_bigrams[(h[-1], w)]) / self.lm_data.fd_unigrams[h[-1]]
        except ZeroDivisionError :
            return 0.0

    def kind_of_model(self):
        return "bigram"

class TrigramLanguageModel :
   
    def __init__(self, lm_data):
        self.lm_data = lm_data
        
    def p(self, w, h):
        try :
            if len(h) == 0 :
                return float(self.lm_data.fd_unigrams[w]) / self.lm_data.N
            elif len(h) == 1 :
                return float(self.lm_data.fd_bigrams[(h[0], w)]) / self.lm_data.fd_unigrams[h[-1]]
            else :
                return float(self.lm_data.fd_trigrams[(h[-2], h[-1], w)]) / self.lm_data.fd_bigrams[(h[-2], h[-1])]
        except ZeroDivisionError :
            return 0.0

    def kind_of_model(self):
        return "trigram"


p_uni = UnigramLanguageModel(lm_data)
p_bi = BigramLanguageModel(lm_data)
p_tri = TrigramLanguageModel(lm_data)

#######################
#
#   Insert code here to use these models to generate
#   a paragraph (or a few sentences) of text.
#
#######################
w_length = 100
w_bank = lm_data.vocab
#remove unnecessary symbol
w_bank.remove("PNCT")
w_bank.remove("NUM")
w_bank.remove("'")
#find the word with max likelihood using the model
def find_max(model, h):
    w_r = [(model.p(w, h), w) for w in w_bank]
    w_r = sorted(w_r, key = lambda data: data[0],reverse=True)
    #little mutation that breaks the loop
    elem = w_r[random.randint(0,1)]
    return elem

## generate the first word with only unigram model
s = []
s.append(find_max(p_uni, [])[1])
## generate the second word with bigram model
prob, w = find_max(p_bi, s)
## if bigram model fails to find a promising candidate
if prob < 0.000001:
    s.append(find_max(p_uni, [])[1])
else:
    s.append(w)
## generate words afterward
for i in range(2, w_length):
    prob, w = find_max(p_tri, s)
    ## if trigram model fails to find a promising candidate, switch model to a bigram model
    if prob < 0.000001:
        prob_bi, w_bi = find_max(p_bi, s)
        ## if bigram model fails to find a promising candidate, switch model to a unigram model
        if prob_bi < 0.000001:
            s.append(find_max(p_uni, s)[1])
        else:
            s.append(w_bi)
    else:
        s.append(w)

print " ".join(s)
