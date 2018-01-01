'''
Created on Oct 5, 2015

@author: tvandrun
'''


import nltk
from nltk.corpus import PlaintextCorpusReader
import sys
from langmod import *
from editdist import edit_distance
import re

# when DEBUG is on, each word will be replaced with a list
# of the top five substitutions with their score; when off,
# each word will be replaced by the word with the best score
DEBUG = False

lang_model = InterpolatedLanguageModel([ConstantLanguageModel(), UnigramLanguageModel(lm_data)])

cap_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
all_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'"

# Given a word (possibly misspelled) and its history, return 
# the most likely word based on edit distance and language model.
# The strategy here doesn't work very well (even with a good language model).
# Your task is to improve on it
def correct_spell(word, history):
    # If it's not all letters and apostrophe, don't touch it
    if not all(c in all_alphabet for c in word) :
        return word

    # remember if the word was all caps or first caps
    all_cap = all(c in cap_alphabet for c in word)
    first_cap = word[0] in cap_alphabet

    # work in lowercase for now
    word = word.lower()
    
    ranks = []
    
    # look at all words in the vocabulary
    for candidate in vocab :
        # find their distance from the observed word
        dist = edit_distance(word, candidate, 10)
        # if a word's distance from the observed word is less than five,
        # then add that word and its score (calculated with a formula)
        # to a list of good candidates. Specifically the formula
        # is the probability assigned by the language model divided
        # by one plus some scaled version of the distance (so a higher
        # distance lowers the score). The "one plus" is so that if
        # the observed word is a real word, that word itself (with distance
        # 0) doesn't cause a division by 0 error.
        if dist < 5 :
            ranks.append((candidate, lang_model.p(candidate, history) / (1 + 10 * dist)))

    # sort the good ones by score
    ranks.sort(key=lambda x: x[1], reverse=True)
    
    # return the best
    if DEBUG :
        return "<" + ",".join([str(x) for x in ranks[:5]])
    else :
        result = ranks[0][0]
        if all_cap :
            result = result.upper()
        elif first_cap :
            result = result[0].upper() + result[1:]
        return result


source_file = open(sys.argv[1], 'r')

history = []

for line in source_file :
    tokens_raw = re.findall(r'[A-Za-z][A-Za-z\']*|\d+|[!$%*()\-:;\"\',.?]', line)
    
    corrected_line = ''
    for word in tokens_raw :
        corrected_word = correct_spell(word, history)
        corrected_line += corrected_word + ' '
        history = history[-4:] + [transform(corrected_word)]
        print "spell checking ->", word, "->", corrected_word
    print corrected_line

source_file.close()
