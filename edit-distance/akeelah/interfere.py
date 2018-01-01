'''
Created on Oct 20, 2015

@author: thomasvandrunen
'''
import nltk
import sys
import random

alphabet = "abcdefghijklmnopqrstuvwxyz'"

# randomly mess up a word
def munge_word(word):
    # mess up a word one quarter of the time
    if random.random() < .25 :
        # pick a random position
        pos = int(random.random() * len(word))
        # pick a random letter
        letter = alphabet[int(random.random() * len(alphabet))]
        # substitute
        word = word[:pos] + letter + word[pos+1:]
    return word


# introduce random errors into a file

source_file = open(sys.argv[1], 'r')

for line in source_file :
    tokens_raw = nltk.word_tokenize(line)
    
    corrected_line = ''
    for word in tokens_raw :
        corrected_line += munge_word(word) + ' '
    print corrected_line
        
        
