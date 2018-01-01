'''
Created on Oct 20, 2015

@author: thomasvandrunen
'''

import sys
from editdist import edit_distance

source_word = sys.argv[1]
target_word = sys.argv[2]

print edit_distance(source_word, target_word, 25)