################
#
#  Usage:
#        python test_recover_tag.py
#
# That will train the model in recover_tag.py on
# the file training-tagged.txt and then run the function
# recover_tag() from recover_tag.py on each message
# in test-tagged.txt
#
# This will print out each message in the test set (with
# one word still tagged), strip the # symbol from that
# message, call recover_tag() with the stripped message,
# and print the word (or None) return from recover_tag().
#
#################

import sys
from recover_tag import recover_tag 

def remove_hash(line) :
    sequence = []
    for w in line.split(' ') :
        if w[0] == '#' :
            sequence.append(w[1:])
        else :
            sequence.append(w)

    return sequence


source_file = open('test-tagged.txt', 'r')

for line in source_file :
    line = line.strip()
    if len(line) != 0 :
        line_reduced = remove_hash(line)
        print line
        print recover_tag(line_reduced)
        
    
