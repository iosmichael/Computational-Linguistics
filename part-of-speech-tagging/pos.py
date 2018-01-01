'''
Created on Aug 22, 2013

@author: tvandrun
'''
from __future__ import division

import math
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist

# 1. Load a (training) corpus.
# In the code below, the corpus will be
# referred to by variable all_text
corpus = PlaintextCorpusReader('.','baum-oo.txt')
all_text = nltk.Text(corpus.words())

# make the training text lowercase
all_text_lower = [x.lower() for x in all_text]
freq_dist = FreqDist(all_text_lower)

# make a reduced vocabulary
# In choosing a vocabulary size there is a trade-off.
# A larger vocabulary will in principle make for a more accurate
# tagger, but will be slower and will have a greater risk of underflow.
vocab_size = 500
vocab = sorted(freq_dist.keys(), key=lambda x : freq_dist[x], reverse=True)[:vocab_size]
# "***" is for all out-of-vocabulary types
vocab = vocab + ["***"]




# 2. Make a reduced form of the PennTB tagset
penntb_to_reduced = {}
# noun-like
for x in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'EX', 'WP'] :
    penntb_to_reduced[x] = 'N'
# verb-like
for x in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD', 'TO'] :
    penntb_to_reduced[x] = 'V'
# adjective-like
for x in ['POS', 'PRP$', 'WP$', 'JJ', 'JJR', 'JJS', 'DT', 'CD', 'PDT', 'WDT', 'LS']:
    penntb_to_reduced[x] = 'AJ'
# adverb-like
for x in ['RB', 'RBR', 'RBS', 'WRB', 'RP', 'IN', 'CC']:
    penntb_to_reduced[x] = 'AV'
# interjections
for x in ['FW', 'UH'] :
    penntb_to_reduced[x] = 'I'
# symbols
for x in ['SYM', '$', '#'] :
    penntb_to_reduced[x] = 'S'
# groupings
for x in ['\'\'', '(', ')', ',', ':', '``'] :
    penntb_to_reduced[x] = 'G'
# end-of-sentence symbols
penntb_to_reduced['.'] = 'E'

reduced_tags = ['N', 'V', 'AJ', 'AV', 'I', 'S', 'G', 'E']

# 3. tag the corpus
all_tagged = nltk.pos_tag(all_text)

# 4. make the probability matrices
# a tally from types to tags; a tally from tags to next tags
tag_word_tally = {y:{x:1 for x in vocab} for y in reduced_tags}
tag_transition_tally = {y:{x:1 for x in reduced_tags} for y in reduced_tags}


previous_tag = 'E' # "ending" will be the dummy initial tag

for (word, tag) in all_tagged :
    # fill this out:
        # For most tags, you want to convert it to the reduced tag
        # For most tags and words, update the tag transition tally
        # and the word-tag tally
        # But what if the tag is '-NONE-'?
        # What if the word is not in the vocabulary?
    word = word.lower()
    #exception handlers
    #if tag is -NONE-, NLTK exception, jump over the word
    if tag == '-NONE-':
        print "(%s, %s) cannot be tagged", word, tag
        continue 
    #if word is not in vocab
    if word not in vocab:
        word = '***'
    reduced_tag = penntb_to_reduced[tag]
    tag_word_tally[reduced_tag][word] += 1
    tag_transition_tally[previous_tag][reduced_tag] += 1
    previous_tag = reduced_tag

# now, make the actual transition probability matrices 
trans_probs = {}
for tg1 in reduced_tags :
    # fill this out:
        # For each tag tg1 compute the probabilities for transitioning to
        # each tag (say, tg2). Using relative frequency estimation,
        # that would mean dividing the number of times tg2 follows tg1 by
        # the absolute number of times t1 occurs. (But, what if tg1 never occurs..?)
        # Recommendation: think in terms of "for each tg2, how many times had
        # we transitioned from tg1?"
    count = []
    for tg2 in reduced_tags:
        count.append(tag_transition_tally[tg1][tg2])
    sum_tg1 = sum(count)
    trans_probs[tg1] = {}
    for tg2 in reduced_tags:
        trans_probs[tg1][tg2] = tag_transition_tally[tg1][tg2]/sum_tg1

# similarly for the emission (observation) probabilities

emit_probs = {}
for tg in reduced_tags:
    count = []
    for word in vocab:
        count.append(tag_word_tally[tg][word])
    sum_w = sum(count)
    emit_probs[tg] = {}
    for word in vocab:
        emit_probs[tg][word] = tag_word_tally[tg][word] / sum_w
    # fill this out:
    #     For each tag tg1 compute the probabilities for emitting each word v.
    #     Recommendation: think in terms of "for each word v, how many times
    #     did tg1 emit v?"


def test_case(tokens, tgs):
    count = 0
    mis_classified = []
    n_tgs = nltk.pos_tag(tokens)
    for i in range(1,len(tgs)):
        word, tg = n_tgs[i-1]
        n_word, n_tg = tgs[i]
        if n_tg == penntb_to_reduced[tg]:
            count += 1
        else:
            mis_classified.append(tgs[i])
    return count/len(tgs), mis_classified
# 5. implement Viterbi. 
# Write a function that takes a sequence of tokens,
# a matrix of transition probs, a matrix of emit probs,
# a vocabulary, a set of tags, and the starting tag

def pos_tagging(sequence, trans_probs, emit_probs, vocab, tags, start) :
    token = sequence[0]
    tagged = [start]
    mat = [0] * (len(sequence)+1)
    mat[0] = 1
    previous_tag = start
    for i in range(0, len(sequence)):
        probs = []
        word = sequence[i]
        if word not in vocab:
            word = '***'
        for tg in reduced_tags:
            probs.append((tg, mat[i] * trans_probs[previous_tag][tg]* emit_probs[tg][word.lower()]))
        sorted_tgs = sorted(probs, key = lambda tup: tup[1], reverse = True)
        n_tg, max_prob = sorted_tgs[0]
        # if n_tg == 'E':
        #     n_tg, _ = sorted_tgs[1]
        # argmax S_ P(O_ | S_, lambda)
        tagged.append((sequence[i], n_tg))
        mat[i+1] = max_prob
        previous_tag = n_tg
    return tagged

# 6. try it out: run the algorithm on the test data
tests = []
tests.append("Kaliko ran out as fast as his spindle legs could carry his fat, round body, and soon the Chief Counselor entered the cavern.")
tests.append("They did not tell their niece the sad news for several days, not wishing to make her unhappy; but one morning the little girl found Aunt Em softly crying while Uncle Henry tried to comfort her.")
tests.append("Dorothy laughed merrily at this speech, and then she became very sober again, for she could see how all this trouble was worrying her aunt and uncle, and knew that unless she found a way to help them their future lives would be quite miserable and unhappy.")
tests.append("Guph went to the King's private cave and sat down upon an amethyst chair and put his feet on the arm of the King's ruby throne.")

for test in tests:
    test_sample = nltk.word_tokenize(test)
    test_tagged = pos_tagging(test_sample, trans_probs, emit_probs, vocab, reduced_tags, 'E')            
    print test_tagged
    per, mis_classified = test_case(test_sample, test_tagged)
    print ""
    print "Accuracy: (%)",round(per * 100, 2)
    print "Miss classified: ", mis_classified
    print "=============="
    print ""
