#!/usr/bin/python
'''
Created on Dec 3, 2013

@modified: michael.liu
@author: tvandrun
'''

from nltk.corpus import wordnet as wn
import sys
import os

# This program will give a set of possible parses, given a sentence.
# Thus each entry in the table that the CKY parser populates
# will have a set of results.


id_gen = 0


# A table record is a single result, indicating a way of
# parsing the range of tokens encompassed by the table entry 
# this record is in. 
# There are three kinds of parsing results, based on three kinds
# of productions:
# NT -> word (what I call a "last stop" non-terminal)
# NT -> a (a "unit" production, which technically violates CNF) 
# NT -> a b (a "dual" production)
# If this were Java, these would be three subclasses of TableRecord
class TableRecord :
    def __init__(self, NT, a=None, b=None, word=None) :
        self.NT = NT
        self.a = a
        self.b = b
        self.word = word
        global id_gen
        self.id = "x" + str(id_gen)
        id_gen += 1
        if a == None :
            assert word != None
            self.parses = ["(%s %s)" % (NT, word)]
            self.trees = [self.id + ' [style=bold, label = "' + self.NT + ':' + self.word + '"];'] 
        elif b == None :
            self.parses = ["(%s %s)" % (NT, p) for p in a.parses]
            self.trees = [self.id + ' [label = "' + self.NT + '"];\n'
                          + self.id + ' -> ' + self.a.id + ';\n'
                          + t
                          for t in a.trees]
        else :
            self.parses = ["(%s %s %s)" % (NT, p, q) for p in a.parses for q in b.parses]
            self.trees = [self.id + ' [label = "' + self.NT + '"];\n'
                          + self.id + ' -> ' + self.a.id + ';\n'
                          + self.id + ' -> ' + self.b.id + ';\n'
                          + t + s
                          for t in a.trees for s in b.trees]
   
    def __str__(self) :
        #return (self.NT, (self.ai, self.aj), (self.bi, self.bj)).__str__()
        return self.NT

    def __repr__(self) :
        return self.__str__()

# A table entry is a dynamic collection of table records
class TableEntry :
    def __init__(self) :
        self.records = []

    # Add a possible parse based on the production NT -> a b
    # where a and b are nonterminals
    def add_dual(self, NT, a, b):
        self.records.append(TableRecord(NT, a, b))

    # Add a possible parse based on the production NT -> a        
    # where a is a nonterminal
    def add_unit(self, NT, a):
        self.records.append(TableRecord(NT, a))
    
    # Add a possible parse based on the production NT -> token
    # where token is a terminal
    def add_term(self, NT, token):
        self.records.append(TableRecord(NT, word=token))

    # Retrieve all non-terminals that the range of tokens
    # can be parsed as. (Some nonterminals may be parsed
    # in more than one way, but this returns the nonterminals
    # as a set.)
    def all_NTs(self) :
        return set([r.NT for r in self.records])

    # Retrieve all records that are ways to parse
    # this range for a given non-terminal
    def records_for_NT(self, NT) :
        return [x for x in self.records if x.NT == NT]

    # Test if there is any way to parse this range of
    # tokens using a unit production at the top level
    def is_unit_here(self, NT) :
        for x in self.records :
            if x.NT == NT and x.b == None :
                return True
        return False

    # Retrieve all records
    def all_records(self) :
        return self.records

    # Modify this record by computing the closure of all unit 
    # productions applicable to this range of tokens.
    # For example, if the grammar contains the production a -> b
    # and this entry already indicates that this range can parse
    # to a b, then this range can also parse to an a.     
    def unit_closure(self) :
        changed = True
        while changed :
            changed = False
            for x in self.records :
                NT = x.NT
                if NT in units and not self.is_unit_here(units[NT]) :
                    self.add_unit(units[NT], x)
                    changed = True         


# Nonterminals in the grammar
nonterminals = ['Sentence', 'NounPhrase', 'AbsNP', 'ConcNP',
                 'That', 'CNPA', 'RelClause', 'Pronoun', 
                 'Det', 'Nominal', 'Adj', 'Noun', 'Adv',
                 'RelPronoun', 'PrepPhrase', 'Prep', 'VerbPhrase',
                 'VPA', 'VPB', 'Verb']

# Those nonterminals that are "last stop", that is, they are the
# righthand side of a production in the form of NT -> word
# where word is a terminal. The category "last stop nonterminals"
# also corresponds to the concept of "parts of speech".
# (I don't actually use this in my solution.)
last_stop_non_terminals = ['That', 'Det', 'Adj', 'Noun', 'Adv',
                           'Pronoun', 'RelPronoun', 'Prep', 'Verb']

# All "dual" productions in the grammar, that is, those in the form
# NT -> a b where a and b are nonterminals. This collection is
# represented as a map from lefthand sides of production to righthand
# sides. The production NT -> a b would be represented as (a, b):NT
duals = {}
duals[('NounPhrase', 'VerbPhrase')] = 'Sentence'
duals[('That', 'Sentence')] = 'AbsNP'
duals[('CNPA', 'RelClause')] = 'ConcNP'
duals[('CNPA', 'PrepPhrase')] = 'ConcNP'
duals[('Det', 'Nominal')] = 'CNPA'
duals[('Adj', 'Nominal')] = 'Nominal'
duals[('RelPronoun', 'VerbPhrase')] = 'RelClause'
duals[('Prep', 'NounPhrase')] = 'PrepPhrase'
duals[('Adv', 'VPA')] = 'VerbPhrase'
duals[('VPB', 'PrepPhrase')] = 'VPA'
duals[('Verb', 'Adj')] = 'VPB'
duals[('Verb', 'NounPhrase')] = 'VPB'

# All "unit" productions in the grammar, that is, those in the form
# NT -> a where a is a nonterminal As with duals, this is represented
# as a dict from lefthand sides to righthand sides.
units = {}
units['AbsNP'] = 'NounPhrase'
units['ConcNP'] = 'NounPhrase'
units['CNPA'] = 'ConcNP'
units['Pronoun'] = 'CNPA'
units['Noun'] = 'Nominal'
units['VPA'] = 'VerbPhrase'
units['VPB'] = 'VPA'
units['Verb'] = 'VPB'

# The list of prepositions
preps = [x.strip() for x in open("preps").readlines()]


# The vocabulary for relative pronouns, personal pronouns, determiners,
# and independent-clause-introducing "that". All other parts of
# speech (besides prepositions) are determined by WordNet.
vocab = {}

vocab['who'] = set(['RelPronoun'])
vocab['that'] = set(['RelPronoun', 'That'])
vocab['which'] = set(['RelPronoun'])
vocab['the'] = set(['Det'])
vocab['a'] = set(['Det'])
vocab['an'] = set(['Det'])
for x in ['I', 'me', 'he', 'him', 'she', 'her', 'it', 
          'they', 'them', 'you']:
    vocab[x] = set(['Pronoun'])

def wn_parser(word):
    type_set = set()
    if word in vocab.keys():
        type_set = type_set.union(vocab[word])
    if word in preps:
        type_set.add("Prep")
    if wn.synsets(word, pos=wn.VERB):
        type_set.add("Verb")
    if wn.synsets(word, pos=wn.NOUN):
        type_set.add("Noun")
    if wn.synsets(word, pos=wn.ADV):
        type_set.add("Adv")
    if wn.synsets(word, pos=wn.ADJ):
        type_set.add("Adj")
    return type_set



# Read the sentence from the commandline
sentence_text = sys.argv[1]
sentence_toks = sentence_text.split()
n = len(sentence_toks)

#sentence_toks_sets = [wn_parser(t) for t in sentence_toks]

# Make an empty parse table
parse_table = [[TableEntry() if i < j else None for j in range(n+1)]
               for i in range(n+1)]

# Populate the bottom row of the parse table,
# that is, parse each word individually
for j in range(n) :
    w = sentence_toks[j]
    nts = wn_parser(w)
    for nt in nts:
        #zero index problem
        parse_table[j][j+1].add_term(nt, w)
    parse_table[j][j+1].unit_closure()
    # TO DO

# Populate the rest of the table
# by going up the diagonals
for k in range(2, n+1) :   # k is the distances between indices
    for i in range(n+1-k) :   # i is the first index in the pair of indices
        j = i + k    # j is the other index
        #partition
        for par in range(i+1, j):
            #for ABC in nts
            #i is beginning index, j is ending index
            a_nts = parse_table[i][par].all_records()
            b_nts = parse_table[par][j].all_records()
            for a in a_nts:
                for b in b_nts:
                    if (a.NT, b.NT) in duals.keys():
                        parse_table[i][j].add_dual(duals[a.NT,b.NT], a, b)
                # if parse_table[i][par].is_unit_here(a) and parse_table[par][j].is_unit_here(b):
                #     parse_table[i][j].add_dual(duals[a,b], a, b)
        parse_table[i][j].unit_closure()
        # TO DO


# Print the results
i = 2
for x in parse_table[0][n].all_records() :
    for s in x.parses :
        print s
    for t in x.trees:
        file = open("tree" + str(i) + ".dot", 'w')
        file.write("digraph G {")
        file.write(t)
        file.write("}")
        file.close()
        os.system("dot -Tpng tree" + str(i) + ".dot -o tree" + str(i) + ".png")
        i += 1
    

                    




