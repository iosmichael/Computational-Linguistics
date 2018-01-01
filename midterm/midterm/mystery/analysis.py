import nltk
from nltk import FreqDist

a_file = open('a', 'r')
words = nltk.word_tokenize(a_file.read())
fds = FreqDist(words)
print "a:", fds.most_common(50)

print ""

b_file = open('b', 'r')
words = nltk.word_tokenize(b_file.read())
fds = FreqDist(words)
print "b:", fds.most_common(50)
