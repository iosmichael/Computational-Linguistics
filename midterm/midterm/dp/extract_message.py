'''
Created on Oct 27, 2017

@author: tvandrun
'''

#####################
#
# Usage:
#    python extract_message.py msg1 msg2
#
#  where msg1 and msg2 are two multi-word messages delimited by
# quotation marks, for example:
#
#      python extract_message.py "CALL OFF THE ATTACK WAIT AT CURRENT LOCATION TIL DAWN" "ENEMY WILL NOT ATTACK AT POSITION BETWEEN DAWN AND NOON"
#
########################

import sys

msg1 = sys.argv[1].split(' ')
msg2 = sys.argv[2].split(' ')

# The next two lines demonstrats what msg1 and msg2 are.
# You may delete them when you write your solution

print msg1
print msg2

def longest_common_subsequence(a, b) :
	lcs = {(i,j):0 for j in range(0,len(b)+1) for i in range(0,len(a)+1)}
	lcs_s = {(i,j):[] for j in range(0,len(b)+1) for i in range(0,len(a)+1)}
	for i in range(1, len(a)+1):
		for j in range(1, len(b)+1):
			if a[i-1] == b[j-1]:
				lcs[i, j] = lcs[i-1, j-1] + 1
				lcs_s[i, j] = lcs_s[i-1, j-1] + [a[i-1]]
			else:
				if lcs[i, j-1] > lcs[i-1, j]:
					lcs[i, j] = lcs[i, j-1]
					lcs_s[i, j] = lcs_s[i, j-1]
				else:
					lcs[i, j] = lcs[i-1, j]
					lcs_s[i, j] = lcs_s[i-1, j]
	return lcs[len(a), len(b)], lcs_s[len(a), len(b)]

print longest_common_subsequence(msg1, msg2)
