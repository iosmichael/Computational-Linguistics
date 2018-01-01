'''
Created on Oct 5, 2015

@author: tvandrun
'''

import sys
import operator


costfile = open("costs", 'r')
# costfile should be formatted as
# ins, del, sub, flip, nop
costs = [int(line.strip()) for line in costfile]
costfile.close()
assert len(costs) == 5
# Return the cost of the minimum-cost mutation from
# source to target, giving up if the cost become greater
# thatn the given cut_off
def edit_distance(source, target, cut_off=sys.maxint):
    distances = [[0 for j in range(len(source) + 1)] for i in range(len(target) + 1)]
    # If we wanted to know *how* a word is transformed along
    # the least cost route to another word (ie, what the individual
    # mutations were), we would need an "actions" (or something like
    # that) array running parallel to distances.

    # Initialize the bottom row and left column
    for i in range(len(target) + 1) :
        distances[i][0] = costs[0] * i
        for j in range(len(source) + 1) :
            distances[0][j] = costs[1] * j
    
    # compute the components of the distances matrix.
    for i in range(1, len(target) + 1) :
        for j in range(1, len(source) + 1) :
            # this is for you to do
            cost_list = []
            # ins case
            cost_list.append(distances[i-1][j]+costs[0])
            # del case
            cost_list.append(distances[i][j-1]+costs[1])
            # sub case
            cost_list.append(distances[i-1][j-1]+costs[2])
            # transp case
            if source[j-2] == target[i-1] and target[i-2] == source[j-1]:
                cost_list.append(distances[i-2][j-2]+costs[3])
            # noop
            if source[j-1] == target[i-1]:
                cost_list.append(distances[i-1][j-1]+costs[4])
            distances[i][j] = min(cost_list) 

    return distances[len(target)][len(source)]


