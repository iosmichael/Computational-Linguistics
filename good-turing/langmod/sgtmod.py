#!/usr/bin/python
'''
Created on Oct 10, 2017

@author: michael liu
'''
from __future__ import division
from langmod import *
from math import *

class SimpleGTModel:
    
    def __init__(self, lm_data):
        self.lm_data = lm_data
        self.ff = lm_data.count_of_counts
        self.fd = lm_data.fd_unigrams
        self.m, self.b = self.regression()

    def regression(self):
        ff = self.ff
        fd = self.fd
        x_ = sum([fd[key] for key in fd.keys()])/len(fd.keys())
        y_ = sum([ff[fd[key]] for key in fd.keys()])/len(fd.keys())
        m = sum([(fd[key]-x_)*(ff[fd[key]]-y_) for key in fd.keys()])/sum([fd[key]**2 for key in fd.keys()])
        b = y_ - m * x_
        return m, b

    def log_diff(self, r):
        if r == 0:
            return self.m * log10(1)
        return 10**(self.m * (log10(r+1)-log10(r)))

    def prob(self, w, freq = 100):
        c = self.fd[w]
        if c < freq:
            return (c+1)*self.ff[c+1]/(self.lm_data.N*self.ff[c])
        return (c+1)/self.lm_data.N * self.log_diff(c)

class KatzCutoffModel:
    
    def __init__(self, lm_data):
        self.lm_data = lm_data
        self.ff = lm_data.count_of_counts
        self.fd = lm_data.fd_unigrams

    def prob(self, k, w):
        ff = self.ff
        c = self.fd[w]
        if c == 0:
            return ff[1]/(self.lm_data.N * ff[0])
        elif c > k:
            return c/self.lm_data.N
        else:
            return ((c + 1) * ff[c+1] / ff[c] - c * (k + 1) * ff[k+1] / ff[1])/(self.lm_data.N * (1 - (c + 1) * ff[k+1] / ff[1]))