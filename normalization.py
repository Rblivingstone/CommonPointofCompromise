# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:24:56 2018

@author: rbarnes
"""

from scipy.stats import percentileofscore, norm, boxcox
import numpy as np


def percentile_norm(data):
    mem_cache = {}
    out_list = []
    for i in data:
        if i not in mem_cache:
            mem_cache[i] = percentileofscore(data, i, kind='strict') / 100.0
        out_list.append(mem_cache[i])
    return out_list


def gaussian_quantile_norm(data, seed=None):
    if seed is not None:
        np.random.seed(seed)
    gauss = norm.rvs(size=len(data))
    gauss = np.sort(gauss)
    test = np.sort(data)
    mem_cache = {}
    out_list = []
    for i in range(len(data)):
        if test[i] not in mem_cache:
            mem_cache[test[i]] = gauss[i]
    for i in range(len(data)):
        out_list.append(mem_cache[data[i]])
    return out_list


def boxcox_norm(data):
    return(boxcox(data))
