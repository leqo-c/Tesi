#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:00:29 2018

@author: leo
"""

import itertools
import matplotlib.pyplot as plt

import os

def absoluteFilePaths(directory):
    res = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            res.append( os.path.abspath(os.path.join(dirpath, f)) )
    return res

def grid_plot(images, grid_size, size_of_figures=(10,4)):

    plt.figure(figsize=size_of_figures) 
    fig_dims = grid_size
    
    positions = itertools.product(range(grid_size[0]), range(grid_size[1]))
    
    i=0
    for col, pos in zip(images, positions) :
        plt.subplot2grid(fig_dims, pos)
        plt.imshow(images[i])
        plt.axis('off')
        i = i+1