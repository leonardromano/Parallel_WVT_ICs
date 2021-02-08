#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:30:26 2021

@author: leonard
"""
from Parameters.constants import BITS_FOR_POSITIONS
from Parameters.parameter import NDIM
from numpy import zeros 

def convert_to_int_position(position, FacIntToCoord):
    """
    converts a coordinate with double accuracy to an integer position
    The left boundary corresponds to 0 and the right boundary to 2^32
    """
    intpos = zeros(NDIM, dtype = int)
    for i in range(NDIM):
        intpos[i] += int(position[i]/FacIntToCoord[i])
    return intpos
        
def get_distance_vector(x, y, NgbTree):
    "returns the minimum distance vector taking into account periodic boundaries"
    dx = zeros(NDIM)
    for i in range(NDIM):
        if NgbTree.Periodic[i]:
            if abs(x[i] - y[i]) < (1 << BITS_FOR_POSITIONS-1):
                dx[i] += x[i] - y[i]
            elif x[i] > y[i]:
                dx[i] += x[i] - y[i] - (1 << BITS_FOR_POSITIONS)
            else:
                dx[i] += x[i] - y[i] + (1 << BITS_FOR_POSITIONS)                    
        else:
            dx[i] += x[i] - y[i]
    return dx * NgbTree.FacIntToCoord