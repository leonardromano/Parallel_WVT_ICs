#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:25:13 2021

@author: leonard
"""
from Parameters.parameter import NDIM
from Parameters.constants import NORM_FAC, DESNNGB

if NDIM == 1:
    def kernel(x, h, derivative = False):
        if derivative:
            if x < 1:
                return -12 * NORM_FAC * (1-x)**2 * x / h**2
            else:
                return 0
        else:
            if x < 1:
                return NORM_FAC * (1-x)**3 * (1 + 3 * x) /h
            else:
                return 0
else:
    def kernel(x, h, derivative = False):
        if derivative:
            if x < 1:
                return -20 * NORM_FAC * (1-x)**3 * x * h**(-(NDIM+1))
            else:
                return 0
        else:
            if x < 1:
                return NORM_FAC * (1-x)**4 * (1 + 4 * x) * h**(-NDIM)
            else:
                return 0 

def wendland_bias_correction(particle):
    "correct for density bias according to Dehn & Aly (2012)"
    epsilon = 0
    if NDIM == 3:
        epsilon = 0.0294 * (DESNNGB * 0.01)**(-0.977)
    elif NDIM == 2:
        epsilon = 0.340 * (DESNNGB/16)**(-0.81)
        
    if epsilon != 0:
        particle.Rho -= epsilon * NORM_FAC * particle.Hsml**(-NDIM)