#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:47:28 2021

@author: leonard
"""
from numpy import sqrt

def volume(vector):
    dV = 1
    for dx in vector:
        dV *= dx
    return dV

def factorial(n, lowerb = 2):
    result = 1
    for i in range(lowerb, n+1):
        result *= i
    return result

def norm(vector):
    r = 0
    for component in vector:
        r += component * component
    return sqrt(r)