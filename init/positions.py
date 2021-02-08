#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:16:40 2021

@author: leonard
"""
from numpy import asarray
import ray
from time import time

from data.structures import particles
from Parameters.constants import NCPU, Load

def evaluate(particle, funcs):
    "evaluate the functions for the particle"
    for func in funcs:
        func(particle)

def sample(Particles_ref, Problem, funcs):
    "Sample the quantity specified by func"
    t0 = time()
    
    #split work evenly among processes
    actors = [particles.remote(Particles_ref, i, Load) for i in range(NCPU)]
    
    result = [actor.process.remote(evaluate, funcs) for actor in actors]
    
    Particles_new = list()
    while len(result):
        done_id, result = ray.wait(result)
        Particles_new += ray.get(done_id[0])
        
    t1 = time()
    Problem.Timer["INIT"] += t1-t0
    return ray.put(asarray(Particles_new))
    
        
        
    