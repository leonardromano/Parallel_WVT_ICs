#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:16:40 2021

@author: leonard
"""
from numpy.random import seed
import ray
from time import time

from Parameters.constants import NCPU

###############################################################################
#parallel worker class

@ray.remote(num_cpus=1)
class worker():
    def __init__(self,ID):
        #initialize output objects or random seeds for the worker
        seed(69 + 420 * ID)
    
    def process(self, Particles, funcs):
        "this function does the heavy lifting for the parallelization"
        for particle in Particles:
            evaluate(particle, funcs)
        return Particles
    
###############################################################################


def evaluate(particle, funcs):
    "evaluate the functions for the particle"
    for func in funcs:
        func(particle)

def sample(Particles, Problem, funcs):
    "Sample the quantity specified by func"
    t0 = time()
    
    #split work evenly among processes
    Load = Problem.Npart//NCPU
    actors = [worker.remote(i) for i in range(NCPU)]
    
    result = [actors[i].process.remote(Particles[i * Load:(i+1) * Load], funcs) \
              for i in range(NCPU-1)]
    result.append(actors[NCPU-1].process.remote(Particles[(NCPU-1) * Load:], funcs))
    
    Particles_new = list()
    while len(result):
        done_id, result = ray.wait(result)
        Particles_new += ray.get(done_id[0])
    
    t1 = time()
    Problem.Timer["INIT"] += t1-t0
    return Particles_new
    
        
        
    