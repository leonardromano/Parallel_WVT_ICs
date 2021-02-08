#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:33:16 2021

@author: leonard
"""
from numpy import sqrt
import ray
from time import time

from Parameters.constants import LARGE_NUM, NCPU, Load
from Parameters.parameter import Npart
from utility.utility import relative_density_error

###############################################################################
#Parallel functions
@ray.remote(num_cpus=1)
def get_errors(Particles):
    err_min = LARGE_NUM
    err_max = 0.
    err_mean = 0.
    err_sigma = 0.
    for particle in Particles:
        err        = relative_density_error(particle)
        err_min    = min(err, err_min)
        err_max    = max(err, err_max)
        err_mean  += err
        err_sigma += err * err
    return err_min, err_max, err_mean, err_sigma

###############################################################################


def compute_l1_error(Particles_ref, Problem):
    t0 = time()
    
    Particles = ray.get(Particles_ref)
    #split work evenly among processes
    pending = [get_errors.remote(Particles[i * Load: (i+1) * Load]) \
               for i in range(NCPU-1)]
    pending.append(get_errors.remote(Particles[(NCPU-1) * Load:]))
    
    #wait for all processes to finish
    err_min = LARGE_NUM
    err_max = 0.
    err_mean = 0.
    err_sigma = 0.
    while len(pending):
        done_id, pending = ray.wait(pending)
        mini, maxi, err, err2 = ray.get(done_id[0])
        err_min = min(err_min, mini)
        err_max = max(err_max, maxi)
        err_mean  += err
        err_sigma += err2
    err_mean /= Npart
    err_sigma = sqrt(err_sigma/Npart - err_mean * err_mean)
    
    t1 = time()
    Problem.Timer["L1-ERROR"] += t1-t0
    del Particles
    return err_min, err_max, err_mean, err_sigma