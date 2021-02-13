#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:15:01 2021

@author: leonard
"""
from numpy import zeros
import ray
from time import time

from Parameters.parameter import NDIM
from Parameters.constants import DESNNGB, NORM_COEFF, BITS_FOR_POSITIONS, MAX_INT, \
    NCPU, Load
from utility.integer_coordinates import convert_to_int_position
from utility.utility import norm

###############################################################################
#parallel worker job

@ray.remote(num_cpus=1)
def drift(Particles, Problem, density_func):
    "this function does the heavy lifting for the parallelization"
    cnts_out = zeros(4, dtype = int)
    for particle in Particles:
        d = norm(particle.delta)
        d_mps = (NORM_COEFF/DESNNGB)**(1/NDIM) * particle.Hsml
        for i in range(cnts_out.shape[0]):
            if d > (0.1)**(i) * d_mps:
                cnts_out[i] += 1
        particle.position = particle.position + \
            convert_to_int_position(particle.delta, Problem.FacIntToCoord)
        keep_inside_box(particle, Problem.Periodic)
        #now update the model density value
        density_func(particle)
        #reset those variables to save memory
        particle.delta = zeros(NDIM)
        particle.Redistributed = 0
            
    return Particles, cnts_out

###############################################################################

def drift_particles(Particles, Problem, density_func):
    "In this function the particles are drifted according to their WVT forces"
    t0 = time()
    
    #split work evenly among processes
    result = [drift.remote(Particles[i * Load:(i+1) * Load], Problem, density_func) \
              for i in range(NCPU-1)]
    result.append(drift.remote(Particles[(NCPU-1) * Load:], Problem, density_func))
    
    Particles_new = list()
    cnts = zeros(4, dtype = int)
    while len(result):
        done_id, result = ray.wait(result)
        particles, cnts_out = ray.get(done_id[0])
        Particles_new += particles
        cnts += cnts_out
        
    t1 = time()
    Problem.Timer["WVT"] += t1-t0
    return Particles_new, cnts
        

def keep_inside_box(particle, Periodic):
    "Makes sure the particle is within the domain"
    for axis in range(NDIM):
        if Periodic[axis]:
            while particle.position[axis] < 0:
                particle.position[axis] += (1 << BITS_FOR_POSITIONS)
            while particle.position[axis] > MAX_INT:
                particle.position[axis] -= (1 << BITS_FOR_POSITIONS)
        else:
            while particle.position[axis] < 0 or \
                particle.position[axis] > MAX_INT:
                    if particle.position[axis] < 0:
                        if particle.position[axis] < -(1 << (BITS_FOR_POSITIONS-1)):
                            particle.position += (1 << BITS_FOR_POSITIONS)
                        else:
                            particle.position[axis] *= -1
                    else:
                        if particle.position[axis] > MAX_INT + (1 << (BITS_FOR_POSITIONS-1)):
                            particle.position[axis] -= (1 << BITS_FOR_POSITIONS)
                        else:
                            particle.position[axis] += 2 * (MAX_INT - \
                                                        particle.position[axis])
            
        