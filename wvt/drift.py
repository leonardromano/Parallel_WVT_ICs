#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:15:01 2021

@author: leonard
"""
from numpy import zeros, asarray
import ray
from time import time

from Parameters.parameter import NDIM
from Parameters.constants import DESNNGB, NORM_COEFF, BITS_FOR_POSITIONS, MAX_INT, \
    NCPU, Load
from utility.integer_coordinates import convert_to_int_position
from utility.utility import norm

###############################################################################
#Parallel functions & classes

@ray.remote(num_cpus=1)
class particles_drift():
    def __init__(self, particles_ref, ID):
        if ID < NCPU-1:
            self.particles = particles_ref[ID * Load:(ID+1) * Load]
        else:
            self.particles = particles_ref[ID * Load:]
    
    def drift(self, Problem, density_func):
        "this function does the heavy lifting for the parallelization"
        cnts_out = zeros(4, dtype = int)
        for particle in self.particles:
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
            particle.delta = zeros(NDIM)
            particle.Redistributed = 0
            
        return [*self.particles], cnts_out

###############################################################################

def drift_particles(Particles_ref, Problem, density_func):
    "In this function the particles are drifted according to their WVT forces"
    t0 = time()
    
    #split work evenly among processes
    actors = [particles_drift.remote(Particles_ref, i) for i in range(NCPU)]
    
    result = [actor.drift.remote(Problem, density_func) for actor in actors]
    
    Particles = list()
    cnts = zeros(4, dtype = int)
    while len(result):
        done_id, result = ray.wait(result)
        particles, cnts_out = ray.get(done_id[0])
        Particles += particles
        cnts += cnts_out
        
    t1 = time()
    Problem.Timer["WVT"] += t1-t0
    return ray.put(asarray(Particles)), cnts
        

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
            
        