#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:39:15 2021

@author: leonard
"""
#public libraries
from numpy import log, zeros
import ray
from time import time

#custom libraries
from Parameters.parameter import NDIM
from Parameters.constants import NCPU, Load
from tree.treewalk import get_minimum_distance_from_wall, add_ghost
from utility.integer_coordinates import get_distance_vector
from utility.utility import relative_density_error, norm, sign

###############################################################################
#parallel worker job

@ray.remote(num_cpus=1)
def process(Particles, NgbTree, step):
    "this function does the heavy lifting for the parallelization"
    for particle in Particles:
        compute_force(particle, NgbTree, step)
    return Particles

###############################################################################
#per particle processing
    
def compute_force(particle, NgbTree, step):
    particle.delta = zeros(NDIM)
    err = relative_density_error(particle)
    delta_fac = err/(1 + err)
    
    #prepare boundary treatment
    if particle.CloseToWall:
        min_dist_from_wall = get_minimum_distance_from_wall(particle, NgbTree)
        
    #now add contributions for all neighbors
    for no in particle.neighbors:
        ngb = NgbTree.Tp[no]
        if particle.ID == ngb.ID:
            continue
        
        dist = get_distance_vector(particle.position, ngb.position, NgbTree)
        if particle.CloseToWall and add_ghost(particle, ngb, dist, min_dist_from_wall, NgbTree):
            #particle and ghost contributions cancel out
            continue
        r  = norm(dist)
        h  = 0.5 * (particle.Hsml + ngb.Hsml)
        sgn = sign(ngb)
        if NDIM == 1:
            wk = log(r/h + 1e-3)
        else:
            wk = (r/h + 1e-3)**(-(NDIM-1))
        particle.delta += sgn * h * wk * dist / r
    #reduce stepsize for particles with small error
    particle.delta *= step * delta_fac
    #reduce memory load
    particle.neighbors = list()

###############################################################################

def compute_wvt_forces(Particles, Problem, NgbTree_ref, step):
    "In this function the WVT-forces are computed"
    t0 = time()
    
    #split work evenly among processes
    result = [process.remote(Particles[i * Load:(i+1) * Load], NgbTree_ref, step) \
              for i in range(NCPU-1)]
    result.append(process.remote(Particles[(NCPU-1) * Load:], NgbTree_ref, step))
    
    Particles_new = list()
    while len(result):
        done_id, result = ray.wait(result)
        Particles_new += ray.get(done_id[0])
        
    t1 = time()
    Problem.Timer["WVT"] += t1-t0
    return Particles_new