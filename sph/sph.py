#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:59:39 2021

@author: leonard
"""
from numpy import asarray
import ray
from time import time

from data.structures import particles
from Parameters.constants import DESNNGB, BITS_FOR_POSITIONS, NORM_COEFF, \
    NCPU, Load
from Parameters.parameter import NDIM
from tree.treewalk import density

###############################################################################
#per particle processing
def guess_hsml(particle, NgbTree):
    i = particle.ID
    no = NgbTree.Father[i]
    while(10 * DESNNGB * NgbTree.Mpart > NgbTree.get_nodep(no).Mass):
        p = NgbTree.get_nodep(no).father
        if p < 0:
            break
        no = p
        
    if(NgbTree.get_nodep(no).level > 0):
        length = (1 << (BITS_FOR_POSITIONS - NgbTree.get_nodep(no).level)) * \
            NgbTree.FacIntToCoord.max()
    else:
        length = NgbTree.Boxsize.max()

    particle.Hsml =  length * (DESNNGB * NgbTree.Mpart / \
                               NgbTree.get_nodep(no).Mass / NORM_COEFF)**(1/NDIM)
        
###############################################################################


def find_sph_quantities(Particles_ref, NgbTree_ref, Problem):
    "Driver routine calling function calls that do all the work"
    t0 = time()
    #now walk the tree to determine the weights and sph-quantities
    Particles_ref = density(Particles_ref, NgbTree_ref)
    
    t1 = time()
    Problem.Timer["DENSITY"] += t1-t0
    return Particles_ref

def initial_guess_hsml(Particles_ref, NgbTree_ref):
    "computes an initial guess for the smoothing lengths"
    #split work evenly among processes
    actors = [particles.remote(Particles_ref, i, Load) for i in range(NCPU)]
    
    result = [actor.process.remote(guess_hsml, NgbTree_ref) for actor in actors]
    
    Particles_new = list()
    while len(result):
        done_id, result = ray.wait(result)
        Particles_new += ray.get(done_id[0])
    
    return ray.put(asarray(Particles_new))
        