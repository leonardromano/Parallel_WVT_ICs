#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:41:23 2021

@author: leonard
"""
from math import erf
from numpy import log, exp, zeros
from numpy.random import uniform
import ray
from time import time

from Parameters.parameter import Npart, NDIM, LastMoveStep, \
    RedistributionFrequency, MoveFractionMin, MoveFractionMax, \
    ProbesFraction
from sph.sph import find_sph_quantities
from tree.tree import ngbtree
from utility.utility import relative_density_error_with_sign

def redistribute(Particles, Problem, density_func, niter):
    "If we have specified this timestep for redistribution redistribute"
    t0 = time()
        
    decay = log(MoveFractionMax / MoveFractionMin) / (LastMoveStep / RedistributionFrequency - 1)
    moveFraction = MoveFractionMax * exp(-decay * (niter / RedistributionFrequency - 1))
    movePart = int(Npart * moveFraction)
    maxProbes = int(Npart * ProbesFraction * moveFraction / MoveFractionMax)
    redistribute_particles(movePart, maxProbes, Particles, Problem, density_func)
        
    t1 = time()
    Problem.Timer["REDIST"] += t1-t0
    #Now redo the SPH-neighbor-search
    NgbTree_ref = ray.put(ngbtree(Particles, Problem))
    Particles = find_sph_quantities(Particles, NgbTree_ref, Problem)
    return Particles, NgbTree_ref

def redistribute_particles(movePart, maxProbes, Particles, Problem, density_func):
    "Take random high density particles and put them in low density regions"
    print("Attempting to redistribute %d particles by probing %d"\
          %(movePart, maxProbes))
    redistCounter = 0
    probeCounter  = 0
    for i in range(movePart):
        if probeCounter < maxProbes:
            part_i, success_flag, probeCounter = \
                find_particle_to_redistribute(probeCounter, maxProbes, \
                                              Particles)
            #only if we found a particle are we allowed to move it
            if success_flag:
                #find a suitable partner and move our particle to its neighborhood
                part_j = find_particle_as_target_location(Particles)
                move_particle_in_neighborhood_of(part_i, part_j, Problem, \
                                                 density_func)
                redistCounter +=1
    print("Redistributed %d particles after probing %d particles\n"\
          %(redistCounter, probeCounter))
    
def find_particle_to_redistribute(probeCounter, maxProbes, Particles):
    "Pick random particles until we find a high density particles"
    part_i = random_particle(Particles)
    success_flag = 1
    run = True
    while run:
        #make sure the particle has not been moved yet
        while part_i.Redistributed:
            part_i = random_particle(Particles)
        #if we have probed more than necessary stop
        run = probeCounter < maxProbes
        if not run:
            success_flag = 0
            break
        probeCounter += 1
        #does it make sense to move this particle?
        if not accept_particle_for_movement(part_i):
            part_i = random_particle(Particles)
            continue
        if part_i.Redistributed:
            probeCounter -= 1
        else:
            #found a particle that may be moved. we are done.
            part_i.Redistributed = 1
            run = False
    return part_i, success_flag, probeCounter

def find_particle_as_target_location(Particles):
    "Pick random particles until we have a low density one"
    part_j = random_particle(Particles)
    #suitable neighbors need to have not been moved and they should have
    #too low density yet
    while(part_j.Redistributed or not accept_particle_as_target(part_j)):
        part_j = random_particle(Particles)
    return part_j

def move_particle_in_neighborhood_of(part_i, part_j, Problem, density_func):
    "Update position of redistributed particle"
    part_i.position = zeros(NDIM, dtype = int)
    for i in range(NDIM):
        part_i.position[i] += position_in_proximity_of(part_j, Problem, i)
    #now update the model density
    density_func(part_i)

def accept_particle_as_target(particle):
    "Only true if density smaller than model and some luck"
    return uniform() < -1 * relative_density_error_with_sign(particle)

def accept_particle_for_movement(particle):
    "True with a higher probability for high density particles"
    return uniform() < erf(relative_density_error_with_sign(particle))
        
def position_in_proximity_of(particle, Problem, axis):
    "Place particle within 0.3Hsml of the target particle"
    ret = -1.
    while(ret < 0 or ret >= Problem.Boxsize[axis]):
        ret = particle.position[axis] * Problem.FacIntToCoord[axis] \
             + ( 2.0 * uniform() - 1.0 ) * particle.Hsml * 0.3
        if Problem.Periodic[axis]:
            if ret >= Problem.Boxsize[axis]:
                ret -= Problem.Boxsize[axis]
            elif ret < 0.0:
                ret += Problem.Boxsize[axis]
    return int(ret/Problem.FacIntToCoord[axis])
    
def random_particle(Particles):
    "Draw a random particle from the set of particles"
    return Particles[int(uniform() * (Npart - 1))]