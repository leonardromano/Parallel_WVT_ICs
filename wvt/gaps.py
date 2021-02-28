#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:52:12 2021

@author: leonard
"""

from numpy import zeros, sqrt, linspace, meshgrid, exp, partition, where, copy
import ray
from sys import exit
from time import time

from data.structures import particle_data
from Parameters.constants import NCPU
from Parameters.parameter import NDIM, Problem_Specifier, Nfill
from sph.Kernel import kernel
from sph.sph import find_sph_quantities
from tree.tree import ngbtree
from utility.integer_coordinates import convert_to_int_position

###############################################################################
#worker class

@ray.remote
class worker():
    def __init__(self, Nbins, dr, Problem):
        self.out = zeros(tuple(Nbins for _ in range(NDIM)))
        self.Nbins = Nbins
        self.dr = dr
        self.Periodic = Problem.Periodic
        
    def update(self, p):
        #initialize the bounds of the region occupied by the particle
        #now compute the bounds of our computation
        h  = p.Hsml
        pt = p.position
        
        bounds = list()
        for i in range(NDIM):
            bounds.append([int((pt[i] - h)/self.dr[i]), int((pt[i] + h)/self.dr[i])])
            if not self.Periodic[i]:
                bounds[i][0] = max(0, bounds[i][0])
                bounds[i][1] = min(self.Nbins, bounds[i][1])
        
        #loop over all bins the particle is occupying
        self.nested_sph_loop(0, bounds, pt, h)
    
    def find_cell_index(self, index):
        ind = copy(index)
        for i in range(NDIM):
            if self.Periodic[i]:
                while ind[i] < 0:
                    ind[i] += self.Nbins
                while ind[i] >= self.Nbins:
                    ind[i] -= self.Nbins
        return ind

    def nested_sph_loop(self, axis, bounds, pt, h, index = zeros(NDIM, dtype=int)):
        if axis < NDIM:
            ind = copy(index)
            for i in range(bounds[axis][0], bounds[axis][1]):
                ind[axis] = i
                self.nested_sph_loop(axis+1, bounds, pt, h, ind)
        else:
            #get the distance between particle and cell
            ds2 = 0
            for i in range(NDIM):
                ds2 += (self.dr[i] * (index[i] + 0.5) - pt[i])**2
            ds = sqrt(ds2)/h
            
            #check if we can discard this cell
            if ds > 1:
                return
            
            #find index of cell within periodic box
            index = self.find_cell_index(index)
            
            #now add weight to the cell
            self.out[tuple(index)] += kernel(ds, h)
    
    def process(self, particles):
        for particle in particles:
            self.update(particle)
        print(self.out)
        return self.out  

###############################################################################
#density functions#############################################################

if Problem_Specifier == "Constant":
    def rho_model(*args):
        return 1
elif Problem_Specifier == "Rayleigh-Taylor":
    def rho_model(x, y):
        return 1. + 1. / (1. + exp(- (y - 0.5)/0.025)) 
else:
    print("Problem not yet implemented!")
    ray.shutdown()
    exit()
    
###############################################################################

def compute_density_map(Particles, Problem, Nbins, dr):
    "Project the density on a grid"
    load = Problem.Npart//NCPU
    
    actors = [worker.remote(Nbins, dr, Problem) for _ in range(NCPU)]
    
    pending = [actors[i].process.remote(Particles[i * load:(i+1) * load]) \
                  for i in range(NCPU-1)]
    pending.append(actors[NCPU-1].process.remote(Particles[(NCPU-1) * load:]))
    
    #now reduce the individual results
    rho = zeros(tuple(Nbins for _ in range(NDIM)))
    print(rho.shape)
    while len(pending):
        done, pending = ray.wait(pending)
        rho += ray.get(done[0])
    print(rho)
    
    rho *= Problem.Mpart
    return rho

def fill_gaps(Particles, Problem):
    "If we have specified this timestep for gap-filling fill gaps"
    t0 = time()
    #first compute the density map
    Nbins = int(Problem.Npart**(1/NDIM))
    dr = Problem.Boxsize/Nbins
    rho = compute_density_map(Particles, Problem, Nbins, dr)
    
    #now compute the relative error field
    grid = list()
    for i in range(NDIM):
        grid.append(linspace(0, Problem.Boxsize[i], Nbins))
    grid = meshgrid(*grid)
    
    error_map = rho/rho_model(*grid) - 1
    
    #Now find Nfill unpopulated positions
    smallest = partition(error_map, Nfill, axis=None)[:Nfill]
    print(smallest)
    indices = [where(error_map == value) for value in smallest]
    print(indices)
    
    #populate unpopulated regions
    ninsert = 0
    for index in indices:
        listOfCoordinates = list(zip(*index))
        print(listOfCoordinates)
        for coord in listOfCoordinates:
            position = zeros(NDIM)
            boundary = 0
            for i in range(NDIM):
                if coord[i] < 3 or Nbins - coord[i] < 3:
                    #Boundary regions are by construction underpopulated -- ignore
                    boundary = 1
                    break
                position[i] += coord[i] * dr[i]
            if boundary:
                continue
            #spawn particle
            print(position, coord, error_map[coord])
            Particles.append(particle_data(Problem.Npart + ninsert))
            Particles[-1].position = convert_to_int_position(position, Problem.FacIntToCoord)
            Particles[-1].Hsml = position.max()
            ninsert += 1
    
    #now update the particle mass
    frac = ninsert/Problem.Npart
    Problem.Npart += ninsert
    Problem.Mpart /= (1 + frac)
    
    print("Spawned %d particles after trying %d minima."%(ninsert, Nfill))
    
    t1 = time()
    Problem.Timer["FILL"] += t1-t0
    #Now redo the SPH-neighbor-search
    NgbTree_ref = ray.put(ngbtree(Particles, Problem))
    Particles = find_sph_quantities(Particles, NgbTree_ref, Problem)
    return Particles, NgbTree_ref
    