#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:52:12 2021

@author: leonard
"""

from numpy import zeros, sqrt, linspace, meshgrid, exp, asarray, where
import ray
from sys import exit
from time import time

from data.structures import particle_data
from Parameters.constants import NCPU
from Parameters.parameter import NDIM, Problem_Specifier, Nfill, \
    DistanceThreshold, BlobSizeThreshold, FillThreshold
from sph.Kernel import kernel
from sph.sph import find_sph_quantities
from tree.tree import ngbtree
from utility.integer_coordinates import convert_to_int_position
from utility.utility import norm

###############################################################################
#worker class

@ray.remote
class worker():
    def __init__(self, Nbins, dr, Problem):
        self.out = zeros(tuple(Nbins for _ in range(NDIM)))
        self.Nbins = Nbins
        self.dr = dr
        self.Periodic = Problem.Periodic
        self.FacIntToCoord = Problem.FacIntToCoord
        
    def update(self, p):
        #initialize the bounds of the region occupied by the particle
        #now compute the bounds of our computation
        h  = p.Hsml
        pt = p.position * self.FacIntToCoord
        
        bounds = list()
        for i in range(NDIM):
            bounds.append([int((pt[i] - h)/self.dr[i]), int((pt[i] + h)/self.dr[i])])
            if not self.Periodic[i]:
                bounds[i][0] = max(0, bounds[i][0])
                bounds[i][1] = min(self.Nbins, bounds[i][1])
        
        #loop over all bins the particle is occupying
        index = [0 for _ in range(NDIM)]
        self.nested_sph_loop(0, bounds, pt, h, index)
    
    def find_cell_index(self, index):
        for i in range(NDIM):
            if self.Periodic[i]:
                while index[i] < 0:
                    index[i] += self.Nbins
                while index[i] >= self.Nbins:
                    index[i] -= self.Nbins
        return index

    def nested_sph_loop(self, axis, bounds, pt, h, index):
        if axis < NDIM:
            for i in range(bounds[axis][0], bounds[axis][1]):
                index[axis] = i
                self.nested_sph_loop(axis+1, bounds, pt, h, index)
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
            self.out[tuple(index)[::-1]] += kernel(ds, h)
    
    def process(self, particles):
        for particle in particles:
            self.update(particle)
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
# blob-class ##################################################################

class blob():
    "This class represents a collection of closeby cells"
    def __init__(self, cell):
        self.cells = [cell[::-1]]
        self.Ncells = 1
    
    def __iadd__(self, other):
        self.cells += other.cells
        self.Ncells += other.Ncells
        return self
    
    def append(self, cell):
        self.cells.append(cell[::-1])
        self.Ncells += 1
    
    def close(self, cell, Problem, Nbins):
        "check if the cell is close to the blob"
        for point in self.cells:
            dx = zeros(NDIM, dtype = int)
            for i in range(NDIM):
                if Problem.Periodic[i]:
                    if abs(cell[i] - point[i]) < Nbins/2:
                        dx[i] += cell[i] - point[i]
                    elif cell[i] > point[i]:
                        dx[i] += cell[i] - point[i] - Nbins
                    else:
                        dx[i] += cell[i] - point[i] + Nbins
                else:
                    dx[i] += cell[i] - point[i]
            if norm(dx) <= DistanceThreshold:
                return 1
        return 0
    
    def center_of_errormass(self, error_map, dr):
        "compute the center of the blob"
        weight = 0.0
        position = zeros(NDIM)
        #determine cell within blob with 
        for i in range(self.Ncells):
            cell = self.cells[i]
            error = error_map[cell[::-1]]
            weight += error
            position += error * asarray(cell)
        position *= dr/weight
        return position
            
                    
def merge(blobs, Problem, Nbins):
    "Merge all closeby blobs"
    done = list()
    while len(blobs) > 0:
        blob1 = blobs.pop()
        i = 0
        L = len(blobs)
        while i < L:
            if close(blob1, blobs[i], Problem, Nbins):
                blob1 += blobs.pop(i)
                #need to repeat until no other blob is close
                L -= 1
                i  = 0
            else:
                i+= 1
        done.append(blob1)
    return done
        
def close(blob1, blob2, Problem, Nbins):
    "Check if two blobs are close"
    for cell in blob1.cells:
        if blob2.close(cell, Problem, Nbins):
            return 1
    return 0
                
    
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
    while len(pending):
        done, pending = ray.wait(pending)
        rho += ray.get(done[0])
    
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
    
    #First find all underpopulated cells
    cells = list(zip(*where(error_map < FillThreshold)))
    
    #identify closeby cells as blobs
    blobs = list()
    for cell in cells:
        if not blobs:
            blobs.append(blob(cell))
        else:
            seed_new_blob = 1
            for region in blobs:
                #first see if this cell can be added to an already existing blob
                if region.close(cell[::-1], Problem, Nbins):
                    region.append(cell)
                    seed_new_blob = 0
                    break
            if seed_new_blob:
                #this cell seeds a new blob
                blobs.append(blob(cell))
    
    #now merge closeby blobs
    blobs = merge(blobs, Problem, Nbins)
    
    #Consider only the Nfill largest blobs
    blobs.sort(key=lambda x: x.Ncells, reverse=True)
    
    ninsert = 0
    for region in blobs[:Nfill]:
        if region.Ncells >= BlobSizeThreshold:
            position = region.center_of_errormass(error_map, dr)
            #if the underpopulated region is close to a boundary, ignore it
            boundary = 0
            for i in range(NDIM):
                if not Problem.Periodic[i]:
                    if position[i] < 3 * dr[i] or position[i]/dr[i] > Nbins - 3:
                        boundary = 1
                        break
            if boundary:
                continue
            #spawn particle
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
    