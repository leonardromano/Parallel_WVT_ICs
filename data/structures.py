#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:12:04 2021

@author: leonard
"""
from numpy import zeros, ones
from numpy.random import seed
import ray

from Parameters.constants import BITS_FOR_POSITIONS, NCPU
from Parameters.parameter import NDIM

@ray.remote(num_cpus=1)
class particles():
    def __init__(self, particles_ref, ID, load):
        if ID < NCPU-1:
            self.particles = particles_ref[ID * load:(ID+1) * load]
        else:
            self.particles = particles_ref[ID * load:]
        seed(69 + 420 * ID)
    
    def process(self, func, *args):
        "this function does the heavy lifting for the parallelization"
        for particle in self.particles:
            func(particle, *args)
        return [*self.particles]


#A data structure for particles
class particle_data():
    def __init__(self, ID):
        self.position      = zeros(NDIM, dtype = int)
        self.velocity      = zeros(NDIM)
        self.delta         = zeros(NDIM)
        self.ID            = ID
        
        self.neighbors     = list()
        self.Redistributed = 0
        self.CloseToWall   = 0
        
        self.Entropy       = 0 
        self.Rho           = 0
        self.Pressure      = 0
        self.Hsml          = 0
        self.Hwvt          = 0
        self.Rho_Model     = 0


class problem():
    def __init__(self):
        self.name          = ""
        self.Mpart         = 0.
        self.Boxsize       = ones(NDIM)
        self.FacIntToCoord = self.Boxsize/(1 << BITS_FOR_POSITIONS)
        self.Rho_Max       = 1.
        self.Periodic      = ones(NDIM, dtype = int)
        self.Timer         = {"INIT" : 0,
                              "TREE": 0,
                              "DENSITY": 0,
                              "L1-ERROR": 0,
                              "WVT": 0,
                              "REDIST": 0,
                              "OUTPUT": 0}
        
    def update_int_conversion(self):
        self.FacIntToCoord = self.Boxsize/(1 << BITS_FOR_POSITIONS)

class functions():
    def __init__(self):
        self.Density_func  = 0
        self.Entropy_func  = 0
        self.Velocity_func = 0
        self.Position_func = 0