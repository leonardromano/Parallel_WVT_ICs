#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:54:12 2021

@author: leonard
"""
import h5py
from numpy import zeros
from time import time

from Parameters.parameter import output, NDIM


def write_output(Particles, Problem):
    "writes all the particle data in a hdf5 file."
    t0 = time()
    
    f = h5py.File("%s%s.hdf5"%(output, Problem.name), "w")
    #first dump all the header info
    header = f.create_group("Header")
    h_att = header.attrs
    h_att.create("NumPart", Problem.Npart)
    h_att.create("Mpart", Problem.Mpart)
    h_att.create("Boxsize", Problem.Boxsize)
    h_att.create("Periodic", Problem.Periodic)
    h_att.create("NDIM", NDIM)
    #now make the data sets for the particle data
    Npart = Problem.Npart
    IDs        = zeros((Npart), dtype = int)
    positions  = zeros((Npart, NDIM), dtype = float)
    velocities = zeros((Npart, NDIM), dtype = float)
    densities  = zeros((Npart), dtype = float)
    hsml       = zeros((Npart), dtype = float)
    pressures  = zeros((Npart), dtype = float)
    entropies  = zeros((Npart), dtype = float)
    errors     = zeros((Npart), dtype = float)
    for particle in Particles:
        i = particle.ID
        IDs[i]        += i
        positions[i]  += particle.position * Problem.FacIntToCoord
        velocities[i] += particle.velocity
        densities[i]  += particle.Rho
        hsml[i]       += particle.Hsml
        pressures[i]  += particle.Pressure
        entropies[i]  += particle.Entropy
        errors[i]     += particle.Error
    f.create_dataset("PartData/IDs", data = IDs,  dtype = "u4")
    f.create_dataset("PartData/Coordinates", data = positions, dtype = "f4")
    f.create_dataset("PartData/Velocity", data = velocities, dtype = "f4")
    f.create_dataset("PartData/Entropy", data = entropies, dtype = "f4")
    f.create_dataset("PartData/Density", data = densities, dtype = "f4")
    f.create_dataset("PartData/Pressure", data = pressures, dtype = "f4")
    f.create_dataset("PartData/SmoothingLength", data = hsml, dtype = "f4")
    f.create_dataset("PartData/Errors", data = errors, dtype = "f4")
    f.close()
    
    t1 = time()
    Problem.Timer["OUTPUT"] += t1 - t0