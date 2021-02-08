#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:43:29 2021

@author: leonard
"""
import h5py
from numpy import zeros
import ray
from time import time

from Parameters.parameter import output, Npart, NDIM

def write_diagnostics(niter, err_diff, move_Mps, err_quad):
    "iteration err_min err_max err_mean err_sigma err_diff move_Mps[4]"
    #write to file
    if niter == 1:
        file = open(output + "diagnostics.log", "w")
    else:
        file = open(output + "diagnostics.log", "a")
    file.write("%03d %g %g %g %g %g %g %g %g %g\n"\
               %(niter, *err_quad, err_diff, *move_Mps))
    file.close()

def write_step_file(Particles_ref, Problem, niter):
    "writes all the particle data in a hdf5 file."
    t0 = time()
    
    f = h5py.File("%sstepfiles/step_%03d.hdf5"%(output, niter), "w")
    #first dump all the header info
    header = f.create_group("Header")
    h_att = header.attrs
    h_att.create("NumPart", Npart)
    h_att.create("Mpart", Problem.Mpart)
    h_att.create("Boxsize", Problem.Boxsize)
    h_att.create("NDIM", NDIM)
    #now make the data sets for the particle data
    IDs        = zeros((Npart), dtype = int)
    positions  = zeros((Npart, NDIM), dtype = float)
    velocities = zeros((Npart, NDIM), dtype = float)
    densities  = zeros((Npart), dtype = float)
    hsml       = zeros((Npart), dtype = float)
    
    Particles = ray.get(Particles_ref)
    for i in range(Npart):
        IDs[i]        += Particles[i].ID
        positions[i]  += Particles[i].position * Problem.FacIntToCoord
        velocities[i] += Particles[i].velocity
        densities[i]  += Particles[i].Rho
        hsml[i]       += Particles[i].Hsml
    f.create_dataset("PartData/IDs", data = IDs,  dtype = "u4")
    f.create_dataset("PartData/Coordinates", data = positions, dtype = "f4")
    f.create_dataset("PartData/Velocity", data = velocities, dtype = "f4")
    f.create_dataset("PartData/Density", data = densities, dtype = "f4")
    f.create_dataset("PartData/SmoothingLength", data = hsml, dtype = "f4")
    f.close()
    
    t1 = time()
    Problem.Timer["OUTPUT"] += t1 - t0