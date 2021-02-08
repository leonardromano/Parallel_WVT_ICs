#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 14:48:49 2021

@author: leonard
"""
from time import time
import ray
from sys import exit

from output.output import write_output

from init.positions import sample
from init.setup import setup
from Parameters.constants import NCPU
from wvt.relax import regularise_particles

def main():
    "Set up a box with SPH particles, then WVT relax the particles"
    ray.shutdown()
    ray.init(num_cpus = NCPU)
    t0 = time()
    Particles_ref, Problem, Functions = setup()
    #sample position and set model density
    Particles_ref = sample(Particles_ref, Problem, \
                           [Functions.Position_func, Functions.Density_func])
    #do the WVT regularisation loop
    Particles_ref = regularise_particles(Particles_ref, Problem, Functions.Density_func)
    #now set the velocity and entropy for each particle
    Particles_ref = sample(Particles_ref, Problem, \
                           [Functions.Velocity_func, Functions.Entropy_func])
    write_output(Particles_ref, Problem)
    t1 = time()
    T = t1 - t0
    print("Successfully created ICs! Took %g seconds.\n"%T)
    print("Compuational cost of individual parts:")
    print("INIT: %g (%g)"%(Problem.Timer["INIT"], Problem.Timer["INIT"]/T))
    print("TREE: %g (%g)"%(Problem.Timer["TREE"], Problem.Timer["TREE"]/T))
    print("DENSITY: %g (%g)"%(Problem.Timer["DENSITY"], Problem.Timer["DENSITY"]/T))
    print("L1-ERROR: %g (%g)"%(Problem.Timer["L1-ERROR"], Problem.Timer["L1-ERROR"]/T))
    print("WVT: %g (%g)"%(Problem.Timer["WVT"], Problem.Timer["WVT"]/T))
    print("REDIST: %g (%g)"%(Problem.Timer["REDIST"], Problem.Timer["REDIST"]/T))
    print("OUTPUT: %g (%g)"%(Problem.Timer["OUTPUT"], Problem.Timer["OUTPUT"]/T))
    print("We are done now.\nBye.")
    ray.shutdown()
    exit()
main()