#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:11:14 2021

@author: leonard
"""
from numpy import asarray
import ray
from sys import exit

from data.structures import particle_data, problem, functions
from Parameters.parameter import Npart, Problem_Specifier
import init.problems as problems

def setup():
    "Initialize list of particles, problem and determine particle mass"
    #first make a list of all particles
    Particles_ref = ray.put(asarray([particle_data(i) for i in range(Npart)]))
    Prob, Funcs = setup_problem()
    print("Problem %s\n"%(Prob.name) + \
          "   Npart: %d \n"%Npart + \
          "   Mpart: %g \n"%Prob.Mpart + \
          "   Boxsize:" + str(Prob.Boxsize) + "\n" + \
          "   Periodic:" + str(Prob.Periodic) + "\n\n")
    return Particles_ref, Prob, Funcs
    
def setup_problem():
    "Initialize problem and functions"
    Prob  = problem()
    Funcs = functions() 
    Prob.name = Problem_Specifier
    if Prob.name == "constant":
        problems.setup_constant(Prob, Funcs)
    elif Prob.name == "Rayleigh-Taylor":
        problems.setup_Rayleigh_Taylor(Prob, Funcs)
    else:
        print("Problem not yet implemented (NYI).")
        exit()
    return Prob, Funcs
    