#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:32:50 2021

@author: leonard
"""
from numpy import zeros, cos, pi, exp
from numpy.random import uniform

from Parameters.parameter import Npart
from Parameters.constants import BITS_FOR_POSITIONS 

#Rayleigh-Taylor #########################################################
##########################################################################

def setup_Rayleigh_Taylor(Problem, Functions):
    "setup problem and functions for Rayleigh-Taylor problem (DIM = 2)"
    Problem.Boxsize[0]      = 0.5
    Problem.Boxsize[1]      = 1.
    Problem.update_int_conversion()
    Problem.Periodic[1]     = 0
    Problem.Mpart           = 0.75/Npart
    
    Problem.Rho_Max         = 2.
    Functions.Density_func  = Rayleigh_Taylor_Instability_Density
    Functions.Entropy_func  = Rayleigh_Taylor_Instability_Entropy
    Functions.Velocity_func = Rayleigh_Taylor_Instability_Velocity
    Functions.Position_func = Rayleigh_Taylor_Instability_Position

def Rayleigh_Taylor_Instability_Density(particle):
    "A step function in y direction"
    y = particle.position[1]/(1 << BITS_FOR_POSITIONS)
    particle.Rho_Model = 1. + 1. / (1. + exp(- (y - 0.5)/0.025)) 

"""  Initial pressure is assigned to produce Hydrostatic Equilibrium with a
     uniform gravitational acceleration g = -1/2 in the y-direction (at the
     interface, P = rho2/gamma = 10/7 so the soundspeed c_s = 1" - taken from
     Hopkins & Raives 2016 Sec. 3.9, second paragraph
     
     Hydrostatic Equation: 
     rho F = nabla P --> dP/dy = rho g --> P = P_0 + rho g y
"""

def Rayleigh_Taylor_Instability_Entropy(particle):
     gamma = 1.4
     rho2 = 2.0
     grav_acc = -0.5
     y = particle.position[1] / (1 << BITS_FOR_POSITIONS)
     particle.Pressure = rho2 / gamma + grav_acc *  particle.Rho * (y - 0.5)
     particle.Entropy =  particle.Pressure * particle.Rho**(-gamma)

def Rayleigh_Taylor_Instability_Velocity(particle):
    "Only particles in the center are moving"
    x = particle.position[0] / (1 << BITS_FOR_POSITIONS)
    y = particle.position[1] / (1 << BITS_FOR_POSITIONS)
    out = zeros(2)
    if (0.3 < y < 0.7):
        #density perturbation in x-direction
        out[1] = 0.025 * (1 + cos(8 * pi * (x + 0.25))) * \
                         (1 + cos(5 * pi * (y - 0.5)))
    particle.velocity = out

def Rayleigh_Taylor_Instability_Position(particle):
    out = zeros(2, dtype = int)
    out[0] += int(uniform() * (1 << BITS_FOR_POSITIONS))
    r = uniform()
    if r < 1/3:
        out[1] += int(3 * r * (1 << (BITS_FOR_POSITIONS - 1)))
    else:
        out[1] += int((1 + 3 * r) * (1 << (BITS_FOR_POSITIONS - 2)))
    particle.position = out
##########################################################################
##########################################################################

#constant ################################################################
##########################################################################

def setup_constant(Problem, Functions):
    "setup problem and functions for Rayleigh-Taylor problem (DIM = 2)"
    Problem.Boxsize[0]      = 1.
    Problem.Boxsize[1]      = 1.
    Problem.update_int_conversion()
    Problem.Mpart           = 1./Npart
    
    Problem.Rho_Max         = 1.
    Functions.Density_func  = constant_Density
    Functions.Entropy_func  = constant_Entropy
    Functions.Velocity_func = constant_Velocity
    Functions.Position_func = constant_Position

def constant_Density(particle):
    "A constant density"
    particle.Rho_Model = 1.


def constant_Entropy(particle):
    "Constant entropy"
    particle.Entropy = 1

def constant_Velocity(particle):
    "No initial motion"
    particle.velocity = zeros(2)

def constant_Position(particle):
    out = zeros(2, dtype = int)
    out[0] += int(uniform() * (1 << BITS_FOR_POSITIONS))
    out[1] += int(uniform() * (1 << BITS_FOR_POSITIONS))
    particle.position = out
##########################################################################
##########################################################################
    
    
    