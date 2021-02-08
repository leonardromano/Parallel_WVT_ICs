#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 14:56:12 2021

@author: leonard
"""

#This File contains all the parameters necessary to setup the initial conditions

#General parameters
####################################################################
#output directory
output   = "../ICs/WVT_ICs/" 
#Number of particles
Npart   = 400
#Name of scheduling system
Scheduler = "local"
#save a snapshot after each step
SAVE_WVT_STEPS = True
####################################################################

#WVT Force & convergence parameter
####################################################################
#inversely proportional to the stepsize of the WVT-"force"
MpsFraction = 1.
#Factor by which the stepsize is decreased whenever it is decreased
StepReduction = 0.95
#Convergence limit for large steps
LimitMps = -1
#convergence limit for smaller steps
LimitMps10 = -1
#convergence limit for even smaller steps
LimitMps100 = -1
#convergence limit for even smaller steps
LimitMps1000 = 1
#Maximum number of iterations
Maxiter = 30
#####################################################################

#Redistribution parameters
#####################################################################
#Which fraction should at least be redistributed?
MoveFractionMin = 0.001
#Which fraction should at most be redistributed?
MoveFractionMax = 0.1
#Which fraction should be probed during redistribution step?
ProbesFraction  = 0.1
#How often should we redistribute (e.g. all x iterations)
RedistributionFrequency = 5
#When should the last redistribution happen?
LastMoveStep = 50
#####################################################################

#Problem related parameters
#####################################################################
#Number of dimensions
NDIM = 2
# The name of the problem (and of IC-file)
Problem_Specifier = "Rayleigh-Taylor"
#####################################################################
