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
output   = "/home/t30/all/ga87reg/Num_seminar/ICs/"  
#Number of particles
Npart   = 4096
#Name of scheduling system
Scheduler = "SGE"
#save a snapshot after each step
SAVE_WVT_STEPS = True
####################################################################

#WVT Force & convergence parameter
####################################################################
#inversely proportional to the stepsize of the WVT-"force"
MpsFraction = 1.
#Factor by which the stepsize is decreased whenever it is decreased
StepReduction = 0.85
#Convergence limit for large steps
LimitMps = -1
#convergence limit for smaller steps
LimitMps10 = -1
#convergence limit for even smaller steps
LimitMps100 = -1
#convergence limit for even smaller steps
LimitMps1000 = 1
#Maximum number of iterations
Maxiter = 200
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
LastMoveStep = 150
#####################################################################

#Gapfilling parameters
#####################################################################
#How many cells can a cell be away from a blob to be considered part of it?
DistanceThreshold = 1
#How many cells does a blob need to have at least to be considered?
BlobSizeThreshold = 5
#How unpopulated does a region need to be?
FillThreshold     = -0.2 
#How many sites should be probed each step?
Nfill = 10
#How many particles do we want to have at most?
MaxNpart = 4450
#How often should we try to fill gaps (e.g. all x iterations)
GapFillingFrequency = 1
#When should the last Gapfilling happen?
LastFillStep = 150
#####################################################################

#Problem related parameters
#####################################################################
#Number of dimensions
NDIM = 2
# The name of the problem (and of IC-file)
Problem_Specifier = "Rayleigh-Taylor"
#####################################################################
