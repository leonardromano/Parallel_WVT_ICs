#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:25:26 2021

@author: leonard
"""
from numpy import pi
from psutil import cpu_count
from  os import environ

from Parameters.parameter import NDIM, Npart, Scheduler
from utility.utility import factorial

#This File contains all the internal constants of the code

#Parallelization constants
if Scheduler == "SGE":
    NCPU = int(environ['NSLOTS'])
elif Scheduler == "SLURM":
    NCPU = int(environ['SLURM_CPUS_PER_TASK'])
elif Scheduler == "PBS":
    NCPU = int(environ['PBS_NP'])
else:
    NCPU = cpu_count(logical=False)
Load = Npart//NCPU
MIN_LOAD_PER_CORE = 20

#small and big numbers
SMALL_NUM = 1e-15
LARGE_NUM = 1e15

#stuff for random numbers
#seed(69420)

#stuff for integer positions
BITS_FOR_POSITIONS = 32
MAX_INT = (1 << BITS_FOR_POSITIONS) - 1

#stuff needed by the tree
TREE_NUM_BEFORE_NODESPLIT = 3

#stuff for neighbor-search
if NDIM == 2:
    DESNNGB = 16
    NNGBDEV = 1
    NGBMAX  = (DESNNGB*8)
else:
    DESNNGB = 64
    NNGBDEV = 2
    NGBMAX  = (DESNNGB*8)
    
#Volume of NDIM-Ball
if NDIM % 2 == 0:
    NORM_COEFF =pi**(NDIM//2)/factorial(NDIM//2)
else:
    NORM_COEFF = 2 * factorial((NDIM - 1)//2) * (4 *pi)**((NDIM-1)//2) / factorial(NDIM)
    
#Normalization constant for Kernel
if NDIM == 1:
    NORM_FAC = 5/4
else:
    NORM_FAC = factorial(NDIM + 5, NDIM + 2) / (120 * NORM_COEFF)