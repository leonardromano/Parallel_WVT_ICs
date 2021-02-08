#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:36:40 2021

@author: leonard

    Settle SPH particle with weighted Voronoi tesselations (Diehl+ 2012).
    Here hsml is not the SPH smoothing length, but is related to a local
    metric defined ultimately by the density model.
    Relaxation is done in units of the boxsize, hence the box volume is 1
    Return code true means that rerun could be useful
"""
#public libraries
from numpy import array
import ray
from sys import exit
from time import time

#custom libraries
from output.diagnostics import write_diagnostics, write_step_file
from Parameters.constants import LARGE_NUM
from Parameters.parameter import Npart, NDIM, Maxiter, MpsFraction, \
    StepReduction, LimitMps, LimitMps10, LimitMps100, LimitMps1000, \
    LastMoveStep, RedistributionFrequency, SAVE_WVT_STEPS
from sph.sph import initial_guess_hsml, find_sph_quantities
from tree.tree import ngbtree
from utility.errors import compute_l1_error
from wvt.drift import drift_particles
from wvt.forces import compute_wvt_forces
from wvt.redistribution import redistribute

def regularise_particles(Particles_ref, Problem, density_func):
    "This is the main loop of the IC making"
    print("Starting iterative SPH regularisation \n" + \
          "   Maxiter=%d, MpsFraction=%g StepReduction=%g "\
              %(Maxiter, MpsFraction, StepReduction) + \
          "LimitMps=(%g,%g,%g,%g)\n\n"\
              %(LimitMps, LimitMps10, LimitMps100, LimitMps1000))
    t0 = time()
    #compute some necessary variables
    avg_boxsize = 0
    for length in Problem.Boxsize:
        avg_boxsize += length
    avg_boxsize /= NDIM
    
    npart_1D = Npart**(1/NDIM)
    step     = 1./(npart_1D * MpsFraction)
    last_cnt = LARGE_NUM
    err_last = LARGE_NUM
    err_diff = LARGE_NUM
    niter    = 0
    
    #build the search tree and update SPH quantities
    NgbTree_ref   = ray.put(ngbtree(Particles_ref, Problem))
    Particles_ref = initial_guess_hsml(Particles_ref, NgbTree_ref)
    
    while(niter < Maxiter):
        Particles_ref = find_sph_quantities(Particles_ref, NgbTree_ref, Problem)
        
        niter += 1
        if SAVE_WVT_STEPS:
            write_step_file(Particles_ref, Problem, niter)
        
        #redistribute particles
        if niter <= LastMoveStep and niter % RedistributionFrequency == 0:
            Particles_ref, NgbTree_ref = redistribute(Particles_ref, Problem, \
                                                  NgbTree_ref, density_func, \
                                                  niter)

        #next find minimum, maximum and average error and their variance
        err_min, err_max, \
        err_mean, err_sigma = compute_l1_error(Particles_ref, Problem)
        
        #update err_diff and err_last
        err_diff  = ( err_last - err_mean ) / err_mean
        print("   #%02d: Err min=%3g max=%3g mean=%03g sigma=%03g diff=%03g step=%g\n"\
              %(niter, err_min, err_max, err_mean, err_sigma, err_diff, step))
        err_last  = err_mean
        
        #now compute the forces
        Particles_ref = compute_wvt_forces(Particles_ref, Problem, NgbTree_ref, step)
        #now drift all particles
        Particles_ref, cnts = drift_particles(Particles_ref, Problem, density_func)
        #now some diagnostics
        move_Mps = cnts * 100/Npart
        print("        Del %g > Dmps; %g > Dmps/10; %g > Dmps/100; %g > Dmps/1000\n"\
              %(move_Mps[0], move_Mps[1], move_Mps[2], move_Mps[3]))
        if niter == 1:
            if move_Mps[0] > 80:
                print("WARNING: A lot of initial movement detected." )
                print("Consider increasing MpsFraction in the parameter file!\n")
                exit()
        write_diagnostics(niter, err_diff, move_Mps, \
                          array([err_min, err_max, err_mean, err_sigma]))
        
        #check whether we're done
        if move_Mps[0] < LimitMps or move_Mps[1] < LimitMps10 \
            or move_Mps[2] < LimitMps100 or move_Mps[3] < LimitMps1000:
                break
        
        #enforce convergence if distribution doesnt tighten
        if cnts[1] > last_cnt and (niter > LastMoveStep or niter % RedistributionFrequency != 0 ):
            step *= StepReduction
        
        last_cnt = cnts[1]
        
        #build the search tree and update SPH quantities for next iteration
        NgbTree_ref = ray.put(ngbtree(Particles_ref, Problem))
    
    if niter > Maxiter:
        print("Max iterations reached, result might not be converged properly.")
    t1 = time()
    print("Finished WVT relaxation after %03d iterations. Took %g seconds.\n"\
          %(niter, t1 - t0))
    return Particles_ref
                      
            
        
        
        
    
