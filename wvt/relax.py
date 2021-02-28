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
from sys import exit, stdout
from time import time

#custom libraries
from output.diagnostics import write_diagnostics, write_step_file
from Parameters.constants import LARGE_NUM
from Parameters.parameter import Npart, NDIM, Maxiter, MpsFraction, \
    StepReduction, LimitMps, LimitMps10, LimitMps100, LimitMps1000, \
    LastMoveStep, RedistributionFrequency, LastFillStep, GapFillingFrequency, \
    MaxNpart, SAVE_WVT_STEPS
from sph.sph import initial_guess_hsml, find_sph_quantities
from tree.tree import ngbtree, update_Tp
from utility.errors import compute_l1_error
from wvt.drift import drift_particles
from wvt.forces import compute_wvt_forces
from wvt.gaps import fill_gaps
from wvt.redistribution import redistribute

def regularise_particles(Particles, Problem, density_func):
    "This is the main loop of the IC making"
    print("Starting iterative SPH regularisation \n" + \
          "   Maxiter=%d, MpsFraction=%g StepReduction=%g "\
              %(Maxiter, MpsFraction, StepReduction) + \
          "LimitMps=(%g,%g,%g,%g)\n\n"\
              %(LimitMps, LimitMps10, LimitMps100, LimitMps1000))
    t0 = time()

    step     = Npart**(-1/NDIM) / MpsFraction
    last_cnt = LARGE_NUM
    err_last = LARGE_NUM
    err_diff = LARGE_NUM
    niter    = 0
    
    #build the search tree and update SPH quantities
    NgbTree_ref = ray.put(ngbtree(Particles, Problem))
    Particles = initial_guess_hsml(Particles, NgbTree_ref)
    
    while(niter < Maxiter):
        Particles = find_sph_quantities(Particles, NgbTree_ref, Problem)
        
        niter += 1
        if SAVE_WVT_STEPS:
            write_step_file(Particles, Problem, niter)
            
        #fill empty regions with new particles
        if niter <= LastFillStep and niter % GapFillingFrequency == 0 \
            and Problem.Npart < MaxNpart:
            Particles, NgbTree_ref = fill_gaps(Particles, Problem)
        
        #redistribute particles
        if niter <= LastMoveStep and niter % RedistributionFrequency == 0:
            Particles, NgbTree_ref = redistribute(Particles, Problem, density_func, \
                                              niter)

        #next find minimum, maximum and average error and their variance
        err_min, err_max, \
        err_mean, err_sigma = compute_l1_error(Particles, Problem)
        
        #update err_diff and err_last
        err_diff  = ( err_last - err_mean ) / err_mean
        print("#%02d: Err min=%3g max=%3g mean=%03g sigma=%03g diff=%03g step=%g\n"\
              %(niter, err_min, err_max, err_mean, err_sigma, err_diff, step))
        err_last  = err_mean
        
        #update the particles referenced by the neighbor tree
        NgbTree_ref = update_Tp(Particles, NgbTree_ref, Problem)
        
        #now compute the forces
        Particles = compute_wvt_forces(Particles, Problem, NgbTree_ref, step)
        #now drift all particles
        Particles, cnts = drift_particles(Particles, Problem, density_func)
        #now some diagnostics
        move_Mps = cnts * 100/Problem.Npart
        print("delta %g > dx; %g > dx/10; %g > dx/100; %g > dx/1000\n"\
              %(move_Mps[0], move_Mps[1], move_Mps[2], move_Mps[3]))
        if niter == 1:
            if move_Mps[0] > 80:
                print("WARNING: A lot of initial movement detected." )
                print("Consider increasing MpsFraction in the parameter file!\n")
                exit()
        write_diagnostics(niter, err_diff, move_Mps, \
                          array([err_min, err_max, err_mean, err_sigma]))
        
        #flush the output buffer
        stdout.flush()
        
        #check whether we're done
        if move_Mps[0] < LimitMps or move_Mps[1] < LimitMps10 \
            or move_Mps[2] < LimitMps100 or move_Mps[3] < LimitMps1000:
                break
        
        #enforce convergence if distribution doesnt tighten
        if cnts[1] >= last_cnt and (niter > LastMoveStep or niter % RedistributionFrequency != 0 ):
            step *= StepReduction
        
        last_cnt = cnts[1]
        
        #build the search tree and update SPH quantities for next iteration
        NgbTree_ref = ray.put(ngbtree(Particles, Problem))
    
    if niter >= Maxiter:
        print("Max iterations reached, result might not be converged properly.")
    t1 = time()
    print("Finished WVT relaxation after %03d iterations. Took %g seconds.\n"\
          %(niter, t1 - t0))
    return Particles
                      
            
        
        
        
    
