#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:50:23 2021

@author: leonard
"""
from numpy import ones, zeros, empty
import ray
from sys import exit
from time import time

from Parameters.constants import BITS_FOR_POSITIONS, MAX_INT, \
                                 TREE_NUM_BEFORE_NODESPLIT
from Parameters.parameter import NDIM

class treepoint():
    "reduced particle data for tree applications"
    def __init__(self, particle):
        self.ID        = particle.ID
        self.position  = particle.position
        self.Hsml      = particle.Hsml
        self.Error     = particle.Error

def update_Tp(particles, NgbTree_ref, Problem):
    t0 = time()

    NgbTree = ray.get(NgbTree_ref)
    NgbTree.Tp = { particle.ID: treepoint(particle) for particle in particles}
    NgbTree_ref = ray.put(NgbTree)

    t1 = time()
    Problem.Timer["TREE"] += t1-t0
    return NgbTree_ref

class ngbnode():
    "A structure for node data"
    def __init__(self):
        #By default initialize the root node
        self.center   =  (1 << (BITS_FOR_POSITIONS - 1)) * \
            ones(NDIM, dtype = int)             #The center of the node
        self.level    =  0                      #A useful way to store the sidelength of the node
        
        #refernce information to other nodes
        self.sibling  = -1                          
        self.nextnode = -1
        self.father   = -1
        #Is this node empty?
        self.notEmpty = 0
        
        #SPH data
        self.Mass              = 0.0
        self.center_offset_min = 0
        self.center_offset_max = 0

class ngbtree():
    "A neighbor search tree"
    def __init__(self, particles, Problem):
        #List of particles
        t0 = time()
        
        self.Tp            = { particle.ID: treepoint(particle) for particle in particles}
        self.Mpart         = Problem.Mpart
        self.FacIntToCoord = Problem.FacIntToCoord
        self.Boxsize       = Problem.Boxsize
        self.Periodic      = Problem.Periodic

        self.Father      = empty(Problem.Npart, dtype = int)
        self.Nextnode    = empty(Problem.Npart + 2, dtype = int)
        self.Nodes       = list()
        
        #data for referencing nodes/particles
        self.MaxPart              = Problem.Npart
        self.MaxNodes             = 1 + Problem.Npart + 100
        self.NextFreeNode         = 0
        self.FirstNonTopLevelNode = 0
        
        #Now build the tree
        self.treebuild()
        t1 = time()
        
        Problem.Timer["TREE"] += t1-t0
        
    def get_nodep(self, no):
        return self.Nodes[no-self.MaxPart]
        
    def treebuild(self):
        "construct the tree and fill the nodes with all the necessary information"
        self.treeconstruct()
        #start updating the tree nodes starting from the root node
        self.update_node_recursive(self.MaxPart)
            
        
    def update_node_recursive(self, no):
        "walk down the tree to update the SPH information in all nodes"
        nop = self.get_nodep(no)
        #initialise the SPH info
        mass      = 0.0
        not_empty = 0
        halflen   = 1 << (BITS_FOR_POSITIONS - 1 - nop.level)
        center_offset_min = ones(NDIM, dtype = int) * (halflen -1)
        
        center_offset_max = -halflen * ones(NDIM, dtype = int)
            
        p = nop.nextnode
        while p != nop.sibling:
            if p>=0:
                if p >= self.MaxPart:
                    #node
                    self.update_node_recursive(p)
                if p < self.MaxPart:
                    #particle
                    mass += self.Mpart
                        
                    offset = self.Tp[p].position - nop.center
                    for i in range(NDIM):
                        if offset[i] < center_offset_min[i]:
                            center_offset_min[i] = offset[i]
                        if offset[i] > center_offset_max[i]:
                            center_offset_max[i] = offset[i]
                        
                    not_empty = 1
                    p = self.Nextnode[p]
                else:
                    #node
                    node = self.get_nodep(p)
                    
                    mass += node.Mass
                        
                    offset_min = node.center_offset_min + node.center \
                                                        - nop.center
                    offset_max = node.center_offset_max + node.center \
                                                        - nop.center
                    for i in range(NDIM):                                        
                        if offset_min[i] < center_offset_min[i]:
                            center_offset_min[i] = offset_min[i]
                        if offset_max[i] > center_offset_max[i]:
                            center_offset_max[i] = offset_max[i]
                        
                    not_empty |= node.notEmpty
                    p = node.sibling
        #now update the node
        self.get_nodep(no).Mass              = mass
        self.get_nodep(no).notEmpty          = not_empty
        self.get_nodep(no).center_offset_min = center_offset_min
        self.get_nodep(no).center_offset_max = center_offset_max 
        
    def treeconstruct(self):
        "create the root node and recursively all the daughternodes and so on"
        #create the root node
        self.NextFreeNode = self.MaxPart
        self.Nodes.append(ngbnode())
        self.NextFreeNode +=1
            
        index_list = list()
        for i in range(self.MaxPart):
            index_list.append([i, 0])
        self.insert_points(self.MaxPart, index_list, self.MaxPart, 0, -1)
        
    def insert_points(self, num, index_list, th, level, sibling):
        "Recursively constructs the tree nodes and fills them with particles"
        if level >= BITS_FOR_POSITIONS:
            print("Need higher resolution! BITS_FOR_POSITIONS = %d, level = %d"\
                  %(BITS_FOR_POSITIONS, level))
            exit()
        mask = 1 << (BITS_FOR_POSITIONS - 1 - level)
        shift = zeros(NDIM, dtype = int)
        for i in range(NDIM):
            shift[i] += (BITS_FOR_POSITIONS - 1 - i - level)
        
        subcount  = zeros(1 << NDIM, dtype = int)
        subnode   = zeros(1 << NDIM, dtype = int)
        subintpos = zeros((1 << NDIM, NDIM), dtype=int)
        #determine the nodes in which the particles will be put
        for i in range(num):
            p = index_list[i][0]
            intpos = self.Tp[p].position
            snode = 0
            for i in range(NDIM):
                snode |= ((intpos[i] & mask) >> shift[i])
            subcount[snode] += 1
            subintpos[snode] = intpos
            index_list[i][1] = snode
        
        centermask = ~(MAX_INT >> level) & MAX_INT
        centermask >>= 1
        centermask |= ~(MAX_INT >> 1) & MAX_INT
        mask >>= 1
            
        #create the daughter nodes
        nfreep_last = 0
        for i in range(1 << NDIM):
            if subcount[i] > TREE_NUM_BEFORE_NODESPLIT:
                thnew = self.NextFreeNode
                self.NextFreeNode +=1
                subnode[i] = thnew
                if nfreep_last:
                    self.get_nodep(thnew - 1).sibling  = thnew
                    self.get_nodep(thnew - 1).nextnode = thnew
                else:
                    self.get_nodep(th).nextnode = thnew
                nfreep = ngbnode()
                nfreep.father = th
                nfreep.level  = level + 1
                for j in range(NDIM):
                    nfreep.center[j] = (subintpos[i,j] & centermask) | mask
                self.Nodes.append(nfreep)
                nfreep_last = nfreep
            
        # now insert the particles that are chained to the node
        p_last = -1
        p_first = -1
        off = 0
        # now insert the particle groups into the created daughter nodes, or chain them to the node
        for count in subcount:
            if count <= TREE_NUM_BEFORE_NODESPLIT:
                # put the particles into the node as a chain
                for i in range(count):
                    p = index_list[off + i][0]
                    if not nfreep_last and p_first == -1:
                        self.get_nodep(th).nextnode = p
                        
                    self.Father[p] = th
                       
                    if p_last >=0:
                        self.Nextnode[p_last] = p
                    p_last = p
                    if p_first == -1:
                        p_first = p
            off += count   
            
        if p_last >= 0:
            self.Nextnode[p_last] = sibling
            
        if nfreep_last:
            if p_last >= 0:
                self.get_nodep(thnew).sibling  = p_first
                self.get_nodep(thnew).nextnode = p_first
            else:
                self.get_nodep(thnew).sibling  = sibling
                self.get_nodep(thnew).nextnode = sibling
            
        off = 0
        for i in range(1 << NDIM):
            if subcount[i] > TREE_NUM_BEFORE_NODESPLIT:
                if subnode[i] < self.MaxPart + 1:
                    print("subnode[i]=%d < MaxPart=%d + 1" \
                          %(subnode[i], self.MaxPart))
                    exit()
                self.insert_points(subcount[i], \
                                   index_list[off : off + subcount[i]], \
                                   subnode[i], level + 1, \
                                   self.get_nodep(subnode[i]).sibling)
            off += subcount[i]