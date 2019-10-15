# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 08:06:10 2018

@author: Salem

By running this script and calling the method
"scaled_hexagonal_lattice()" in the console, an example will run saving data in the folder Results.

If you then open the accompanying mathemtaica file you can visualize the results.

"""

#The usual importations
import numpy as np
import numpy.linalg as la
import numpy.random as npr

import scipy as sp
import scipy.linalg as sla
import scipy.optimize as op

import MeshHandling as MH
import EnergyFunctions as EF



import importlib
importlib.reload(MH)
importlib.reload(EF)


from copy import deepcopy


#=================================================================================================================================
# defining the current working directory
#=================================================================================================================================
from pathlib import Path
import os
cwd = Path(os.getcwd())


results_dir = cwd / "Results" 
#=================================================================================================================================


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#==================================================================================================================================
# Example: Scale a hexagonal lattice uniformly
#==================================================================================================================================
def scaled_hexagonal_lattice():
    '''
    Applies the growth for a hexagonal lattice that is growing with time. The underlying metric growth is homogeneous and isotropic 
    and simply grows exponentially at the rate r = 1 with time.
    
    
    This is an example, for a general problem, what you need is to define the metric tensor as a function of space and time, 
    which in this case is defined by the method "scaling_growth". 
    
    You also would need to define the initial surface as an array of vertex positions (the variable verts). and an array of edges.
    
    edge_array is an array whose rows are [index of first vertex, index of second vertex] for each edge in the lattice.
    '''
    
    #the initial hexagonal lattice
    verts, edge_array = MH.hex_lattice()
    
    
    return growth(scaling_growth, verts, edge_array)                        
#==================================================================================================================================   

def growth(metric_function, init_vertices, init_edges, final_time=1, delta_T=0.01):
    '''
    using the given metric as a function of time the mesh is evolved. 
    
    the initial metric is assumed to be flat since we can take the initial surface, which is 
    embeddable, to parametrize the surface at later times. Meaning we take the x and y coordinates of 
    the initial mesh as the coordinate system for all time
    
    metric_function(edge_center, time): a method that returns the metric as a function of space (edge centers) and time.
    init_vertices: a [number of vertices, 2] array giving the positions of the nodes or vertices
    init_edges: a [number of edges, 2] array giving the edges of the lattice, which for now do not change 
    during growth. Each edge is given as [index_of_vert1, index_of_vert2]
    '''
    
    num_of_edges = init_edges.shape[0]
    
    #the variable containing the vertex positions at each instant
    new_verts = deepcopy(init_vertices)
    
    
    #has to do with getting rid of euclidean motions
    deleteIndices =  (0,1,3)
    insertIndices = (0,0,1)
    
    #the current value of time, set initially to zero
    time = 0
    
    #where the metric will be evaluated to get the new edge length squared
    edge_centers = 0.5  *  (init_vertices[init_edges[:,0]] + init_vertices[init_edges[:,1]])
    # the coordiante vectors du used in calculation of the length squared (dotted with metric tensor)
    edge_vectors = init_vertices[init_edges[:,0]] - init_vertices[init_edges[:,1]]
    
    print("edge_centers.shape = ", edge_centers.shape)
    print("edge_vectors.shape = ", edge_vectors.shape)
    
    #initialize the square lengths, these are taken from the initial lattice since it is assumed embeddable# 
    sqr_edge_lengths = get_square_lengths(metric_function, edge_centers, edge_vectors, time)
    
    
    #spring constants are simply assumed to all be equal to 1
    spring_consts = np.ones(num_of_edges)
    
    num_steps = int(final_time/delta_T) # number of growth steps
    
    
    save_rate = 5 # every save_rate steps the mesh is stored to results folder
    
    #set the springs constants simply all to be equal 1
    spring_consts = np.ones(num_of_edges)
    
    for i in range(num_steps):
        
        #will save the before the first step. files are labeled as (0, 1, 2,...) not actual time or step number
        if (i%save_rate == 0 or i==num_steps - 1):
            print("saving step number:", i)
            #save vertices
            np.savetxt(cwd  / "Results" /
                    ("save-num-" + str(i//save_rate + i//(num_steps - 1)) +  ".npy"), new_verts, delimiter=',')
         
        
        #update the time
        time += delta_T
            
        #the change in the square of the lengths at every time steps. this defines the growth strategy given from gij(t)
        sqr_length_change = get_square_lengths(metric_function, edge_centers, edge_vectors, time) - sqr_edge_lengths

    
        new_verts = EF.update_mesh(new_verts, init_edges, sqr_length_change=sqr_length_change, 
                                   sqr_bar_lengths=sqr_edge_lengths,spring_consts=spring_consts, 
                                   insertIndices=insertIndices, deleteIndices=deleteIndices)
        
        
        
        #calculate the squared edge lengths at the new time
        sqr_edge_lengths += sqr_length_change
        
    return 
#==================================================================================================================================
#==================================================================================================================================

#==================================================================================================================================
# gives the length of an array of coordinate vectors using the position of their centers
#==================================================================================================================================    
def get_square_lengths(metric_function, edge_centers, edge_vectors, time):
    '''
    The length of an array of vectors at the current time using the value of the metric
    
    edge_vectors: vector on the initial lattice which are the displacement between points connected
    by edges.
    edge_centers: centers of the edges, where the metric will be calculated
    '''
    
    num_edges = edge_centers.shape[0]
    
    new_squared_lengths = list(map(lambda i: get_length_of_vec(metric_function, edge_centers[i], edge_vectors[i], time),
                                   range(num_edges)))
    
    
    return np.array(new_squared_lengths)
#==================================================================================================================================
#==================================================================================================================================

#==================================================================================================================================
# gives the length of a single coordinate vector using its position 
#==================================================================================================================================
def get_length_of_vec(metric_function, edge_center, edge_vector, time):
    '''
    gives the length squared of the infinitesimal vector edge_vector which is located at
    the point edge_center.
    '''
    
    length_squared = np.dot(np.dot(edge_vector, metric_function(edge_center, time)), edge_vector)
    
    return length_squared
#==================================================================================================================================
#==================================================================================================================================
    


#==================================================================================================================================
# Grow an initial metric homogeneously and isotropically with time. This is an example metric function
#==================================================================================================================================
def scaling_growth(position, time):
   '''
   This metric is proportional to the identity and grows exponentially at the rate r = 2 log 2. meaning that lenght will double after
   1 time unit.
   
   position is an array that looks like [x, y]
   '''
   
   r = 2*np.log(2)
   return np.eye(2)*np.exp(r*time)
#==================================================================================================================================
#==================================================================================================================================



#==================================================================================================================================
#checks if a length is bigger than a max allowed value then subdivides it correctly
#==================================================================================================================================    
def check_then_subdivide(sqr_bar_lengths, max_allowed, faceVerts, edge_array, newVerts, maxT, num_verts):
    '''
    not implemented yet
    '''
    #check lengths and find where they exceed max allowed
    mask = sqr_bar_lengths > max_allowed
    
    print("mask any:", mask.any())
    
    if(mask.any()):
        #add the new verts (calculate the added)
        added_verts = 0.5*(newVerts[edge_array[mask, 1]] + newVerts[edge_array[mask, 2]])
        #add the new verts (insert them)
        np.insert(newVerts, -1,added_verts, axis=0)
        
        #remove the corresponding edge and replace it with 4 new ones
        #find the new bar_lengths
    
#==================================================================================================================================
#==================================================================================================================================















            