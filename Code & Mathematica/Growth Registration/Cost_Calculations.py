# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 07:03:57 2018

@author: Salem

load anaconda in the cluder: 
module load Anaconda3/5.0.1-fasrc02


"""

import numpy as np
import numpy.linalg as la
import numpy.random as npr 
   

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import scipy.optimize as op
from scipy.special import sph_harm


from pathlib import Path
import os

cwd = Path(os.getcwd())


mesh_dir = cwd / "Mesh_Data" / "time9"


def optimize_fun(landmark_indices=[], landmark_disps=np.zeros(3)):
    
    init_vertices = np.loadtxt(mesh_dir / "vertices.txt")
    simplices = np.loadtxt(mesh_dir / "faces.txt", dtype=np.int64) - 1

    simplex_list = get_simplex_list(simplices, init_vertices.shape[0])
    
    final_coordinates = np.array(list(map(get_spherical_coods, init_vertices)))
    
    # below is uncommented when not optimizing
    return np.array(list(map(final_surface, final_coordinates.reshape(final_coordinates.shape))))
    
    print(final_coordinates.shape)
    
    res = op.minimize(cost_function, final_coordinates, args=(
            init_vertices, simplices, simplex_list, landmark_indices, landmark_disps), method='BFGS')
    
    final_verts = np.array(list(map(final_surface, res.x.reshape(final_coordinates.shape))))
    
    np.savetxt(cwd / "init_vertices.txt", init_vertices)
    np.savetxt(cwd / "final_vertices.txt", final_verts)
    
    np.savetxt(cwd / "faces.txt", simplices)
    
    return final_verts


#=========================================================================================================================
# calculates the cost for two meshes with the same simplex structure.
#=========================================================================================================================
def cost_function(coordinates, init_vertices, simplices, simplex_list,
                  landmark_indices=[], landmark_disps=np.zeros(3)) :
    
    '''
    
    simplex_list: a numpy list giving the simplices that each vertex belongs to. 
    It will have a shape  [number of vertices, number of triangles the vertex belongs to]
    '''
    f_coordinates = coordinates.reshape((coordinates.size//2, 2))
    
    f_verts = np.array(list(map(final_surface, f_coordinates)))
    
    cost = np.zeros(1)
    
    list(map(lambda simplex: cost_per_simplex(simplex, init_vertices, f_verts, simplices, simplex_list, cost), simplices))
        
                
    return cost[0]
#=========================================================================================================================
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================================
    



#=========================================================================================================================
# returns the cost for each simplex from the corresponding positions
#=========================================================================================================================
def cost_per_simplex(simplex, init_vertices, final_vertices, simplices, simplex_list, cost,  a=1, b=1, c=1, d=1):
    '''
    This will give the cost associated with the simplex, by finding the corresponding distortions in surrounding 
    simplices. The simplex is given as an array with indices of the corresponding points
    
    simplex: [index1, index2, index3]
    
    vertices: position of all vertices on the mesh, [num of vertices, num of dimensions]
    
    simplex_list: a numpy list giving the simplices that each vertex belongs to. 
    It will have a shape  [number of vertices, number of triangles the vertex belongs to]
    '''
    

    
    Dil_Eccen0 = np.array([0.0, 0.0]) #array with dilation and eccentricity for vertex zero
    for sim_index in simplex_list[simplex[0]]:
        
        sim = simplices[sim_index]
        
        Dil_Eccen0 +=  get_simplex_distortion(init_vertices[sim], final_vertices[sim])
        
    dilation0, eccentricity0 = Dil_Eccen0/len(simplex_list[simplex[0]])             
    
    
    Dil_Eccen1 = np.array([0.0, 0.0]) #array with dilation and eccentricity for vertex zero
    for sim_index in simplex_list[simplex[1]]:
        
        sim = simplices[sim_index]
        
        
        Dil_Eccen1 +=  get_simplex_distortion(init_vertices[sim], final_vertices[sim])
        
    dilation1, eccentricity1 = Dil_Eccen1/len(simplex_list[simplex[1]])   

    Dil_Eccen2 = np.array([0.0, 0.0]) #array with dilation and eccentricity for vertex zero
    for sim_index in simplex_list[simplex[2]]:
        
        sim = simplices[sim_index]
        
        Dil_Eccen2 +=  get_simplex_distortion(init_vertices[sim], final_vertices[sim])
        
    dilation2, eccentricity2 = Dil_Eccen2/len(simplex_list[simplex[2]])    


    average_dilation = (dilation0 + dilation1 + dilation2)/3.0
    average_eccen = (eccentricity0 + eccentricity1 + eccentricity2)/3.0
    
    dilation_grad = np.array([dilation1 - dilation0, dilation2 - dilation0])
    eccen_grad = np.array([eccentricity1 - eccentricity0, eccentricity2 - eccentricity0])
    
    
    g = get_simplex_metric(init_vertices[simplex])
    
    
    cost +=  (a * (average_dilation - 1)**2 + b * (average_eccen - 1)**2 + c * np.einsum(
             'i, ij, j', dilation_grad, g, dilation_grad) + d * np.einsum('i, ij, j', eccen_grad, g, eccen_grad))*np.sqrt(la.det(g))
    
    
#=========================================================================================================================
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================================



#=========================================================================================================================
# returns a metric on a triangle in a special coordinate syste,
#=========================================================================================================================
def get_simplex_metric(positions):
    '''
    a triangle with three vertices (p0, p1, p2),  will have a coordinate system (a, b) in [0, 1]^2 where 
    each point in the triangle is represented as p = p0 + a (p1 - p0) + b (p2 - p0). This method
    return the metric of the triangle in this coordinate system
    '''
    
    v1 = positions[1] - positions[0]
    v2 = positions[2] - positions[0]
    
    return np.array([[np.dot(v1,v1), np.dot(v1,v2)], [np.dot(v2,v1), np.dot(v2,v2)]])
#=========================================================================================================================
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================================    
    
    
#=========================================================================================================================
# returns the shear and dilation for each triangle
#=========================================================================================================================
def get_simplex_distortion(init_positions, final_positions):
    '''
    This will give the shear and dilation and initial area for each triangle. The triangle is 
    represented in the original and final surface.
    
    init_positions: 
    array with shape (3,3) of the positions of the vertices of the triangle on the first surface.
     
    final_positions: 
    array with shape (3,3) of the positions of the vertices of the triangle on the second surface.
    '''
    
    
    vb1 = init_positions[1] - init_positions[0]
    vb2 = init_positions[2] - init_positions[0]
    
    v1 = final_positions[1] - final_positions[0]
    v2 = final_positions[2] - final_positions[0]
    
    gb = np.array([[np.dot(vb1,vb1), np.dot(vb1,vb2)], [np.dot(vb2,vb1), np.dot(vb2,vb2)]])
    
    g = np.array([[np.dot(v1,v1), np.dot(v1,v2)], [np.dot(v2,v1), np.dot(v2,v2)]])
    
    g_tilde = np.dot(la.inv(gb), g)  
    
    dilation = 0.5*(np.trace(g_tilde)**2 - np.trace(np.dot(g_tilde, g_tilde)))  # this gives the square of the dilation
    
    dY = np.sqrt((2*np.trace(np.dot(g_tilde, g_tilde))/np.trace(g_tilde)**2) - 1)
    
    eccentricity = (1 + dY)/(1 - dY) # this gives the square of the eccentricity
    
    
    return  np.array([dilation, eccentricity])
#=========================================================================================================================
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================================
  

#=========================================================================================================================
# get the list of simplices each vertex belongs to
#=========================================================================================================================    
def get_simplex_list(simplices, num_verts):
    '''
    From a list of simplices find the simplex list of every vertex.
    In other words, the triangles it belongs to.
    '''

    tri_list = np.empty((num_verts,1)).tolist()
    for i in range(num_verts):
        tri_list[i].pop()
    
    for tri_index, tri in enumerate(simplices):
        for vert_index in tri:
            tri_list[vert_index].append(tri_index)
    
    return tri_list
#=========================================================================================================================
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================================


#=========================================================================================================================
# a smooth representation of the final surface used as a constraint
#=========================================================================================================================  
from numpy import cos, sin
def final_surface(coordinates):
    '''
    
    
    '''
    #defining a surface with spherical harmonics
    phi, theta = coordinates 
    
    radius = np.abs(1 + sph_harm(1, 1, phi, theta))
    
    return np.array([sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)])*radius
#=========================================================================================================================
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#=========================================================================================================================

def get_spherical_coods(position):
    x, y, z = position
    
    theta =  np.arccos(z/np.sqrt(x**2 + y**2 + z**2))
    phi =  np.arctan2(y, x) 
    
    return np.array([phi, theta])
    

   
from time import *


init_vertices = np.loadtxt(mesh_dir / "vertices.txt")
simplices = np.loadtxt(mesh_dir / "faces.txt", dtype=np.int64) - 1

simplex_list = get_simplex_list(simplices, init_vertices.shape[0])
    
final_coordinates = np.array(list(map(get_spherical_coods, init_vertices)))
    
print(final_coordinates.shape)
    
    


t = clock() 

cost_function(final_coordinates, init_vertices, simplices, simplex_list)
t = clock() - t

print(t)






























    