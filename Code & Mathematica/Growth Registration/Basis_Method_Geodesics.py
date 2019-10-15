# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:58:12 2018

@author: Salem

This script will calculate the "length" of a path in shape space using the basis function method. In other words,
The deformation path will be represented as a sum of linearly independent components, such as spherical harmonics.

"""
#=========================================================================================================================
# importing useful packages
#=========================================================================================================================

import numpy as np
import numpy.linalg as la
import numpy.random as npr 

import matplotlib.pyplot as plt


import time # to be used in timing methods and evaluating performance
#=========================================================================================================================
#=========================================================================================================================


#=========================================================================================================================
# define the working directory and mesh directory
#=========================================================================================================================
from pathlib import Path
import os
cwd = Path(os.getcwd())


mesh_dir = cwd / "Mesh_Data" 

print("Directory of meshes:", mesh_dir)
#=========================================================================================================================
#=========================================================================================================================


#=========================================================================================================================
# gets the names of meshes in the mesh directory and their corresponding order
#=========================================================================================================================
def get_mesh_names():
    mesh_names = []
    mesh_number_name = []
   
    
    for (dirpath, dirnames, filenames) in os.walk(mesh_dir):
        mesh_names.extend(dirnames)
        break
    
    for i, name in enumerate(mesh_names):
        mesh_number_name.append(str(i + 1) + " --> " + name)
    
    return mesh_names, mesh_number_name
#=========================================================================================================================
#=========================================================================================================================
 
#=========================================================================================================================
# Loads and saves data to be used in the calculation
#=========================================================================================================================
def load_data(mesh_num = 1, ask_for_mesh_name=True):
    '''
    
    '''
    mesh_names, mesh_number_names = get_mesh_names()
    print("\n " , mesh_number_names)
    
    if ask_for_mesh_name:    
        mesh_num = int(input("select a number between 1 and " + str(len(mesh_names))  +
                         " for the mesh you want to run:  "))
                             
    mesh_name = mesh_names[mesh_num - 1]
    
    print("loading data for: ", mesh_name, " ...")
    
    #the initial vertices to be deformed to a new surface
    init_vertices = np.loadtxt(mesh_dir / mesh_name  / "vertices.txt")
    
    #number of vertices
    num_verts = init_vertices.shape[0]
    
    #save the initial vertices
    np.savetxt(cwd  / "init_verts.npy" , init_vertices, delimiter=',')
    
    faces = np.loadtxt(mesh_dir / mesh_name  / "faces.txt", dtype=np.int64) - 1
    
    np.savetxt(cwd / "faces.npy" , faces, delimiter=',')
    
    unit_normals = np.loadtxt(mesh_dir  / mesh_name  / "normal.txt")
    
    m_index = npr.randint(0,9,9)
    
    
    tangent_field = np.zeros((num_verts, 3))
    for i in range(num_verts):
        tangent_field[i] = tangent_projection(init_vertices[i], unit_normals[i], m_index)
    
    
    np.savetxt(cwd / "tangent_field.npy" , tangent_field, delimiter=',')
    np.savetxt(cwd / "unit_normals.npy" , unit_normals, delimiter=',')
    #tangent_x = np.loadtxt(mesh_dir  / mesh_name / "principal_dir1.txt")
    #tangent_y = np.loadtxt(mesh_dir  / mesh_name / "principal_dir2.txt")
    
    
    #position_change = np.loadtxt(mesh_dir  / mesh_name / "position_change.txt")
    #normal_disps = np.loadtxt(mesh_dir  / mesh_name  / "distance.txt")
    
    #normal_disps = 0.1*npr.rand(vertices.shape[0])
    
    #curv_x = np.loadtxt(mesh_dir  / mesh_name  / "principal_curv1.txt")
    #curv_y = np.loadtxt(mesh_dir  / mesh_name  / "principal_curv2.txt")
    
    #verts_deformed = np.loadtxt(mesh_dir / mesh_name  / "vertices_deformed.txt")
    
    #gaussC = curv_x*curv_y; gaussC = np.sum(gaussC[faces],axis=1)/3
    #meanC = 0.5*(curv_x + curv_y); meanC = np.sum(meanC[faces],axis=1)/3
    
    
    
    #landmark_indices = np.loadtxt(mesh_dir / mesh_name  / "landmark.txt", dtype=np.int64) - 1
    #print(landmark_indices)
    
    #landmark_disps = np.loadtxt(mesh_dir  / mesh_name  / "landmark_displacement.txt")
    
    #num_verts = vertices.shape[0]
    
    #tangent_disps =  0.01*(npr.rand(2*num_verts) - 0.5)
    
    #face_list = get_face_list(faces, num_verts)
    
    
   # start = time.perf_counter()
   # C = cost_function(tangent_disps, normal_disps, vertices, faces, face_list, unit_normals, tangent_x, tangent_y,
    #     num_verts)
   # end = time.perf_counter()
   # print(end - start)   
   # print(normal_disps.shape)
   # print("number of vertices:", num_verts)
   # print("number of faces: ", faces.shape[0])
    return
#=========================================================================================================================
#=========================================================================================================================
    

#=========================================================================================================================
# Energy of a path connecting two parametrized surfaces
#=========================================================================================================================    
def cost_function(tangent_disps, normal_disps, verts, tris, tri_list, normals,
         e_theta, e_fi, num_verts, landmark_indices=[], landmark_disps=np.zeros(3), a=1, b=1) :
    '''
    '''
    
    disps = (tangent_disps[::2].reshape((num_verts, 1)) * e_theta + 
      tangent_disps[1::2].reshape((num_verts, 1)) * e_fi + 
       normal_disps.reshape((num_verts, 1)) * normals)
    
   
    disps[landmark_indices] = landmark_disps
    
    # calculate the contribution from each triangle
    cost = 0
    for tri in tris:        
        cost += cost_per_simplex(verts[tri], disps[tri], a=a, b=b)
                
    return cost
#=========================================================================================================================
#=========================================================================================================================


#=========================================================================================================================
# cost per simplex
#=========================================================================================================================    
def cost_per_simplex(init_positions, displacements, a=1, b=1, c=0):
    '''
    This will give the contribution to the cost (or metric) from each simplex
    
    init_positions: 
    List with shape (3,3) of the positions of the vertices of the triangle on 
    the original plane.
    
    displacements: 
    List with shape (3,3) of the changes in positions of the vertices of the triangle on 
    the deformed surface
    
    a: is the coefficient of the area preserving changes. 
    b: is the coefficient of area changes
    c: is the coefficient of the bending (change in normal) term
    '''
    
    
    vb1 = init_positions[1] - init_positions[0]
    vb2 = init_positions[2] - init_positions[0]
    
    deltaV1 = displacements[1] - displacements[0]
    deltaV2 = displacements[2] - displacements[0]
    
    gb = np.array([[np.dot(vb1,vb1), np.dot(vb1,vb2)], [np.dot(vb2,vb1), np.dot(vb2,vb2)]])
    
    deltaG = np.array([[2*np.dot(vb1, deltaV1), np.dot(vb1,deltaV2) + np.dot(vb2,deltaV1)],
                       [np.dot(vb1,deltaV2) + np.dot(vb2,deltaV1), 2*np.dot(vb2, deltaV2)]])
    
    area = np.sqrt(gb[0,0] * gb[1,1] - gb[1,0]*gb[0,1])
    
    strain = np.dot(la.inv(gb), deltaG)
    
    #the normal vetor
    #n = np.cross(v1, v2)/la.norm(np.cross(v1, v2))
    #nb = np.cross(vb1, vb2)/area
    
    return area *  (a * np.trace(strain)**2 + b * np.trace(np.dot(strain,strain)))/ 8
#=========================================================================================================================
#=========================================================================================================================


#=========================================================================================================================
# get the triangle list of every vertex as a python list
#=========================================================================================================================
def get_face_list(simplices, num_verts):
    '''
    From a list of simplices find the trinagle list of every vertex.
    In other words, the triangles it belongs to. This will return a python list contaning num_verts lists,
    each of which contains the indices of the triangles the vertex belongs to
    '''

    tri_list = np.empty((num_verts,1)).tolist()
    for i in range(num_verts):
        tri_list[i].pop()
    
    for tri_index, tri in enumerate(simplices):
        for vert_index in tri:
            tri_list[vert_index].append(tri_index)
    
    return tri_list
#=========================================================================================================================
#=========================================================================================================================


#=========================================================================================================================
# project 3D vector fields onto a surface
#=========================================================================================================================
def tangent_projection(vert_position, vert_normal, m_index, box_size = np.pi):
    '''
    This gives a vector field (ux(x,y,z), uy(x,y,z), uz(x,y,z)) inside the box [-pi, pi]^3 for each Fourier index, m_index. 
    Along the x direction and on the x-boundaries, ux = 0. Similarly for y, z directions.
    '''
    
    R3_field = R3_basis_fields(vert_position, m_index)
    
    tangent_field = R3_field - np.dot(vert_normal, R3_field)*vert_normal
    
    return tangent_field
#=========================================================================================================================
#=========================================================================================================================

#=========================================================================================================================
# returns a given basis vector field in three dimensions corresponding to the multi-index m_index
#=========================================================================================================================
def R3_basis_fields(position, m_index, box_size = np.pi):
    '''
    This gives a vector field (ux(x,y,z), uy(x,y,z), uz(x,y,z)) inside the box [-pi, pi]^3 for each Fourier index, m_index. 
    Along the x direction and on the x-boundaries, ux = 0. Similarly for y, z directions.
    '''
    x,y,z = position
    
    ux = np.sin(m_index[0]*x)*np.sin(m_index[1]*y/2.0)*np.sin(m_index[2]*z/2.0)
    uy = np.sin(m_index[3]*x/2.0)*np.sin(m_index[4]*y)*np.sin(m_index[5]*z/2.0)
    uz = np.sin(m_index[6]*x/2.0)*np.sin(m_index[7]*y/2.0)*np.sin(m_index[8]*z)
    
    return np.array([ux, uy, uz])
#=========================================================================================================================
#=========================================================================================================================

#=========================================================================================================================
# returns a given basis vector field in three dimensions corresponding to the multi-index m_index
#=========================================================================================================================
def diffeomorphism(point, coeffs):
    '''
    This gives a vector field (y1(u1, u2), y2(u1, u2)) corresponding to a coodrinate transformation. 
    '''
    u1, u2 = point
    y1 = np.sum(coeffs*np.sin(u1*np.pi*np.arange(10)))
    y2 = np.sum(coeffs*np.sin(u1*np.pi*np.arange(10)))
    return 
#=========================================================================================================================
#=========================================================================================================================


#=========================================================================================================================
# returns the index of the point closest (euclidean distance) to the given node among an array of nodes
#=========================================================================================================================
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)   
#=========================================================================================================================
#=========================================================================================================================



#TODO test the speed of map with lambda vs a loop