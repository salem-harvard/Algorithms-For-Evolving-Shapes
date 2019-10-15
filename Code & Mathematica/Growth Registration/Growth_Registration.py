# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 07:30:45 2018

@author: Salem

TODO - Check whether nearby tangent vectors and normal displacements are small. 
TODO - Check whether solutions are close to the desired surface.
"""

import numpy as np
import numpy.linalg as la
import numpy.random as npr 
   

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import scipy.optimize as op
#start with deforming the plane, and use height representation
# Triangulate the first surface. 

from pathlib import Path
import os
cwd = Path(os.getcwd())


mesh_dir = cwd / "Mesh_Data" 

print(mesh_dir)

def get_mesh_names():
    mesh_names = []
    mesh_number_name = []
   
    
    for (dirpath, dirnames, filenames) in os.walk(mesh_dir):
        mesh_names.extend(dirnames)
        break
    
    for i, name in enumerate(mesh_names):
        mesh_number_name.append(str(i + 1) + " --> " + name)
    
    return mesh_names, mesh_number_name

def load_data(mesh_num = 1, ask_for_mesh_name=True, isOptimizing=True):
    mesh_names, mesh_number_names = get_mesh_names()
    print("\n " , mesh_number_names)
    
    if ask_for_mesh_name:    
        mesh_num = int(input("select a number between 1 and " + str(len(mesh_names))  +
                         " for the mesh you want to run:  "))
                             
    mesh_name = mesh_names[mesh_num - 1]
    
    print("loading data for: ", mesh_name, " ...")
    
    vertices = np.loadtxt(mesh_dir / mesh_name  / "vertices.txt")
    print("shape of vertex array:", vertices.shape)
    
    faces = np.loadtxt(mesh_dir / mesh_name  / "faces.txt", dtype=np.int64) - 1
    normals = np.loadtxt(mesh_dir  / mesh_name  / "normal.txt")
    
    tangent_x = np.loadtxt(mesh_dir  / mesh_name / "principal_dir1.txt")
    tangent_y = np.loadtxt(mesh_dir  / mesh_name / "principal_dir2.txt")
    
    #position_change = np.loadtxt(mesh_dir  / mesh_name / "position_change.txt")
    normal_disps = np.loadtxt(mesh_dir  / mesh_name  / "distance.txt")
    
    #normal_disps = 0.1*npr.rand(vertices.shape[0])
    
    curv_x = np.loadtxt(mesh_dir  / mesh_name  / "principal_curv1.txt")
    curv_y = np.loadtxt(mesh_dir  / mesh_name  / "principal_curv2.txt")
    
    verts_deformed = np.loadtxt(mesh_dir / mesh_name  / "vertices_deformed.txt")
    
    gaussC = curv_x*curv_y; gaussC = np.sum(gaussC[faces],axis=1)/3
    meanC = 0.5*(curv_x + curv_y); meanC = np.sum(meanC[faces],axis=1)/3
    
    
    np.savetxt(cwd / "verts_deformed.npy" , verts_deformed, delimiter=',')   
    np.savetxt(cwd / "gauss_curv.npy" , gaussC, delimiter=',')   
    np.savetxt(cwd / "mean_curv.npy" , meanC, delimiter=',')   
    np.savetxt(cwd / "normal_disps.npy" , normal_disps, delimiter=',')   
    np.savetxt(cwd / "tangent_x.npy" , tangent_x, delimiter=',') 
    np.savetxt(cwd / "tangent_y.npy" , tangent_y, delimiter=',') 
    np.savetxt(cwd / "normals.npy" , normals, delimiter=',') 
    
    
    landmark_indices = np.loadtxt(mesh_dir / mesh_name  / "landmark.txt", dtype=np.int64) - 1
    print("landmark indices:", landmark_indices)
    
    #landmark_indices = []#np.hstack((np.arange(700), -np.arange(700) -1))
    landmark_disps = np.loadtxt(mesh_dir  / mesh_name  / "landmark_displacement.txt")
    #landmark_disps = np.zeros(3)#verts_deformed[landmark_indices] -  vertices[landmark_indices]
    
    res = optimal_map(vertices, faces, normals, tangent_x, tangent_y, normal_disps, landmark_indices, landmark_disps, isOptimizing)
    
    #areal_defs, eccentricities = get_deformation(vertices, position_change, faces)
    areal_defs, eccentricities = res[0], res[1]
    
    #np.savetxt(mesh_dir  + mesh_name  + "\\areal_defs.npy" , areal_defs, delimiter=',') 
    #np.savetxt(mesh_dir  + mesh_name  + "\\eccentricities.npy" , eccentricities, delimiter=',') 
    #np.savetxt(mesh_dir  + mesh_name  + "\\vertices.npy" , vertices, delimiter=',') 
    #np.savetxt(mesh_dir  + mesh_name  + "\\faces.npy" , faces, delimiter=',') 
    #np.savetxt(mesh_dir  + mesh_name + "\\gauss_curv.npy" , gaussC, delimiter=',')   
    #np.savetxt(mesh_dir  + mesh_name + "\\mean_curv.npy" , meanC, delimiter=',')   
    
    x_disps = res[2][::2]
    y_disps = res[2][1::2]
    
    print("\n max x-deviation (x_disps * curv_x): " , max(x_disps*curv_x))
    print("\n max y-deviation (y_disps * curv_y): ", max(y_disps*curv_y))
    print("\n mean x-deviation (x_disps * curv_x): " ,np.mean(x_disps*curv_x))
    print("\n mean y-deviation (y_disps * curv_y): " ,np.mean(y_disps*curv_y))
    
    return gaussC, meanC, areal_defs, eccentricities, res[3]



from time import *
def optimal_map(verts, tris, n, et, ef, normal_disps, landmark_indices, landmark_disps, isOptimizing):
    
    # make initial hump triangulartion and vertices. 
    #radius =1
    #verts, tris = triangulate_sphere(radius)
    
    num_verts = verts.shape[0]
    #initialize the tangent displacements to zero
    init_tangent_disps = np.zeros(2*num_verts)
    
    #set the normal displacements
    #normal_disps = 0.01*np.ones(num_verts)
    #normal_disps = 0.2*set_normal_disps(verts,5,5)
    
    #n, et, ef = sphere_ortho_basis(verts)
    
    tri_list = get_tri_list(tris,num_verts)
    
    #calculate the displacement and areal deformation for the initial tangent displacements
    init_disps =  (init_tangent_disps[::2].reshape((num_verts, 1)) * et + 
      init_tangent_disps[1::2].reshape((num_verts, 1)) * ef + 
       normal_disps.reshape((num_verts, 1)) * n)
    
    init_costs = check_cost(verts, init_disps, tris)
    
    np.savetxt(cwd / "init_areal_defs.npy" , init_costs, delimiter=',')
    np.savetxt(cwd  / "init_verts.npy" , verts, delimiter=',')
    
    #print(grad_of_cost(init_tangent_disps,normal_disps, verts, tris,tri_list, n, et, ef, num_verts))
    #error = op.check_grad(cost, grad_of_cost, init_tangent_disps,normal_disps, verts, tris,tri_list, n, et, ef, num_verts)
    #print(error)
    
    print("optimizing using 'Newton-CG' method ...")

    
    if (isOptimizing):
        res = op.minimize(cost, init_tangent_disps, args=(
                normal_disps, verts, tris,tri_list, n, et, ef, num_verts, landmark_indices, landmark_disps),
                      jac=grad_of_cost, hess=hess_of_cost, method='Newton-CG')
        tangent_disps = res.x
    else:
        tangent_disps = init_tangent_disps
    
    
    
    #This part is uncommented only if we are not optimizing.
    #tange
    
    
    new_disps =  (tangent_disps[::2].reshape((num_verts, 1)) * et + 
      tangent_disps[1::2].reshape((num_verts, 1)) * ef + 
       normal_disps.reshape((num_verts, 1)) * n)
    
    new_verts  = verts + new_disps
    
    areal_defs, eccentricities = get_deformation(verts, new_disps, tris)
    
    np.savetxt(cwd  / "areal_defs.npy" , areal_defs, delimiter=',')
    
    np.savetxt(cwd  / "eccentricities.npy" , eccentricities, delimiter=',')
    
    costs = check_cost(verts, new_disps, tris)
    
    np.savetxt(cwd  / "verts.npy" , new_verts, delimiter=',')
    np.savetxt(cwd / "tris.npy" , tris, delimiter=',')
    np.savetxt(cwd / "costs.npy" , costs, delimiter=',')
    return areal_defs, eccentricities, tangent_disps, costs
    

from copy import deepcopy
def cost(tangent_disps, normal_disps, verts, tris, tri_list, normals,
         e_theta, e_fi, num_verts, landmark_indices=[], landmark_disps=np.zeros(3), a=1, b=2) :
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

def grad_of_cost(tangent_disps, normal_disps, verts, simplices, tri_list,  normals, 
                 e_theta, e_fi, num_verts, landmark_indices=[],landmark_disps=np.zeros(3), isHess=False, a=1,b=2):
    '''
    '''
    
    disps = (tangent_disps[::2].reshape((num_verts, 1)) * e_theta + 
      tangent_disps[1::2].reshape((num_verts, 1)) * e_fi + 
       normal_disps.reshape((num_verts, 1)) * normals)
    
    
    if(isHess):
        grad = np.zeros((2*num_verts, 2, 2))
    else:
        disps[landmark_indices] = landmark_disps
        grad = np.zeros(2*num_verts)
    
    
    for vert_index, tri_indices in enumerate(tri_list):
        for tri_index in tri_indices:
            simplex = simplices[tri_index]
            Y = np.abs(simplex - vert_index)
            new_simplex = [x for _,x in sorted(zip(Y,simplex))]
            grad_et, grad_ef = grad_per_simplex(verts[new_simplex], disps[new_simplex], 
                                                e_theta[vert_index], e_fi[vert_index],a=a,b=b,isHess=isHess)
            
            grad[2*vert_index] += grad_et
            grad[2*vert_index + 1] += grad_ef
        
    
    return grad

def hess_of_cost(tangent_disps, normal_disps, verts, simplices, tri_list, normals,
                 e_theta, e_fi, num_verts, landmark_indices=[],landmark_disps=np.zeros(3), a=1,b=2):
    '''
    '''
  
    if (len(landmark_indices) > 0):
        tangent_disps[::2][landmark_indices] = np.einsum('ij,ij->i',landmark_disps, e_theta[landmark_indices])
        tangent_disps[1::2][landmark_indices] = np.einsum('ij,ij->i',landmark_disps,e_fi[landmark_indices])
        normal_disps[landmark_indices] = np.einsum('ij,ij->i',landmark_disps, normals[landmark_indices])
    
    
    hess = np.zeros((2*num_verts, 2*num_verts))
    
    
    for simplex in simplices:
        extened_simplex = [2*simplex[0], 2*simplex[0] + 1, 2*simplex[1], 2*simplex[1] + 1,
                                           2*simplex[2], 2*simplex[2] + 1]
        
        sim_tangent_disps = tangent_disps[extened_simplex]
        sim_normal_disps = normal_disps[simplex]
        sim_verts = verts[simplex]
        sim_tri_list = [[0],[0],[0]]
        sim_normals = normals[simplex]
        sim_et = e_theta[simplex]
        sim_ef = e_fi[simplex]
        sim_num_verts = 3
        
        sim_grad = grad_of_cost(sim_tangent_disps, sim_normal_disps, sim_verts, 
                                np.array([[0,1,2]]), sim_tri_list,  sim_normals, 
                                                sim_et, sim_ef, sim_num_verts, b=0, a=a)
        
        sim_cost = cost(sim_tangent_disps, sim_normal_disps, sim_verts, 
                        np.array([[0,1,2]]), sim_tri_list,  sim_normals, 
                                                sim_et, sim_ef, sim_num_verts,b=0,a=1)
        
        #area = simplex_area(sim_verts)
        
        sim_hess =np.outer(sim_grad, sim_grad)/(2*sim_cost)
        
        
        hess[np.ix_(extened_simplex,extened_simplex)] += sim_hess
        
        sim_grad = grad_of_cost(sim_tangent_disps, sim_normal_disps, sim_verts, 
                                np.array([[0,1,2]]), sim_tri_list,  sim_normals, 
                                sim_et, sim_ef, sim_num_verts, isHess=True, a=0,b=b)
        
        
        #area = simplex_area(sim_verts)
        
        sim_hess = b*np.einsum('ijk, lkj', sim_grad, sim_grad)/2
        
        
        hess[np.ix_(extened_simplex,extened_simplex)] += sim_hess
        
    return hess

    
import numdifftools as nd
def test_grad():
    # make initial hump triangulartion and vertices. 
    verts = np.array([[0,0.5,0],[1.0,0,0.5], [0,1,0]])
    tris =  np.array([[0,1,2]])
    
    num_verts = verts.shape[0]
    
    #initialize the tangent displacements to zero
    init_tangent_disps = np.zeros(2*num_verts)
    
    #set the normal displacements
    #normal_disps = 0.01*np.ones(num_verts)
    normal_disps = 0.01*np.array([1,1,1])
    
    n = np.array([[-1,0.0,0],[0,0,1], [0,1,0]])
    et = np.array([[0,-1,0],[0,-1,0], [1,0.0,0]])
    ef = np.array([[0,0.0,1],[1,0,0], [0,0.0,1]])
    
    
    tri_list = get_tri_list(tris,num_verts)
    
    print("cost:", cost(init_tangent_disps, normal_disps, verts, tris,tri_list, n, et, ef, num_verts), "\n\n")
    print(grad_of_cost(init_tangent_disps, normal_disps, verts, tris,tri_list, n, et, ef, num_verts), "\n\n")
    print(op.approx_fprime(init_tangent_disps, cost, 0.00001, normal_disps, 
                           verts, tris,tri_list, n, et, ef, num_verts), "\n\n")
    print(hess_of_cost(init_tangent_disps, normal_disps, verts, tris,tri_list, n, et, ef, num_verts,b=0), "\n\n")
    
    
    return hessian_approx(init_tangent_disps, normal_disps, verts, tris,tri_list, n, et, ef, num_verts, b=0)

def cost_per_simplex(init_positions, displacements, a=1, b=2, c=0):
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
    #  + c * np.dot(n - nb, n - nb))


def grad_per_simplex(init_positions, displacements, et, ef, a=1, b=2, isHess=False):
   # Y = np.abs(simplex - vert_index)
    #new_simplex = [x for _,x in sorted(zip(Y,simplex))]
    vb1 = init_positions[1] - init_positions[0]
    vb2 = init_positions[2] - init_positions[0]
    
    deltaV1 = displacements[1] - displacements[0]
    deltaV2 = displacements[2] - displacements[0]
    
    gb = np.array([[np.dot(vb1,vb1), np.dot(vb1,vb2)], [np.dot(vb2,vb1), np.dot(vb2,vb2)]])
    
    deltaG = np.array([[2*np.dot(vb1, deltaV1), np.dot(vb1,deltaV2) + np.dot(vb2,deltaV1)],
                       [np.dot(vb1,deltaV2) + np.dot(vb2,deltaV1), 2*np.dot(vb2, deltaV2)]])
    
    area = np.sqrt(gb[0,0] * gb[1,1] - gb[1,0]*gb[0,1])
    
    strain = np.dot(la.inv(gb), deltaG)
    
    disp_grad_et =0.5* np.dot(la.inv(gb), [[-2*np.dot(vb1,et), -np.dot(vb1,et)-np.dot(vb2,et)], 
                          [-np.dot(vb1,et)-np.dot(vb2,et), -2*np.dot(vb2,et)]])
    disp_grad_ef =0.5* np.dot(la.inv(gb), [[-2*np.dot(vb1,ef), -np.dot(vb1,ef)-np.dot(vb2,ef)], 
                          [-np.dot(vb1,ef)-np.dot(vb2,ef), -2*np.dot(vb2,ef)]])
    
    grad_et = area * (a * np.trace(strain) * np.trace(disp_grad_et)) / 2
    grad_ef = area * (a * np.trace(strain) * np.trace(disp_grad_ef)) / 2
    
    grad_et += area * (b * np.trace(np.dot(strain, disp_grad_et))) / 2
    grad_ef += area * (b * np.trace(np.dot(strain, disp_grad_ef))) / 2
    
    if(isHess):
        return np.sqrt(area)*disp_grad_et, np.sqrt(area)*disp_grad_ef
    
    return grad_et, grad_ef    


def simplex_area(positions):
    vb1 = positions[1] - positions[0]
    vb2 = positions[2] - positions[0]

    gb = np.array([[np.dot(vb1,vb1), np.dot(vb1,vb2)], [np.dot(vb2,vb1), np.dot(vb2,vb2)]])
    
    return np.sqrt(gb[0,0] * gb[1,1] - gb[1,0]*gb[0,1])

def Gaussian_hump(position, peak_pos = np.array([0,0]), amplitude = 1,  width = 0.4):
    '''
    position: gives the coordinates (x,y) on the plane.
    amplitude: The size of the gaussian hump.
    width: gives the width (radius, variance) of the hump.
    '''
    x = position[0] - peak_pos[0]
    y = position[1] - peak_pos[1]
    return np.hstack((position, amplitude*np.exp(-(x**2 + y**2)/width**2)))


def sphere(position, radius = 1):
    '''
    position: gives the coordinates (x,y) on the plane.
    radius: The size of the sphere.
    
    ''' 
    if np.sum(position**2) > 1:
        print(position)
        print(np.sum(position**2))
    return np.hstack((position, np.sqrt(1 - np.sum(position**2))))

def triangulate_sphere(radius=1):
    x=np.linspace(- radius,  radius,25)
    y=np.linspace(- radius,  radius, 25)
    x,y=np.meshgrid(x,y)
    
    x=x.flatten()
    y=y.flatten()

    #define 2D points, as input data for the Delaunay triangulation of U
    points2D=np.vstack([x, y]).T
    
    new_array = []
    #remove points outside the sphere
    for point in points2D:
        if la.norm(point) < 0.99*radius:
            new_array.append(point)
            
    points2D = np.array(new_array)        
    #print(points2D)
    points3D = np.array([sphere(pos, radius=radius) for pos in points2D])
    #np.array(list(map(Gaussian_hump, points2D)))
    
    return  points3D, Delaunay(points2D).simplices#triangulate the rectangle U

def triangulate_hump(peak_pos, domain_size = 1.2):
    u=np.linspace(-domain_size,domain_size, 15)
    v=np.linspace(-domain_size,domain_size, 15)
    u,v=np.meshgrid(u,v)
    
    u=u.flatten()
    v=v.flatten()

    #define 2D points, as input data for the Delaunay triangulation of U
    points2D=np.vstack([u,v]).T
    #print(points2D)
    points3D = np.array([Gaussian_hump(x, peak_pos=peak_pos) for x in points2D])
    #np.array(list(map(Gaussian_hump, points2D)))
    
    return  points3D, Delaunay(points2D).simplices#triangulate the rectangle U

def sphere_ortho_basis(point3D):
    '''
    takes in the points in 3d and finds the local orthonormal frame for each point.
    It will return theta hat phi hat and r hat. 
    
    points3D: an array of 3-vectors
    '''
    
    # get the spherical coordinates of the points
    theta = np.arctan2(point3D[:, 1], point3D[:, 0])
    phi =  np.arcsin(la.norm(point3D[:, :2], axis=1)/ la.norm(point3D, axis=1))
    
    normals = np.vstack((np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi))).T
    
    e_theta = np.vstack((-np.sin(theta), np.cos(theta), np.zeros_like(theta))).T
                       
    e_phi = np.vstack((np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), -np.sin(phi))).T                  
    
    return normals, e_theta, e_phi
    

import scipy.special as sps
def set_normal_disps(point3D, m, n):
    # get the spherical coordinates of the points
    theta = np.arctan2(point3D[:, 1], point3D[:, 0])
    phi =  np.arcsin(la.norm(point3D[:, :2], axis=1)/ la.norm(point3D, axis=1))
    
    
    normal_disps = np.real(sps.sph_harm(m, n, theta, phi))
    
    return normal_disps
    
    #the normal disps will be some desired function, like a spherical harmonic
    
    
def get_tri_list(simplices, num_verts):
    '''
    From a list of simplices find the trinagle list of every vertex.
    In other words, the triangles it belongs to.
    '''

    tri_list = np.empty((num_verts,1)).tolist()
    for i in range(num_verts):
        tri_list[i].pop()
    
    for tri_index, tri in enumerate(simplices):
        for vert_index in tri:
            tri_list[vert_index].append(tri_index)
    
    return tri_list


def check_cost(verts, new_disps, tris) :
    '''
    ''' 

    # calculate the contribution from each triangle
    costs = []
    for tri in tris:        
        costs.append(cost_per_simplex(verts[tri], new_disps[tri]))
        
        
    return costs

    
def get_deformation(positions, displacements, tris):
    area_defs = []
    eccentricities = []
    directions = []
    
    for tri in tris:
        tri_positions = positions[tri]
        vb1 = tri_positions[1] - tri_positions[0]
        vb2 = tri_positions[2] - tri_positions[0]
        
        tri_displacements = displacements[tri]
    
        deltaV1 = tri_displacements[1] - tri_displacements[0]
        deltaV2 = tri_displacements[2] - tri_displacements[0]
    
        
        gb = np.array([[np.dot(vb1,vb1), np.dot(vb1,vb2)], [np.dot(vb2,vb1), np.dot(vb2,vb2)]])
    
        deltaG = np.array([[2*np.dot(vb1, deltaV1), np.dot(vb1,deltaV2) + np.dot(vb2,deltaV1)],
                       [np.dot(vb1,deltaV2) + np.dot(vb2,deltaV1), 2*np.dot(vb2, deltaV2)]])
    
    
        strain = np.dot(la.inv(gb), deltaG)
        
        eigs, vecs = la.eig(strain)
            
        direction = vecs[0,0]*vb1 + vecs[0,1]*vb2
        direction /= la.norm(direction)
        
        directions.append([tri_positions[0], direction + tri_positions[0]])
        
        area_defs.append(0.25*np.trace(strain)**2)
        # the line below removes the square on the areal strain
        #area_defs.append(0.25*np.trace(strain))
        
        eccentricities.append(0.5*np.trace(np.dot(strain - 0.5*np.trace(strain)*np.identity(2),
                                                  strain - 0.5*np.trace(strain)*np.identity(2))))
            
        
    return  np.array(area_defs), np.array(eccentricities)
    
    
    
def hessian_approx (tangent_disps, normal_disps, verts, tris, tri_list, normals, e_theta, e_fi, num_verts, a=1, b=1):
    """
    """
    
    return nd.Hessian(lambda tangent_disps: cost(tangent_disps, 
                        normal_disps, verts, tris, tri_list, normals, e_theta, e_fi, num_verts, a=a, b=b))(tangent_disps)
    
    
    
def list_to_array(list_array):
    
    Arr = list_array[0]
    for arr in list_array[1:]:
        Arr = np.hstack((Arr, arr))
        
    return Arr.flatten()
    
def collect_all_data():
    gaussCs = []
    meanCs = []
    areals = []
    eccents = []
    costs = []
    mesh_names, num_mesh_names = get_mesh_names()
    
    for i, mesh_name in enumerate(mesh_names):
        if(mesh_name.startswith( "time" )):
            x = load_data(i + 1, ask_for_mesh_name=False)
            gaussCs.append(x[0])
            meanCs.append(x[1])
            areals.append(x[2])
            eccents.append(x[3]) 
            costs.append(x[4])
        
    gaussCs = list_to_array(gaussCs)
    meanCs = list_to_array(meanCs)
    areals = list_to_array(areals)
    eccents = list_to_array(eccents)
    costs = list_to_array(costs)
    
    
    y=[]
    for costings in costs:
       y.append(costings.mean())
       
    y = np.array(y)
    y = y[[0,1,8,9,10,11,12,13,14,15,2,3,4,5,6,7]]     
    plt.plot(np.array(y))
    plt.xlabel('time step')
    plt.ylabel('average cost')
    plt.show()
    
    
    y=[]
    for costings in eccents:
       y.append(costings.mean())
       
    y = np.array(y)
    y = y[[0,1,8,9,10,11,12,13,14,15,2,3,4,5,6,7]]     
    plt.plot(np.array(y))
    plt.xlabel('time step')
    plt.ylabel('average eccentricity')
    plt.show()
    
    y=[]
    for costings in areals:
       y.append(costings.mean())
       
    y = np.array(y)
    y = y[[0,1,8,9,10,11,12,13,14,15,2,3,4,5,6,7]]     
    plt.plot(np.array(y))
    plt.xlabel('time step')
    plt.ylabel('average dilation')
    plt.show()
    
    #np.savetxt(cwd + "\\gauss_curv.npy" , gaussCs, delimiter=',')   
    #np.savetxt(cwd + "\\mean_curv.npy" , meanCs, delimiter=',')   
    #np.savetxt(cwd  + "\\areal_defs.npy" , areals, delimiter=',')
    #np.savetxt(cwd + "\\eccentricities.npy" , eccents, delimiter=',')
    
    return gaussCs, meanCs, areals, eccents, costs

def get_all_defs():
    mesh_names, num_mesh_names = get_mesh_names()
    
    for i, mesh_name in enumerate(mesh_names):
        if(mesh_name.startswith( "time" )):
            x = load_data(i + 1, ask_for_mesh_name=False)
            
    
    
    
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)   
    
    
    
    
    
    
    
    