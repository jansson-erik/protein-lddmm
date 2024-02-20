#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:40:41 2024

@author: erikjans
"""

import numpy as np


class Optimize_list:
    """
        Class to compute norm over all images and their gradients. 
    """
    def __init__(self, data, h):  
        """
        Constructor function

        Parameters
        ----------
        data : Object of type data. 
        h: float, step size. 
        """    
        self.data = data
        self.h = h
        self.M = int(1/h)
        self.u_full = np.zeros((self.M, data.N, 3, 3))
  
    def F(self,u_full,x = None):
        """
        Computes the norm over all images. 

        Parameters
        ----------
        u_full: np.array, shape = (M,N,3,3)
        x: np.array, shape = (N,3,3), optional, default = None. Protein conformation template. 
        
        Returns
        -------
        sum of norm over all images, float. 
        """
        if x is None:  
            template_model = self.data.template_model
        else:
            template_model = x
        R_full = self.euler(u_full) # compute the full group path. 
        R1 = R_full[-1] # select the last element in  group path. 
        def_model = self.data.forward_model.M(self.data.forward_model.act(R1, template_model)) # deform the template model. 
        target_projections_blur = self.data.target_projections_blur # extract the noise target images. 
        d = np.sum([np.linalg.norm(self.data.forward_model.blur(self.data.forward_model.project(def_model,proj),self.data.grid,self.data.blob_size) 
                                   -target_projections_blur[i])**2 for (i,proj) in enumerate(self.data.forward_model.projections) ])
        return d
    def grad(self,u_full,index = None):
        """
        Computes the gradient of the norms of all images. 
        
        Parameters
        ----------
        u_full: np.array, shape = (M,N,3,3)
        index: int, optional, default = None. Index of atom to extract gradient over. If none full gradient is returned. 

        Returns
        -------
        np.array, shape = (M,N,3,3)
        """
        grid = self.data.grid
        blob_size = self.data.blob_size
        template_model = self.data.template_model
        R_full = self.euler(u_full)
        R1 = R_full[-1]
        def_model = self.data.forward_model.act(R1, template_model)
        def_pc = self.data.forward_model.M(def_model)
        grad_out = 0 
        for (i,proj) in enumerate(self.data.forward_model.projections):
            
            def_pc_project = self.data.forward_model.project(def_pc,proj)
            def_pc_blur = self.data.forward_model.blur(def_pc_project,grid, blob_size)
            pi = def_pc_blur - self.data.target_projections_blur[i]
            
            grad =  self.data.forward_model.dblurstar(def_pc_project, pi, grid, blob_size)
            grad = self.data.forward_model.project_star(grad,proj)
            grad = self.data.forward_model.Mstar(grad)
            grad =  self.data.forward_model.dlxLRstar(R1,template_model, grad)
            grad =  self.data.forward_model.Adstar_alt(R_full, grad)
            grad_out += grad
        if index is None:
            return grad_out
        else: 
            return grad_out[:,index,:,:]
  
     

    def euler(self, u_full):
        """
        Computes the full group path using Euler's method. 

        Parameters
        ----------
        u_full: np.array, shape = (M,N,3,3)

        Returns
        -------
        np.array, shape = (M,N,3,3)
        """
        M = self.M
        N = self.data.N
        R_full =  np.zeros_like(u_full)
        #exp = expm(self.h*u_full)#one index too many
        for i in range(N):
            R_full[0,i,:,:] = np.eye(3)
        for i in range(1,M):
            exp = np.eye(3) + np.sin(self.h)*u_full[i-1]+(1-np.cos(self.h))*u_full[i-1]@u_full[i-1]
            R_full[i] = np.einsum('bik,bkj->bij', exp, R_full[i-1])
        return R_full
    
    
    
    
    
    