import numpy as np

import gauss
from scipy import ndimage
from scipy.stats import special_ortho_group

class Forward:
    """
    Forward model class. This stores the forward model and the adjoint functions. 
    """
    def __init__(self,projections = None): 
        """
        Initializes the forward model. If no projections are given, the forward model is initialized with x, y, z projections. If an integer is given, 
        random projections are generated. If a list of 3x3  rotation matrices is given, the forward model is initialized with these projections. 

        Parameters
        ----------
        projections : list of 3x3 rotation matrices OR an integer, optional. Defaults to None.
        """
        if projections is None:
            self.projections = [np.array([[0,1.0,0.0],[0,0,1.0],[1,0,0]]), np.array([[1,0,0.0],[0,0,1],[0,-1,0]]),np.eye(3)]
        elif isinstance(projections,int):
            self.projections = [special_ortho_group.rvs(3) for _ in range(projections)]
        else:
            self.projections = projections
    def M(self,x):
        """
        Computes the change of coordinates into point clouds. 

        Parameters
        ----------
        x : array of shape (N, 3), the chain coordinates to be transformed into point cloud. 

        Returns
        -------
        p : array of shape (N, 3), the transformed coordinates.
        """
        N = x.shape[0]
        p = np.zeros((N, 3))
        sum = np.zeros(3)
        for i in range(N):
            sum = sum + x[i]
            p[i] = sum
        return p
    
    def invM(self,p):
        """
        Inverse of the change of coordinates into point clouds. Takes point cloud to chain coordinates. 

        Parameters
        ----------
        p : array of shape (N, 3), the chain coordinates to be transformed into point cloud.

        Returns
        -------
        x : array of shape (N, 3), the transformed coordinates.
        """
        N = p.shape[0]
        x = np.zeros((N, 3))
        x[0] = p[0]
        for i in range(1, N):
            x[i] = p[i] - p[i-1]
        return x
    
    def Mstar(self,p):
        """
        Transpose of the change of coordinates into point clouds. Takes point cloud to chain coordinates. 

        Parameters
        ----------
        p : array of shape (N, 3), the chain coordinates to be transformed into point cloud.

        Returns
        -------
        x : array of shape (N, 3), the transformed coordinates.
        """
        N = p.shape[0]
        x = np.zeros((N, 3))
        sum = np.zeros(3)
        for i in range(N):
            sum = sum + p[-1-i]
            x[-1-i] = sum
        return x
    
    def Adstar(self,R,v):
        """
        Old function. Remove after testing. 
        """
        # Calculates R^TvR batchwise
        # Do more efficiently by representing lie algebra element in R3
        return np.einsum('bki,bkl,blj->bij', R, v, R)
    
    def Adstar_alt(self,Rt,v):
        """
        Computes the coadjoint action batchwise. 

        Parameters
        ----------
        Rt : array of shape (N, 3, 3).
        v : array of shape (N, 3).

        Returns
        -------
        array of shape (N, 3,3).
        """
        # Calculates RvR^T batchwise
        # Do more efficiently by representing lie algebra element in R3
        return np.einsum('tbik,bkl,tbjl->tbij', Rt, v, Rt)

    def dlxLRstar(self,R,x,y):
        """
        Computes the adjoint of the translations. 

        Parameters
        ----------
        R : array of shape (N, 3, 3).
        x : array of shape (N, 3).
        y : array of shape (N, 3).

        Returns
        -------
        array of shape (N, 3).

        """
        t1 = np.einsum('bi,bk,bkj->bij', x, y, R)
        t2 = np.einsum('bji->bij', t1)
        return (t2 - t1)/2

    def project(self,p, rotation):#p is Nx3 and R is 3x3
        """
        Projects the point p into the plane defined by the rotation matrix R. 

        Parameters
        ----------
        p : array of shape (N, 3), the chain coordinates to be transformed into point cloud.
        rotation : array of shape (3, 3), the rotation matrix.

        Returns
        -------
        x : array of shape (N, 2), the projected coordinates.
        """
        #For projection in xz-plane take np.array([[1,0,0.0],[0,0,1],[0,-1,0]])
        #For projection in yz-plane take np.array([[0,0,-1.0],[0,1,0],[1,0,0]])
        #xy-plane is just the identity
        Rp = np.einsum('ij,dj->di', rotation, p)
        PRp = Rp[:,:2]
        return PRp

    def project_star(self,p, rotation):#p is Nx2 and R is 3x3
        """
        Computes the adjoint of the projection. 

        Parameters
        ----------
        p : array of shape (N, 2), the chain coordinates to be transformed into point cloud.
        rotation : array of shape (3, 3), the rotation matrix.

        Returns
        -------
        array of shape (N, 3), the transformed coordinates.
        """
        PTp = np.hstack((p, np.zeros((p.shape[0], 1))))
        RTPTp = np.einsum('ji,dj->di', rotation, PTp)
        return RTPTp


    def Pz(self,p):
        """
        Old function. Remove after testing. 
        """
        return p[:, :2].copy()
    
    def Pzstar(self,q):
        """
        Old function. Remove after testing. 
        """
        p = np.hstack((q, np.zeros((q.shape[0], 1))))
        return p
    
    def Px(self,p):
        """
        Old function. Remove after testing. 
        """        
        return p[:, 1:].copy() 
    
    def Pxstar(self,q):
        """
        Old function. Remove after testing. 
        """       
        return np.hstack((np.zeros((q.shape[0], 1)), q))
    
    def Py(self,p):
        """
        Old function. Remove after testing. 
        """        
        return p[:,np.setdiff1d(np.arange(3),1)].copy()
    
    def Pystar(self,q):
        """
        Old function. Remove after testing. 
        """        
        return np.insert(q,1,values = 0,axis = 1)
    
    def blur(self,p, grid, blob_size):
        """
        Blurs a projected protein conformation.

        Parameters
        ----------
        p : array of shape (N,2), the 2D projection to blur. 
        grid : Grid object
        blob_size : float
        """
        xs = grid.xs
        ys = grid.ys
    
        x_diff = xs[:, None] - p[:, 0]  # Compute differences in x direction for all points
        y_diff = ys[:, None] - p[:, 1]  # Compute differences in y direction for all points
    
        f1 = gauss.gaussian1(x_diff, blob_size)  # Gaussian values for x differences
        f2 = gauss.gaussian1(y_diff, blob_size)  # Gaussian values for y differences
    
        # Compute the outer product of f1 and f2 for each point and sum them up
        image = np.einsum('ik,jk->ij', f1, f2)
    
        return image
    
    def dblurstar_j(self,p, precomp, grid, blob_size):
        """
        Old function. Remove after testing. 
        """
        N = p.shape[0]
        v= np.zeros((N, 2))
        sigma = np.sqrt(2)*blob_size
        for i in range(N):
            for j in range(N):
                v[i] = v[i] + gauss.grad_gaussian2(p[i]-p[j], sigma)
            coords = grid.r2_to_grid(p[i])
            if 0<=coords[0]<grid.res and 0<=coords[1]<grid.res:
                v[i] = v[i] + precomp[:, coords[0], coords[1]]
            else:
                print('Atom out of frame')
                assert False
        return v
    def dblurstar(self,p, pi, grid, blob_size):
        """
        Computes the adjoint of the blur. 
        
        Parameters
        ----------
        p : array of shape (N,2), the 2D projection to compare to.
        pi : error image  
        grid : Grid object
        blob_size : float

        Returns
        -------
        array of shape (N,2), the adjoint of the blur.
        """
        
        inds = np.array([grid.r2_to_grid(p[i,:]) for i in range(len(p))])
        
        im_conv_x = ndimage.gaussian_filter(pi, blob_size, order = (1,0),mode = 'nearest')
        im_conv_y = ndimage.gaussian_filter(pi, blob_size, order = (0,1),mode = 'nearest')
        
        coords_1 = np.expand_dims(ndimage.map_coordinates(im_conv_x,inds.T),1)
        coords_2 = np.expand_dims(ndimage.map_coordinates(im_conv_y,inds.T),1)
        return np.concatenate((coords_1, coords_2), axis=1)
    
    def act(self,R, x, transpose=False): #not used
        if transpose:
            return np.einsum('bji,bj->bi', R, x)
        else:
            return np.einsum('bij,bj->bi', R, x)
