#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:26:07 2024

@author: erikjans
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def planeplot(da, color = "gray_r",plotobjs = None, save = False, name = None,**kwargs):
    """
    2D plot of projected protein conformation. 

    Parameters
    ----------
    da: np.array, shape = (n_atoms,2)
    color: str, color to plot in. 
    plotobjs : tuple, optional, contains the figure and axes objects. Default = None.
    save : bool, optional. Default = False.
    name : str, optional. Default = None. File name to save.
    **kwargs : optional. If you are saving, include the filename and dpi.

    Returns
    -------
    fig,axs : tuple, contains the figure and axes objects. 
    """
    if plotobjs is None:
        fig, axs = plt.subplots(1, 1)
    else: 
        fig,axs = plotobjs
    xdata = da[:,0]
    ydata = da[:,1]
    axs.plot(xdata[:,0], ydata[:,1], '-', c=color)
    if save:
        assert 'filename' in kwargs #"supply filename if printing is desired"
        assert 'dpi' in kwargs #supply dpi
        fig.savefig(fname = kwargs['filename'], dpi = kwargs['dpi'], transparent = True)
    return fig,axs


def spaceplot(da,color,plotobjs = None, save = False, name = None, **kwargs):
    """
    3D plot of protein conformation. 

    Parameters
    ----------
    da : np.array, shape = (n_atoms,3)
    color : str, colorscheme name. Default = "gray_r". 
    plotobjs : tuple, optional, contains the figure and axes objects. Default = None. 
    save : bool, optional. Default = False.
    name : str, optional. Default = None. File name to save.
    **kwargs : optional. If you are saving, include the filename and dpi.
    
    Returns
    -------
    fig,axs : tuple, contains the figure and axes objects. 
    """
    if plotobjs is None:
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection='3d')
        ax.set_axis_off()
    else: 
        fig,ax = plotobjs
    xdata = da[:,0]
    ydata = da[:,1]
    zdata = da[:,2]
    ax.plot3D(xdata,ydata,zdata,'black',zorder =0)
    ax.scatter(xdata,ydata,zdata, c = zdata,cmap = color, s = 200, zorder = 1)
    if save:
        assert 'filename' in kwargs #"supply filename if printing is desired"
        assert 'dpi' in kwargs #supply dpi
        fig.savefig(fname = kwargs['filename'], dpi = kwargs['dpi'], transparent = True)
    return fig,ax


def imageplot(image,color = "gray_r", plotobjs = None, save = False, name = None, **kwargs):
    """
    Plots the 2D projected electrostatic potential. 

    Parameters
    ----------
    image: np.array, shape = (n_pixels,n_pixels)
    color: str, colorscheme name. Default = "gray_r". 
    plotobjs : tuple, optional, contains the figure and axes objects. Default = None.
    save : bool, optional. Default = False.
    name : str, optional. Default = None. File name to save.
    **kwargs : optional. If you are saving, include the filename and dpi.

    Returns
    -------
    fig,axs : tuple, contains the figure and axes objects. 
    """
    if plotobjs is None:
        fig, axs = plt.subplots(1, 1)

    else: 
        fig, axs = plotobjs
    im = axs.imshow(image, cmap = color)
    if 'cbar' in kwargs:
        if kwargs.pop('cbar'):
            fig.colorbar(im,ax = axs)
    if save: 
        assert 'filename' in kwargs #"supply filename if saving"
        assert 'dpi' in kwargs #supply dpi
        fig.savefig(fname = kwargs['filename'], dpi = kwargs['dpi'], transparent = True)
    return fig,axs
            
    
def imagegridplot(image_dict,shape,span,color = "gray_r", save = False, name = None, **kwargs):
    """
    Plots a grid of projected electrostatic potentials. Supply a dictionary of images indexed by their respective row and column. 

    Parameters
    ----------
    image_dict: dictionary of np.array, shape = (n_pixels,n_pixels) indexed by their respective row and column.
    shape: tuple, shape = (n_rows, n_cols), shape of the grid. 
    span: float, span of the grid.
    color: str, colorscheme name. Default = "gray_r".
    save: bool, optional. Default = False.
    name : str, optional. Default = None. File name to save.
    **kwargs : optional. If you are saving, include the filename and dpi.

    Returns
    -------
    fig,axs : tuple, contains the figure and axes objects. 

    """
    nrows,ncols = shape
    for i in range(nrows):
        for j in range(ncols):
            assert (i,j) in image_dict #missing figure! Make sure that you supply exactly n xm figures. 
    fig, axs = plt.subplots(nrows,ncols, figsize=(20, 30))
    do_colorbar = False 
    if 'cbar' in kwargs:
        do_colorbar = kwargs.pop('cbar')
    for i in range(nrows):
        for j in range(ncols):
            im = axs[i,j].imshow(image_dict[(i,j)].T, extent=[-span, span, -span, span], origin='lower',cmap = color)
            if do_colorbar:
                fig.colorbar(im,ax = axs)
    pad = 10 # in points
    if 'colnames' in kwargs:
        cols = kwargs.pop('colnames')
        for ax, col in zip(axs[0], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')
    if 'rownames' in kwargs:
        rows = kwargs.pop('rownames')
        for ax, row in zip(axs[:,0], rows):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')
    
    if save: 
        assert 'filename' in kwargs #"supply filename if saving"
        assert 'dpi' in kwargs #supply dpi
        fig.savefig(fname = kwargs['filename'], dpi = kwargs['dpi'], transparent = True)
    return fig,axs


