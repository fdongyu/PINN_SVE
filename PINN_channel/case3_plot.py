#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 15:15:34 2021

@author: feng779
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from datetime import datetime
import time
from pyDOE import lhs
import tensorflow as tf
import pickle
import os
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from SVE_module_dynamic_uh import SVE

import pdb


def time_convert(intime):
    """
    function to convert the time from string to datetime
    """
    Nt = intime.shape[0]
    outtime = []
    for t in range(Nt):
        timestr = intime[t].decode('utf-8')
        outtime.append(datetime.strptime(timestr, '%d%b%Y %H:%M:%S'))
    return outtime

if __name__ == "__main__": 
    
    """
    step one: read model output
    """
    hdf_filename = '/qfs/people/feng779/PYTHON/ML/PINN_SVE/HEC-RAS/case3/MixedFlow.p02.hdf'
    hf = h5py.File(hdf_filename,'r') 
    
    attrs = hf['Geometry']['Cross Sections']['Attributes'][:]
    staid = []
    eles = []
    reach_len = []
    for attr in attrs:
        staid.append(attr[2].decode('utf-8'))
        eles.append(attr[14])
        reach_len.append(attr[6])

    coor = np.cumsum(np.array(reach_len[:-1]))
    coor = [0] + coor.tolist()
    coor = coor[::-1]
    eles = np.array(eles)
    slope = np.gradient( eles, coor)
     
    water_surface = hf['Results']['Unsteady']['Output']["/Results/Unsteady/Output"]['Output Blocks']['Base Output']["Unsteady Time Series"]["Cross Sections"]['Water Surface'][:]
    velocity_total = hf['Results']['Unsteady']['Output']["/Results/Unsteady/Output"]['Output Blocks']['Base Output']["Unsteady Time Series"]["Cross Sections"]['Velocity Total'][:]
    Timestamp = hf['Results']['Unsteady']['Output']["/Results/Unsteady/Output"]['Output Blocks']['Base Output']["Unsteady Time Series"]['Time Date Stamp'][:]
    time_model = time_convert(Timestamp)
    
    water_depth = water_surface - eles[None,:]
    b = 10 # channel width
    
    warmup_step = 1
    velocity_total = velocity_total[warmup_step:]
    water_depth = water_depth[warmup_step:]

    ## dnstrm reach
    ind_dnstrm = 102 
    velocity_total = velocity_total[:,ind_dnstrm:]
    water_depth = water_depth[:,ind_dnstrm:]
    slope = slope[ind_dnstrm:]
    eles = eles[ind_dnstrm:]
    eles = eles - eles[-1]

    Nt = water_depth.shape[0]
    Nx = water_depth.shape[1]

    Nt_train = water_depth.shape[0]
    layers = [2] + 3*[1*64] + [2]
    
    t = np.arange(Nt_train)[:,None]
    x = np.array(coor[::-1])[:,None]
    x = x[ind_dnstrm:] - x[ind_dnstrm] ## dnstrm reach
    u_exact = velocity_total[:Nt_train,:]
    h_exact = water_depth[:Nt_train,:]
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = u_exact.flatten()[:,None]    
    h_star = h_exact.flatten()[:,None]  
    
    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    ##
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) ## IC
    uu1 = u_exact[0:1,:].T
    hh1 = h_exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))   ## upstrm BC
    uu2 = u_exact[:,0:1]
    hh2 = h_exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))   ## dnstrm BC
    uu3 = u_exact[:,-1:]
    hh3 = h_exact[:,-1:]

    X_u_IC = xx1
    X_h_IC = xx1
    u_IC = uu1
    h_IC = hh1
    X_u_BC = np.vstack([xx2, xx3])
    X_h_BC = np.vstack([xx2, xx3])
    u_BC = np.vstack([uu2, uu3])
    h_BC = np.vstack([hh2, hh3])

    useObs = True
    ## obs velocity
    ind_obs_u = [20]
    t_obs_u = np.array([])
    x_obs_u = np.array([])
    u_obs = np.array([])
    for iobs in ind_obs_u:
        t_obs_u = np.append( t_obs_u, t.flatten() )
        x_obs_u = np.append( x_obs_u, np.ones(Nt_train)*x[iobs] )
        u_obs = np.append( u_obs, u_exact[:Nt_train, iobs] )
    X_u_obs = np.vstack([x_obs_u, t_obs_u]).T
    u_obs = u_obs[:,None]
    ## obs water depth
    ind_obs_h = [20]
    t_obs_h = np.array([])
    x_obs_h = np.array([])
    h_obs = np.array([])
    for iobs in ind_obs_h:
        t_obs_h = np.append( t_obs_h, t.flatten() )
        x_obs_h = np.append( x_obs_h, np.ones(Nt_train)*x[iobs] )
        h_obs = np.append( h_obs, h_exact[:Nt_train, iobs] )
    X_h_obs = np.vstack([x_obs_h, t_obs_h]).T
    h_obs = h_obs[:,None]

    X_f_train = X_star
    slope = np.hstack([np.array(slope) for _ in range(Nt_train)])[:,None]

    exist_mode = 2
    saved_path = 'saved_model/case3/PINN_SVE.pickle'
    weight_path = 'log/case3/weights.out'
    # Training
    model1 = SVE(X_u_IC, X_h_IC,
                X_u_BC, X_h_BC,
                X_u_obs, X_h_obs,
                X_f_train,
                u_IC, h_IC,
                u_BC, h_BC,
                u_obs,h_obs,
                layers,    
                lb, ub, slope, b,
                X_star, u_star, h_star,
                ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path,
                useObs=False) 

    saved_path = 'saved_model/case3_obs/PINN_SVE.pickle'
    weight_path = 'log/case3_obs/weights.out'
    # Training
    model2 = SVE(X_u_IC, X_h_IC,
                X_u_BC, X_h_BC,
                X_u_obs, X_h_obs,
                X_f_train,
                u_IC, h_IC,
                u_BC, h_BC,
                u_obs,h_obs,
                layers, 
                lb, ub, slope, b,
                X_star, u_star, h_star,
                ExistModel=exist_mode, uhDir=saved_path, wDir=weight_path,
                useObs=True) 
    
    # Test data
    Nt_test = Nt_train
    N_test = Nt_test * Nx    ## Nt_test x Nx
    X_test = X_star[:N_test,:]
    x_test = X_test[:,0:1]
    t_test = X_test[:,1:2]
    u_test = u_star[:N_test,:]
    h_test = h_star[:N_test,:]
    
    # Prediction
    u_pred1, h_pred1 = model1.predict(x_test, t_test)
    error_u1 = np.linalg.norm(u_test-u_pred1,2)/np.linalg.norm(u_test,2)
    error_h1 = np.linalg.norm(h_test-h_pred1,2)/np.linalg.norm(h_test,2)
    print('Error u: %e' % (error_u1))
    print('Error h: %e' % (error_h1))

    u_pred2, h_pred2 = model2.predict(x_test, t_test)
    error_u2 = np.linalg.norm(u_test-u_pred2,2)/np.linalg.norm(u_test,2)
    error_h2 = np.linalg.norm(h_test-h_pred2,2)/np.linalg.norm(h_test,2)
    print('Error u: %e' % (error_u2))
    print('Error h: %e' % (error_h2))
    
    u_pred1 = u_pred1.reshape([Nt_test, Nx])
    h_pred1 = h_pred1.reshape([Nt_test, Nx])
    u_pred2 = u_pred2.reshape([Nt_test, Nx])
    h_pred2 = h_pred2.reshape([Nt_test, Nx])
    u_test = u_test.reshape([Nt_test, Nx])
    h_test = h_test.reshape([Nt_test, Nx])
    
    hours = np.arange(len(time_model[:Nt_test]))/4.
    x = x[::-1]  ## reverse the channel for plotting
    xx, tt = np.meshgrid(x.flatten(), hours)

    factor = 0.3048   # ft to m 
    xx *= factor 
    x  *= factor
    u_test *= factor
    u_pred1 *= factor
    u_pred2 *= factor
    h_test *= factor
    h_pred1 *= factor
    h_pred2 *= factor
    eles *= factor

    plt.rcParams.update({'font.size': 16})
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True, sharey=True)

    levels = np.linspace(0, 4, 9)
    cs = axes[0,0].contourf(xx, tt, u_test[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[0,0].set_ylabel('Time (h)')
    axes[0,0].set_title('Reference')

    cs = axes[0,1].contourf(xx, tt, u_pred1[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[0,1].set_title('Prediction (w/o obs)')

    cs = axes[0,2].contourf(xx, tt, u_pred2[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[0,2].set_title('Prediction (w/ obs)')
    divider = make_axes_locatable(axes[0,2])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(cs, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=14)
    cb.ax.yaxis.offsetText.set_fontsize(14)
    cb.set_label('Velocity (m/s)', fontsize=14)

    levels = np.linspace(0, 5, 9)
    cs = axes[1,0].contourf(xx, tt, h_test[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[1,0].set_ylabel('Time (h)')
    axes[1,0].set_xlabel('Distance upstream (m)')

    cs = axes[1,1].contourf(xx, tt, h_pred1[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[1,1].set_xlabel('Distance upstream (m)')

    cs = axes[1,2].contourf(xx, tt, h_pred2[:Nt_test,:], cmap='rainbow', levels=levels, alpha = 0.8)
    axes[1,2].set_xlabel('Distance upstream (m)')
    divider = make_axes_locatable(axes[1,2])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(cs, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=14)
    cb.ax.yaxis.offsetText.set_fontsize(14)
    cb.set_label('Water depth (m)', fontsize=14)

    axes = axes.ravel()
    for i in range(len(axes)):
        axes[i].text(0.05, 0.9, '{}'.format(labels[i]), fontsize=16, transform=axes[i].transAxes)

    plt.tight_layout() 
    plt.savefig('figures/case3/contour.png')
    plt.close()



    tlist = [12, 36, 64, 92]
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots( len(tlist), 2, figsize=(15, 8), sharex=True)
    for k in range(len(tlist)):
        axes[k, 0].plot(x, u_test[tlist[k],:], '-k', linewidth=2, label='reference')
        axes[k, 0].plot(x, u_pred1[tlist[k],:], '-b', linewidth=2, label='PINN (w/o obs)')
        axes[k, 0].plot(x, u_pred2[tlist[k],:], '-r', linewidth=2, label='PINN (w/ obs)')
        axes[k, 0].text(0.85, 0.85, 't={} h'.format(int(tlist[k]/4.)), fontsize=15, transform=axes[k,0].transAxes)
        axes[k, 0].set_ylabel('Velocity (m/s)')
        axes[k, 0].set_ylim([0,4])
        axes[k, 0].grid()
        axes[k, 1].plot(x, h_test[tlist[k],:]+eles, '-k', linewidth=2, label='reference')
        axes[k, 1].plot(x, h_pred1[tlist[k],:]+eles, '-b', linewidth=2, label='PINN (w/o obs)')
        axes[k, 1].plot(x, h_pred2[tlist[k],:]+eles, '-r', linewidth=2, label='PINN (w/ obs)')
        axes[k, 1].fill_between(x.flatten(), eles, color='0.7')
        axes[k, 1].set_ylabel('Water stage (m)')
        axes[k, 1].set_ylim([0,5])
        axes[k, 1].grid()
        if k == len(tlist) - 1:
            axes[k, 0].set_xlabel('Distance upstream (m)')
            axes[k, 1].set_xlabel('Distance upstream (m)')
        if k == 0:
            axes[k, 0].legend(loc=7,prop={'size': 12})

    plt.tight_layout()
    plt.savefig('figures/case3/along_channel.png')
    plt.close()

