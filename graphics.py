#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  graphics.py
#  
#  Copyright 2017 Anupam Mitra <anupam@unm.edu>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn

def plot_pwc_controlwaveform(phi, Nsteps, Tstep, fig=None, ax=None, titlestring=None, outfilename=None):
    """
    Plots a piecewise constant control waveform returned
    by GRAPE
    
    Parameters
    ----------
    phi:
    piecewise constant waveform
    
    Nsteps:
    number of steps
    
    Tstep:
    time interval of each step
    """
    if fig == None and ax == None:
        #fig = plt.figure()
        #ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
        fig, axes = plt.subplots(1, 1)
        ax=axes

    ax.step(np.arange(Nsteps+1)*Tstep/(np.pi), \
            np.concatenate([phi, [phi[-1]]]), \
            where='post', lw=1)
    
    '''
    # Decorations: phi is between 0 and 2*pi
    #ax.set_yticks(np.arange(0, 2.5, 0.5)*np.pi)
    #ax.set_yticklabels([0, r'$\frac{\pi}{2}$', r'$\pi$', r'$3\frac{\pi}{2}$', r'$2\pi$'])

    # Decorations: phi is between -pi and pi
    #ax.set_yticks(np.arange(-1.0, 1.5, 0.5)*np.pi)
    #ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', 0, r'$+\frac{\pi}{2}$', r'$+\pi$'])

    
    # Setting labels
    if(maketitle and len(delta_Delta_values) <= 3 and len(delta_Omega_values) <=3 ):
        title_str = r'$\Omega = %1.1f, \delta\Omega \in %s, \Delta = %1.1f, \delta\Delta \in %s$, 1-F=%g' \
                % (Omega, \
                   '\{' + ','.join([str(o) for o in np.round(delta_Omega_values, 3)]) + '\}', \
                   Delta, \
                   '\{' + ','. join([str(o) for o in np.round(delta_Delta_values, 3)]) + '\}', \
                   infidelity_min)
        ax.set_title(title_str)
    '''
    if titlestring != None:
        ax.set_title(titlestring, fontdict={'size': 16})
    
    ax.set_xlabel(r'$\Omega t / \pi$', fontdict={'size': 16})
    ax.set_ylabel(r'$\phi$', fontdict={'size': 16})
    ax.tick_params(labelsize=16)
    
    plt.tight_layout()

    if outfilename != None:
        fig.savefig(outfilename, filetype='pdf', bboxinches='tight', dpi=1024)
