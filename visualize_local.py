"""
visualization_Functions.py
============
Shared utility functions for visualizing vector data 

This module is imported in analysis notebooks as:
    import Visualize_Magnetization_vectors as vis

All functions are then called as vis.function_name(...)

Author: Emma Bernard
"""

# ========================= IMPORTS =========================


import os
import datetime

import numpy as np
import tifffile
import matplotlib as mpl
import matplotlib.pyplot as plt


from skimage.io import imread, imsave
from scipy.ndimage import rotate, shift
#import dxchange
import time

from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar
from numpy.fft import fft2, fftshift

from scipy.optimize import curve_fit, fsolve
from skimage.measure import profile_line


from scipy.ndimage import gaussian_filter
# ===========================================================

# -----------------------------------------------------------------------
# SECTION 1: Process vector slices
# -----------------------------------------------------------------------
def straighten_data(x, z, theta, mz, mzn, mnorm, n_show = 10): 
    # ---------- group points into x-columns ----------
    # unique_x[inv] reconstructs x; inv tells you which column each point belongs to
    unique_x, inv = np.unique(x, return_inverse=True)
    n_cols = len(unique_x)
    # print(n_cols)
    
    # ---------- find local top/bottom z per column ----------
    zmin_per_x = np.full(n_cols, np.inf)
    zmax_per_x = np.full(n_cols, -np.inf)
    np.minimum.at(zmin_per_x, inv, z)
    np.maximum.at(zmax_per_x, inv, z)
    
    # ---------- normalized thickness coordinate for every point ----------
    zmin_i = zmin_per_x[inv]
    zmax_i = zmax_per_x[inv]
    eta = (z - zmin_i) / (zmax_i - zmin_i)   # 0 at bottom, 1 at top, per column
    
    # ---------- choose evenly spaced eta levels to plot ----------
    eta_targets = np.linspace(0, 1, n_show)
    
    # ---------- interpolate theta(eta_target) within each column, build grid ----------
    theta_grid = np.full((n_show, n_cols), np.nan)
    mz_grid = np.full((n_show, n_cols), np.nan)
    mzn_grid = np.full((n_show, n_cols), np.nan)
    mnorm_grid = np.full((n_show, n_cols), np.nan)
    
    for j in range(n_cols):
        col_mask = (inv == j)
        eta_col = eta[col_mask]
        th_col  = theta[col_mask]
        mz_col = mz[col_mask]
        mzn_col = mzn[col_mask]
        mnorm_col = mnorm[col_mask]
        order = np.argsort(eta_col)
        eta_col, th_col, mz_col, mzn_col, mnorm_col = eta_col[order], th_col[order], mz_col[order], mzn_col[order], mnorm[order]
        # np.interp needs increasing x; also leaves NaN outside if you want strictness
        theta_grid[:, j] = np.interp(eta_targets, eta_col, th_col)
        mz_grid[:,j] = np.interp(eta_targets, eta_col, mz_col)
        mzn_grid[:,j] = np.interp(eta_targets, eta_col, mzn_col)
        mnorm_grid[:,j] = np.interp(eta_targets, eta_col, mnorm_col)

    return unique_x, theta_grid, mz_grid, mzn_grid, mnorm_grid, n_cols, eta_targets
 


def slice_straighten_and_store(cfg, xrange, field_pairs, n_show):
    xmin, xmax = xrange[0], xrange[1]
    center = xmin + (xmax - xmin)/2
    xrange_mask = (cfg['x'] >= xmin) & (cfg['x'] <= xmax)

    cfg_crop = dict(
        sample=cfg['sample'],
        coord = cfg['coord']
    )
    
    for field, field_n in field_pairs:

        x_crop, z_crop, theta_crop, m_crop, mn_crop, mnorm_crop = cfg['x'][xrange_mask], cfg['z'][xrange_mask], cfg['theta'][xrange_mask], cfg[field][xrange_mask], cfg[field_n][xrange_mask], cfg['mnorm'][xrange_mask]

        unique_x, theta_grid, grid, grid_n, norm_grid, n_cols, eta_targets = straighten_data(
            x_crop, z_crop, theta_crop, m_crop, mn_crop, mnorm_crop, n_show=n_show)

        cfg_crop.setdefault('unique_x', unique_x)
        cfg_crop.setdefault('theta_grid', theta_grid)
        cfg_crop.setdefault('n_cols', n_cols)
        cfg_crop.setdefault('eta_targets', eta_targets)
        cfg_crop.setdefault('M_mag', norm_grid)

        cfg_crop[f'{field}_grid'] = grid
        cfg_crop[f'{field_n}_grid'] = grid_n

    return cfg_crop



def find_crossings(x_arr, theta_arr, target=np.pi/2, tol_deg=0.05):
    tol_rad = np.deg2rad(tol_deg)
    diffs = theta_arr - target
    sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]

    crossings = []
    for i0 in sign_changes:
        i1 = i0 + 1
        x0, x1 = x_arr[i0], x_arr[i1]
        th0, th1 = theta_arr[i0], theta_arr[i1]
        if th1 == th0:  # avoid divide-by-zero on a flat segment
            continue
        x_star = x0 + (target - th0) * (x1 - x0) / (th1 - th0)
        # sanity check: interpolated theta should sit within tolerance of target
        th_check = np.interp(x_star, [x0, x1], [th0, th1])
        if abs(th_check - target) <= tol_rad:
            crossings.append(x_star)
    return crossings 
    
# -----------------------------------------------------------------------
# SECTION 2: Calculate Ideal Magnetic Profiles
# -----------------------------------------------------------------------

def theta_profile(x, delta):
    """Bloch/Neel/Hybrid wall profile: theta(x) = 2*arctan(exp(x/delta))"""
    return 2 * np.arctan(np.exp(x / delta))

def magnetization_components(x, delta, psi):
    """
    Compute Mx, My, Mz for a domain wall with:
      - wall normal along x
      - easy axis along z
      - azimuthal angle psi (psi=0: Neel, psi=pi/2: Bloch)
    Returns normalized components (M/Ms).
    """
    theta = theta_profile(x, delta)
    Mx = np.sin(theta) * np.cos(psi)
    My = np.sin(theta) * np.sin(psi)
    Mz = np.cos(theta)
    return Mx, My, Mz

# -----------------------------------------------------------------------
# SECTION 3: Calculate Magnetic Parameters from line cuts
# -----------------------------------------------------------------------

# def gradient(cfg, field_pairs, colors, smooth=True, normalized = True, plot=True):
#     dx = cfg['unique_x'][1] - cfg['unique_x'][0]
#     cfg.setdefault('dx', dx)

#     key_add = 'n_grid' if normalized else '_grid'
#     grad_key = 'n_dx' if normalized else '_dx'

#     for field, field_n in field_pairs:
#         m_grid = cfg[field + key_add]

#         if smooth:
#             set_sigma = 2 ############################################################################## CHANGE SIGMA HERE 
#             m_grid_used = gaussian_filter(m_grid, sigma=(0, set_sigma))
#             cfg[field+key_add+'_smooth'] = m_grid_used  # keep raw grid untouched
#         else:
#             m_grid_used = m_grid
#             cfg[f'{field}+{key_add}'] = m_grid_used


#         dM_dx = np.gradient(m_grid_used, dx, axis=1)
#         cfg[f'd{field}{grad_key}'] = dM_dx
        
#     if plot:
#         plot_profile_gradient(cfg, key_add, grad_key, field_pairs, colors, smooth = smooth)

#     return cfg

def gradient(cfg, field_pairs, colors, smooth=True, normalized=True, plot=True):
    dx = cfg['unique_x'][1] - cfg['unique_x'][0]
    cfg.setdefault('dx', dx)

    key_add = 'n_grid' if normalized else '_grid'
    grad_key = 'n_dx' if normalized else '_dx'

    for field, field_n in field_pairs:
        grid_key = field + key_add
        m_grid = cfg[grid_key]

        if smooth:
            set_sigma = 2  ############################################## CHANGE SIGMA HERE
            m_grid_used = gaussian_filter(m_grid, sigma=(0, set_sigma))

            cfg.setdefault(grid_key + '_raw', m_grid)  # keep original for reference
            cfg[grid_key] = m_grid_used                # overwrite main key going forward
        else:
            m_grid_used = m_grid

        dM_dx = np.gradient(m_grid_used, dx, axis=1)
        cfg[f'd{field}{grad_key}'] = dM_dx

    if plot:
        plot_profile_gradient(cfg, key_add, grad_key, field_pairs, colors)

    return cfg


def hyperbolic(x, M0, x0, w): 
    M =  M0*np.tanh(((x- x0))/w) 
    return M

def fit_hyperbolic(x, y):
    """
    For given array x of pixel intensity across a domain wall, returns a hyperbolic tangent fit of the domain wall profile and center position 

    Parameters
    ----------
    x : array_like
        array of x values at which the dw should be evaluated


    Returns
    -------
    hyperbolic fit : array_like
        y value(s) of trapezoid with above parameters, evaluated at x
    hyperbolic error : array_like
    """
    M0 = np.max(y) - np.min(y)
    x0 = np.mean(x)
    w = np.std(x) * 2 # start with this for now... doesn't really mean anything physically.
    #print(f'initial guess: M0={M0}, x0={x0}, w={w}')
    guess = [M0, x0, w]
    fit  = curve_fit(hyperbolic, x, y, p0=guess)    

    return fit[0], fit[1]


def hyperbolic_noM0(x, x0, w): 
    M0 = 0.1-(-0.1)
    M =  M0*np.tanh(((x- x0))/w) 
    return M

def fit_hyperbolic_noM(x, y):
    """
    For given array x of pixel intensity across a domain wall, returns a hyperbolic tangent fit of the domain wall profile and center position 
    Mo is fixed to 0.1-(-0.1)
    Parameters
    ----------
    x : array_like
        array of x values at which the dw should be evaluated


    Returns
    -------
    hyperbolic fit : array_like
        y value(s) of trapezoid with above parameters, evaluated at x
    hyperbolic error : array_like
    """
    
    x0 = np.mean(x)
    w = np.std(x) * 2 # start with this for now... doesn't really mean anything physically.
    #print(f'initial guess: x0={x0}, w={w}')
    guess = [x0, w]
    fit  = curve_fit(hyperbolic_noM0, x, y, p0=guess)    

    return fit[0], fit[1]


def solve_for_a(x,y, height,slope):
    # Define the function to find roots for
    def equation(a):
        return height* np.tanh((x + 25* a)/slope) - height * np.tanh((x - 25* a)/slope) - y
    
    # Use fsolve to find the value of 'a' that satisfies the equation
    initial_guess = 1  # Initial guess for 'a'
    solution = fsolve(equation, initial_guess)
    
    return solution[0]

def domain_wall_width(x, y):

    left = x[y.searchsorted(0.99, 'left')]
    right = x[y.searchsorted(0.01, 'left')]

    dww = left - right
    
    return dww


def plot_YX_linecut(recon, subsect_x, subsect_y, shift_z, center, sample, Gamma): 
    pixel_size_nm   = 8         # voxel size in nm  
    k = center
    min_x, max_x = subsect_x
    y_lo, y_hi = subsect_y


    colors = plt.cm.plasma(np.linspace(0, 1, len(shift_z)))
    
    linecut_z = [ ]
    dw_width_bloch = []
    dw_center = []

    fig = plt.figure(figsize=(21, 6))
    ax1 = fig.add_subplot(1, 4, 1)
    cut = ax1.imshow(recon[k,:,:], cmap='viridis')
    ax1.set_title(f'Gamma = {Gamma}, YX')
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(center-10, center+10)
    fig.colorbar(cut, ax=ax1)
    
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_xlabel('cross section (nm)')
    ax2.set_ylabel('Intensity (arb)')
    ax2.set_title('Line cuts of domains at different z')
    
    ax3 = fig.add_subplot(1,4, 3)
    ax3.set_xlabel('slice')
    ax3.set_ylabel('DW width')
    ax3.set_title('fit DW width at different slices')


    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_xlabel('slice')
    ax4.set_ylabel('DW center')
    ax4.set_title('fit DW center at different slices')


    for i, z in enumerate(shift_z):
        p0 = (y_lo + z, 0)
        p1 = (y_lo + z, center*2)
    
        # Compute the linecut
        linecut = profile_line(
            recon[k, :, :],
            src=p0,
            dst=p1,
            linewidth=1,
            order=1,
            mode='reflect'
        )
        
        linecut_fit = linecut[min_x:max_x]
        x = np.arange(len(linecut)) * pixel_size_nm
        x_fit = x[min_x:max_x]
        linecut_z.append(linecut_fit)
        
        
        hyperbolic_fit, hyperbolic_error = fit_hyperbolic(x_fit, linecut_fit)
        hyperbolic_plot = hyperbolic(x_fit, *hyperbolic_fit)
        dw_width_bloch.append(hyperbolic_fit[2]) ## might need to be converted into nm
        dw_center.append(hyperbolic_fit[1])
        
        ax1.plot([p0[1], p1[1]], [p0[0], p1[0]], color=colors[i], label=f'shift {z} nm')

        # Overlay linecut in second subplot
        ax2.plot(x_fit, linecut_fit, color=colors[i], label=f'shift {z} nm', marker='o')
        ax2.plot(x_fit, hyperbolic_plot, color=colors[i], linewidth=3, linestyle = '--', label=f'Hyperbolic Fit {sample}, {Gamma},\n DW width = {hyperbolic_fit[2]:.2f} nm')

    
    ax3.plot(shift_z, np.abs(dw_width_bloch), marker='o')
    ax4.plot(shift_z, dw_center, marker='o')
    
    plt.tight_layout()
    plt.show()

    return linecut_z, dw_width_bloch, dw_center

def plot_YZ_linecut(recon, subsect_x, subsect_y, shift_z, center, sample, Gamma): 
    
    pixel_size_nm   = 8         # voxel size in nm  
    k = center
    min_x, max_x = subsect_x
    y_lo, y_hi = subsect_y

    colors = plt.cm.plasma(np.linspace(0, 1, len(shift_z)))
    
    linecut_z = [ ]
    dw_width_bloch = []
    dw_center = []

    fig = plt.figure(figsize=(21, 6))
    ax1 = fig.add_subplot(1, 4, 1)
    cut = ax1.imshow(recon[:,:,k], cmap='viridis')
    ax1.set_title(f'Gamma = {Gamma}, YZ')
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_xlim(center-10, center+10)
    fig.colorbar(cut, ax=ax1)
    
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_xlabel('cross section (nm)')
    ax2.set_ylabel('Intensity (arb)')
    ax2.set_title('Line cuts of domains at different z')
    
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.set_xlabel('slice')
    ax3.set_ylabel('DW width')
    ax3.set_title('fit DW width at different slices')

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_xlabel('slice')
    ax4.set_ylabel('DW center')
    ax4.set_title('fit DW center at different slices')


    for i, z in enumerate(shift_z):
        p0 = (0, min_x + z)
        p1 = (center*2, min_x + z)
    
        # Compute the linecut
        linecut = profile_line(
            recon[:, :, k],
            src=p0,
            dst=p1,
            linewidth=1,
            order=1,
            mode='reflect'
        )
        
        linecut_fit = linecut[y_lo:y_hi]
        x = np.arange(len(linecut)) * pixel_size_nm
        x_fit = x[y_lo:y_hi]
        linecut_z.append(linecut_fit)
        
        
        hyperbolic_fit, hyperbolic_error = fit_hyperbolic(x_fit, linecut_fit)
        hyperbolic_plot = hyperbolic(x_fit, *hyperbolic_fit)
        dw_width_bloch.append(hyperbolic_fit[2]) ## might need to be converted into nm
        dw_center.append(hyperbolic_fit[1])
        
        ax1.plot([p0[1], p1[1]], [p0[0], p1[0]], color=colors[i], label=f'shift {z} nm')

        # Overlay linecut in second subplot
        ax2.plot(x_fit, linecut_fit, color=colors[i], label=f'shift {z} nm', marker='o')
        ax2.plot(x_fit, hyperbolic_plot, color=colors[i], linewidth=3, linestyle = '--', label=f'Hyperbolic Fit {sample}, {Gamma},\n DW width = {hyperbolic_fit[2]:.2f} nm')

    
    ax3.plot(shift_z, np.abs(dw_width_bloch), marker='o')
    ax4.plot(shift_z, dw_center, marker='o')
    
    plt.tight_layout()
    plt.show()

    return linecut_z, dw_width_bloch, dw_center




# -----------------------------------------------------------------------
# SECTION 3: Plot linecuts
# -----------------------------------------------------------------------
def plot_magnitude(cfg, colors):
    fig, ax = plt.subplots(figsize=(7, 7))
    for j, eta_v in enumerate(cfg['eta_targets']):
        ax.plot(cfg['unique_x'], cfg['M_mag'][j, :], color=colors[j], lw=1.5, label=fr'{eta_v*100:.0f}%')
    
    ax.set_ylabel(r'$M_{mag}$')
    ax.set_xlabel('x (pix)')
    ax.legend(fontsize=8, ncol=2, title=r'fraction of local thickness', loc = 'upper left')
    ax.set_title(f"Magnitude \n full {cfg['coord']}-slice profile \n {cfg['sample']}")


# def plot_profile_gradient(cfg, key_add, grad_key, field_pairs, colors, smooth=True): 
#     for field, field_n in field_pairs:
#         #print(field_n)
#         fig, ax1 = plt.subplots(1, 2, figsize=(15, 5))

#         # pick the same array actually used to compute the gradient
#         profile_key = field + key_add + ('_smooth' if smooth else '')
#         profile_grid = cfg[profile_key]

#         for j, eta_v in enumerate(cfg['eta_targets']):
#             ax1[0].plot(cfg['unique_x'], profile_grid[j, :], color=colors[j], lw=1.5,
#                         label=fr'{eta_v*100:.0f}%')
#             ax1[1].plot(cfg['unique_x'], cfg[f'd{field}{grad_key}'][j, :], color=colors[j], lw=1.5,
#                         label=fr'{eta_v*100:.0f}%')

#         ax1[0].set_ylabel(r'$M$')
#         ax1[0].set_xlabel('x (pix)')
#         ax1[0].legend(fontsize=8, ncol=2, title='fraction of local thickness', loc='upper right')
#         ax1[0].set_title(f"{field} \n full {cfg['coord']}-DW profile\n {cfg['sample']}"
#                           + (" (smoothed)" if smooth else ""))

#         ax1[1].set_xlabel('x (pix)')
#         ax1[1].set_ylabel(r'$dM(x)$/dx')
#         ax1[1].legend(fontsize=8, ncol=2, title='fraction of local thickness', loc='upper right')
#         ax1[1].set_title(f"{field} \n full {cfg['coord']}-DW Gradient\n {cfg['sample']}"
#                           + (" (smoothed)" if smooth else ""))

#         plt.tight_layout()
#         plt.show() 


def plot_profile_gradient(cfg, key_add, grad_key, field_pairs, colors):
    for field, field_n in field_pairs:
        fig, ax1 = plt.subplots(1, 2, figsize=(15, 5))

        for j, eta_v in enumerate(cfg['eta_targets']):
            ax1[0].plot(cfg['unique_x'], cfg[field + key_add][j, :], color=colors[j], lw=1.5,
                        label=fr'{eta_v*100:.0f}%')
            ax1[1].plot(cfg['unique_x'], cfg[f'd{field}{grad_key}'][j, :], color=colors[j], lw=1.5,
                        label=fr'{eta_v*100:.0f}%')

        ax1[0].set_ylabel(r'$M$')
        ax1[0].set_xlabel('x (pix)')
        ax1[0].legend(fontsize=8, ncol=2, title='fraction of local thickness', loc='upper right')
        ax1[0].set_title(f"{field} \n full {cfg['coord']}-DW profile\n {cfg['sample']}")

        ax1[1].set_xlabel('x (pix)')
        ax1[1].set_ylabel(r'$dM(x)$/dx')
        ax1[1].legend(fontsize=8, ncol=2, title='fraction of local thickness', loc='upper right')
        ax1[1].set_title(f"{field} \n full {cfg['coord']}-DW Gradient\n {cfg['sample']}")

        plt.tight_layout()
        plt.show()


def plot_full_plot(cfg,field_pairs,colors):
    ## Look at full slice, normalized and unnormalized
    
    for field, field_n in field_pairs:

        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        for j, eta_v in enumerate(cfg['eta_targets']):
            #ax1.plot(unique_x, mz_grid[j, :], color=colors[j], lw=1.5, label=fr'{eta_v*100:.0f}%')
            ax[0].plot(cfg['unique_x'], cfg[field + '_grid'][j, :], color=colors[j], lw=1.5, label=fr'{eta_v*100:.0f}%')
        
        ax[0].set_ylabel(r'$M$')
        ax[0].set_xlabel('x (pix)')
        ax[0].legend(fontsize=8, ncol=2, title=r'fraction of local thickness', loc = 'upper left')
        ax[0].set_title(f"{field} \n full {cfg['coord']}-slice profile \n {cfg['sample']}")
        
        for j, eta_v in enumerate(cfg['eta_targets']):
            ax[1].plot(cfg['unique_x'], cfg[field_n+ '_grid'][j, :], color=colors[j], lw=1.5, label=fr'{eta_v*100:.0f}%')
        
        ax[1].set_ylabel(r'$M / M_{norm}$')
        ax[1].set_xlabel('x (pix)')
        ax[1].legend(fontsize=8, ncol=2, title=r'fraction of local thickness', loc = 'upper left')
        ax[1].set_title(f"Normalized {field} \n full {cfg['coord']}-slice profile \n {cfg['sample']}")
        plt.tight_layout()
        plt.show()


def plot_wall_gradients(wall, cfg, x0):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = {
        'dMz_dx': r'$dM_z(x)/dx = \cos\theta(x)$',
        'dMy_dx': r'$dM_y(x)/dx = \sin\theta\sin\psi$',
        'dMx_dx': r'$dM_x(x)/dx = \sin\theta\cos\psi$',
    }

    ylim_settings = {
        ('bloch', 'dMx_dx'): {'ylim': (-1.1, 1.1), 'yticks': [-1, -0.5, 0, 0.5, 1]},
        # add more (wall, comp) entries here as needed, e.g.:
        # ('neel', 'Mx'): {'ylim': (-0.1, 1.1), 'yticks': [0, 0.5, 1]},
    }

    psi = cfg['psi']

    if np.isscalar(psi):
        for ax, comp in zip(axes, ['dMz_dx', 'dMy_dx', 'dMx_dx']):
            ax.plot(x0, cfg[comp], color=cfg['color'], label=labels[comp])
    else:
        # hybrid: one line per psi value, colored by cmap
        for ax, comp in zip(axes, ['dMz_dx', 'dMy_dx', 'dMx_dx']):
            for psi_val, color, values in zip(psi, cfg['colors'], cfg[comp]):
                #label = rf'$\psi = {np.degrees(psi_val):.0f}°$'
                ax.plot(x0, values, color=color)#, label=label)
                
    for ax, comp in zip(axes, ['dMz_dx', 'dMy_dx', 'dMx_dx']):
        ax.set_xlabel(r'$x$ (pix)')
        ax.set_ylabel(f'${comp[0]}{comp[1]}_{comp[2]}$')
        ax.set_title(fr'${comp[0]}{comp[1]}_{comp[2]}(x)$ {wall.capitalize()} Wall Gradient')
        # ax.legend()
        ax.axhline(0, color='gray', ls='-', alpha=0.3, lw=1)
        ax.axvline(0, color='gray', ls='--', alpha=0.5, lw=1)
        
        key = (wall, comp)
        if key in ylim_settings:
            ax.set_ylim(*ylim_settings[key]['ylim'])
            ax.set_yticks(ylim_settings[key]['yticks'])

    plt.tight_layout()
    plt.show()



def plot_wall_components(wall, cfg, x0):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = {
        'Mz': r'$M_z(x) = \cos\theta(x)$',
        'My': r'$M_y(x) = \sin\theta\sin\psi$',
        'Mx': r'$M_x(x) = \sin\theta\cos\psi$',
    }

    ylim_settings = {
        ('bloch', 'Mx'): {'ylim': (-1.1, 1.1), 'yticks': [-1, -0.5, 0, 0.5, 1]},
        # add more (wall, comp) entries here as needed, e.g.:
        # ('neel', 'Mx'): {'ylim': (-0.1, 1.1), 'yticks': [0, 0.5, 1]},
    }


    psi = cfg['psi']

    if np.isscalar(psi):
        for ax, comp in zip(axes, ['Mz', 'My', 'Mx']):
            ax.plot(x0, cfg[comp], color=cfg['color'], label=labels[comp])
    else:
        # hybrid: one line per psi value, colored by cmap
        for ax, comp in zip(axes, ['Mz', 'My', 'Mx']):
            for psi_val, color, values in zip(psi, cfg['colors'], cfg[comp]):
                #label = rf'$\psi = {np.degrees(psi_val):.0f}°$'
                ax.plot(x0, values, color=color) #, label=label)
    
    for ax, comp in zip(axes, ['Mz', 'My', 'Mx']):
        ax.set_xlabel(r'$x$ (pix)')
        ax.set_ylabel(f'${comp[0]}_{comp[1]}$')
        ax.set_title(fr'${comp[0]}_{comp[1]}(x)$ {wall.capitalize()} Wall')
        ax.legend()
        ax.axhline(0, color='gray', ls='-', alpha=0.3, lw=1)
        ax.axvline(0, color='gray', ls='--', alpha=0.5, lw=1)
        
        key = (wall, comp)
        if key in ylim_settings:
            ax.set_ylim(*ylim_settings[key]['ylim'])
            ax.set_yticks(ylim_settings[key]['yticks'])

        elif comp == 'Mz':
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())          # same data limits as left axis (both in cos space)
            
            cos_tick_vals = [-1, -0.5, 0, 0.5, 1]
            ax2.set_yticks(cos_tick_vals)         # positioned at the same heights as left axis ticks
            ax2.set_yticklabels([
                r'$\pi$',
                r'$2\pi/3$',
                r'$\pi/2$',
                r'$\pi/3$',
                r'$0$',
            ])
            ax2.set_ylabel(r'$\theta(x)$ (rad.)')


    plt.tight_layout()
    plt.show()
