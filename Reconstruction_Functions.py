"""
Reconstruction_Functions.py
============
Shared utility functions for tomography data processing and absorbtion contrast analysis

This module is imported in analysis notebooks as:
    import Reconstruction_Functions as reconstruct

All functions are then called as reconstruct.function_name(...)

Author: Emma Bernard
"""

# ========================= IMPORTS =========================
import tomopy

import os
import datetime

import numpy as np
import tifffile
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

from skimage.io import imread, imsave
from scipy.ndimage import rotate, shift
#import dxchange
import time

from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar
from numpy.fft import fft2, fftshift

from scipy.optimize import curve_fit, fsolve
from skimage.measure import profile_line
# ===========================================================

# -----------------------------------------------------------------------
# SECTION 1: Tomography Alignment Functions
# -----------------------------------------------------------------------

def plot_crosssections (k, reconstruction): 
  
     # --- Extract Nx, Ny ---
    Nz, Ny, Nx = reconstruction.shape 

    # --- Define physical extent in μm ---
    FOV_x = 4.0  # μm
    FOV_y = 4.0  # μm

    extent = [0, FOV_x, 0, FOV_y]  # [xmin, xmax, ymin, ymax] in μm

    # Tick positions every 1 μm
    ticks_um = np.arange(0, (FOV_x+0.1), 1.0)

    # --- Plot ---
    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(reconstruction[k,:,:], extent = extent, origin = 'lower')
    ax1.set_title(f'YX')
    ax1.set_ylabel("y (μm)")
    ax1.set_yticks(ticks_um)
    ax1.set_xlabel("x (μm)")
    ax1.set_xticks(ticks_um)
    
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(reconstruction[:,k,:], extent = extent, origin = 'lower')
    ax2.set_title('ZX')
    ax2.set_ylabel("z (μm)")
    ax2.set_yticks(ticks_um)
    ax2.set_xlabel("x (μm)")
    ax2.set_xticks(ticks_um)
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(reconstruction[:,:,k], extent = extent, origin = 'lower')
    ax3.set_title('ZY')
    ax3.set_ylabel("z (μm)")
    ax3.set_yticks(ticks_um)
    ax3.set_xlabel("y (μm)")
    ax3.set_xticks(ticks_um)
    
    
def estimate_rotation(img_ref, img_mov):
    F1 = np.abs(fftshift(fft2(img_ref)))
    F2 = np.abs(fftshift(fft2(img_mov)))

    radius = min(img_ref.shape) // 2

    W1 = warp_polar(F1, radius=radius)
    W2 = warp_polar(F2, radius=radius)

    shift, error, _ = phase_cross_correlation(W1, W2)
    rotation = shift[0] * 360 / W1.shape[0]

    return rotation

def crop_center(img, cy, cx, half):
    y0 = max(cy - half, 0)
    y1 = min(cy + half, img.shape[0])
    x0 = max(cx - half, 0)
    x1 = min(cx + half, img.shape[1])
    return img[y0:y1, x0:x1]

# ## Rotate the projection 90 degrees so that the axis of rotation is vertical
# rotated_0 = rotate(proj_0,angle=90,axes=(1, 2),reshape=False, order=1)
# print(rotated_0.shape)

def rotate_about_COR(image, rotation_angle, y_centroid, x_centroid, angle, sample, Gamma, order=1, plot = True):
    '''
    Can only rotate one image at a time
    '''
    ny, nx = image.shape

    # image center
    cy = (ny) / 2
    cx = (nx) / 2
    
    # shift needed to bring centroid to image center (unrotated coordinates)
    shift_to_center = (cy - y_centroid, cx - x_centroid)
    # print(shift_to_center)
    
    # 1. shift centroid to center (x and y)
    img_shifted = shift(image, shift=shift_to_center, order=order, mode='constant')
    
    # 2. rotate about center
    rotated_img = rotate(img_shifted, angle=rotation_angle, reshape=True, order=order)

    ## shift centroid by 90
    x_cent_rot = y_centroid
    y_cent_rot = ny - x_centroid

    # 3. shift back
    img_final = shift(rotated_img, shift=(-shift_to_center[0], -shift_to_center[1]),
                      order=order, mode='constant')

    if plot == True:
        
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(image)
        ax1.axhline(y=cy, color='b', linewidth=2)
        ax1.axvline(x=cx, color='b', linewidth=2)
        ax1.plot(cx, cy, 'bs', markersize=10, label = f'image center \n ({cx}, {cy})')
        ax1.plot(x_centroid, y_centroid, 'ro', markersize=10, label = f'centroid \n ({x_centroid}, {y_centroid})')
        #ax1.plot(x_centroid-shift_to_center[0], y_centroid-shift_to_center[1], 'g^', markersize=10, label = f'centroid shift to center /n ({x_centroid-shift_to_center[0]}, {y_centroid-shift_to_center[1]})')
        ax1.legend(loc= 'upper left')
        ax1.set_title(f'Gamma {Gamma}, alpha = {angle:.2f}, unrotated')

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(img_shifted)
        #ax2.plot(x_centroid, y_centroid, 'ro', markersize=10, label = f'centroid \n ({x_centroid}, {y_centroid})')
        ax2.axhline(y=cy, color='b', linewidth=2)
        ax2.axvline(x=cx, color='b', linewidth=2)
        ax2.plot(cx, cy, 'bs', markersize=10, label = f'old image center \n ({cx}, {cy})')
        ax2.plot(x_centroid+shift_to_center[1], y_centroid+shift_to_center[0], 'ro', markersize=10, label = f'centroid shifted to center \n ({x_centroid+shift_to_center[1]}, {y_centroid+shift_to_center[0]})')
        ax2.set_title(f'Gamma {Gamma} shifted')
        ax2.legend(loc= 'upper left')

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(rotated_img)
        #ax3.plot(x_centroid, y_centroid, 'ro', markersize=10, label = f'centroid \n ({x_centroid},{ y_centroid})')
        ax3.plot(cx, cy, 'bs', markersize=10, label = f'old image center \n ({cx}, {cy})')
        ax3.axhline(y=cy, color='b', linewidth=2)
        ax3.axvline(x=cx, color='b', linewidth=2)
        ax3.plot(x_centroid+shift_to_center[1], y_centroid+shift_to_center[0], 'ro', markersize=10, label = f'centroid shift to center \n ({x_centroid+shift_to_center[1]}, {y_centroid+shift_to_center[0]})')
        ax3.set_title(f'Gamma {Gamma} rotated')
        ax3.legend()

        # ax4 = fig.add_subplot(2, 2, 4)
        # ax4.imshow(img_final)
        # ax4.plot(x_cent_rot, y_cent_rot, 'g^', markersize=10, label = f'centroid rotated by 90 deg. \n ({x_cent_rot},{ y_cent_rot})')
        # #ax4.plot(x_centroid-shift_to_center[0], y_centroid-shift_to_center[1], 'g^', markersize=10, label = f'centroid shift to center \n ({x_centroid-shift_to_center[1]}, {y_centroid-shift_to_center[0]})')
        # ax4.plot(cx, cy, 'bs', markersize=10, label = f'old image center \n ({cx}, {cy})')
        # ax4.plot(x_centroid+shift_to_center[1], y_centroid+shift_to_center[0], 'ro', markersize=10, label = f'centroid shift to center \n ({x_centroid+shift_to_center[1]}, {y_centroid+shift_to_center[0]})')
        # ax4.plot(cx, cy, 'bs', markersize=10, label = f'old image center \n ({cx}, {cy})')
        # ax4.set_title(f'Gamma {Gamma} rotated')
        # ax4.legend()
        

    return rotated_img

# -----------------------------------------------------------------------
# SECTION 1: Tomography Analysis Functions
# -----------------------------------------------------------------------


def make_slab_mask(recon, film_thickness_voxels, film_center_y=None):
    """
    Build a binary slab mask that is 1 inside the film thickness band
    (in the Y / depth direction) and 0 outside.

    Parameters
    ----------
    recon : np.ndarray, shape (Z, Y, X)
    film_thickness_voxels : int   – physical thickness converted to voxels
    film_center_y : int or None   – row index of film centre; defaults to mid-plane

    Returns
    -------
    mask  : np.ndarray, same shape as recon, dtype float32
    y_lo  : int  – first masked row
    y_hi  : int  – last masked row  (inclusive)
    """
    nz, ny, nx = recon.shape
    if film_center_y is None:
        film_center_y = ny // 2                    # assume film centred in FOV

    half = film_thickness_voxels // 2
    y_lo = max(0,      film_center_y - half)
    y_hi = min(ny - 1, film_center_y + half)

    mask = np.zeros_like(recon, dtype=np.float32)
    mask[:, y_lo:y_hi + 1, :] = 1.0               # slab in Z-Y-X volume
    return mask, y_lo, y_hi

def plot_crosssections_compare(k, recon_free, recon_msk, y_lo, y_hi):
    """
    Side-by-side cross-sections for slice k:
      row 0 – XY plane (looking down Z)
      row 1 – XZ plane (looking down Y)  ← shows film thickness band
      row 2 – YZ plane (looking down X)  ← shows film thickness band
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    titles_free = ["XY", "XZ", "YZ"]
    titles_msk  = ["Masked XY", "XZ", "Masked YZ"]

    slices_free = [
        recon_free[k, :, :],          # XY at slice k
        recon_free[:, k, :],          # XZ at row k
        recon_free[:, :, k],          # YZ at col k
    ]
    slices_msk = [
        recon_msk[k, :, :],
        recon_msk[:, k, :],
        recon_msk[:, :, k],
    ]

    # vmin = recon_free.min()
    # vmax = recon_free.max()

    for row in range(3):
        for col, (sl, title) in enumerate(
            [(slices_free[row], titles_free[row]),
             (slices_msk[row],  titles_msk[row])]
        ):

            ax = axes[row, col]
            im = ax.imshow(sl, cmap='viridis', origin='lower')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Overlay the mask boundaries on XZ and YZ views
            
            if row in (0, 2) and col == 0:
                if row == 0:
                    ax.axhline(y_lo, color='cyan', lw=0.5,  ls='--', label='mask edge')
                    ax.axhline(y_hi, color='cyan', lw=0.5, ls='--')
                    ax.legend(fontsize=7, loc='upper right')
                elif row == 2:
                    ax.axvline(y_lo, color='cyan', lw=0.5, ls='--', label='mask edge')
                    ax.axvline(y_hi, color='cyan', lw=0.5, ls='--')
                    ax.legend(fontsize=7, loc='upper right')
                    
            if row in (0, 2) and col == 1:
                if row == 0:
                    ax.axhline(y_lo, color='cyan', lw=0.5,  ls='--', label='mask edge')
                    ax.axhline(y_hi, color='cyan', lw=0.5, ls='--')
                    ax.legend(fontsize=7, loc='upper right')
                elif row == 2:
                    ax.axvline(y_lo, color='cyan', lw=0.5, ls='--', label='mask edge')
                    ax.axvline(y_hi, color='cyan', lw=0.5, ls='--')
                    ax.legend(fontsize=7, loc='upper right')

    plt.suptitle(f"Reconstruction, index k={k}", fontsize=13)
    plt.tight_layout()
    plt.show()
    
    # crop to masked section 
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    ax = axes[0]
    im = ax.imshow(slices_msk[0][y_lo:y_hi, 0:514], cmap='viridis', origin='lower')
    ax = axes[1]
    im = ax.imshow(slices_msk[2][0:514, y_lo:y_hi], cmap='viridis', origin='lower')
    # plt.suptitle(f"Zoomed in reconstruction k={k}", fontsize=13)
    plt.tight_layout()
    plt.show()


# def masked_sirt(rotated, ang, center, mask, n_iter=100):
#     """
#     SIRT reconstruction with the physical slab mask applied
#     after every iteration.  Requires tomopy ≥ 1.11 (astra backend).
#     """
#     recon_iter = tomopy.recon(
#         rotated, ang,
#         center=center,
#         algorithm='sirt',
#         num_iter=n_iter,
#         sinogram_order=False
#     )
#     # apply physical constraint after every pass  ← enforcement
#     recon_iter *= mask
#     return recon_iter

def masked_sirt(rotated, ang, center, mask, n_iter=100):
    """
    True iterative SIRT with physical slab mask enforced between
    every iteration. Runs tomopy.recon with num_iter=1 in a loop
    so the mask acts as a hard constraint on each update step.
    """
    nz, ny, nx = mask.shape

    # Initialise reconstruction to zero
    recon_iter = np.zeros((nz, ny, nx), dtype=np.float32)

    for i in range(n_iter):
        # One SIRT update step, warm-started from current estimate
        recon_iter = tomopy.recon(
            rotated, ang,
            center=center,
            algorithm='sirt',
            num_iter=1,
            sinogram_order=False,
            init_recon=recon_iter       # pass current estimate as starting point
        )

        # Enforce physical constraint: zero out everything outside the film
        recon_iter *= mask

        if i % 10 == 0:
            print(f"  Iteration {i:>4d}/{n_iter}  |  "
                  f"max density inside mask: {recon_iter[mask > 0].max():.4f}")

    return recon_iter



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