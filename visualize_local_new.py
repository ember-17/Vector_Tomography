"""
visualize_local.py
============
Shared utility functions for visualizing vector data

This module is imported in analysis notebooks as:
    import visualize_local as vis

All functions are then called as vis.function_name(...)

Author: Emma Bernard

-------------------------------------------------------------------------
NORMALIZED vs UN-NORMALIZED, and SMOOTHING conventions used throughout
-------------------------------------------------------------------------
Every field pair passed around as (field, field_n), e.g. ('mz', 'mzn'),
already carries an un-normalized version (field, e.g. 'mz') and a
normalized version (field_n, e.g. 'mzn'). This module now computes and
stores BOTH versions of every derived quantity automatically -- callers
no longer choose one or the other with a boolean flag. Downstream keys
follow this naming scheme:

    quantity                       normalized (default / unsuffixed)   un-normalized ('_raw')
    ---------------------------------------------------------------------------------------
    gradient of a grid              d{field_n}_dx                       d{field}_dx
    psi (azimuthal angle)           psi_grid, mx_psi_grid, my_psi_grid   psi_raw_grid, mx_psi_raw_grid, my_psi_raw_grid
    Neel/Bloch wall fractions       neel_pct_overall, bloch_pct_overall, ...   neel_pct_overall_raw, bloch_pct_overall_raw, ...
    energy densities                exchange_grid, dmi_grid, pma_grid   exchange_raw_grid, dmi_raw_grid, pma_raw_grid

SMOOTHING: build_grids() / straighten_and_store() / slice_straighten_and_store()
all accept a `smooth` flag + `sigma`. When smooth=True, a Gaussian filter
(sigma applied along the x-direction only, i.e. sigma=(0, sigma)) is applied
to every m*_grid / m*n_grid / theta_grid / M_mag grid immediately after it is
built -- BEFORE any other function (gradient, psi, energies, ...) ever sees it.
That means every later computation automatically operates on smoothed data,
for both the normalized and un-normalized branches, with no further action
needed downstream. The pre-smoothing versions are kept under a `_unsmoothed`
suffix in case they're needed for comparison/debugging.
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

import matplotlib.colors as mcolors

from scipy.ndimage import gaussian_filter
# ===========================================================

# -----------------------------------------------------------------------
# SECTION 0: Smoothing helper
# -----------------------------------------------------------------------
def smooth_grid(grid, sigma=2, axis=1):
    """
    Gaussian-smooth a 2D (n_show x n_cols) grid along one axis only
    (by default, along the x/column axis, leaving the eta/row axis
    untouched). Returns a new array; does not modify in place.
    """
    sigma_tuple = tuple(sigma if ax == axis else 0 for ax in range(grid.ndim))
    return gaussian_filter(grid, sigma=sigma_tuple)


# -----------------------------------------------------------------------
# SECTION 1: Process vector slices
# -----------------------------------------------------------------------
# def straighten_data(x, z, theta, mz, mzn, mnorm, n_show=10):
#     # ---------- group points into x-columns ----------
#     # unique_x[inv] reconstructs x; inv tells you which column each point belongs to
#     unique_x, inv = np.unique(x, return_inverse=True)
#     n_cols = len(unique_x)

#     # ---------- find local top/bottom z per column ----------
#     # This handles the fact that the slice imported from paraview may be tilted and sliced at an angle so the ..
#     # ..min and max points may be at a slant.
#     zmin_per_x = np.full(n_cols, np.inf)
#     zmax_per_x = np.full(n_cols, -np.inf)
#     np.minimum.at(zmin_per_x, inv, z)
#     np.maximum.at(zmax_per_x, inv, z)

#     # ---------- normalized thickness coordinate for every point ----------
#     #  THIS IS THE STRAIGHTENING STEP, eta takes each column (which may not be the same length) and..
#     # .. asign them a 0 to 1 for normalization 
#     zmin_i = zmin_per_x[inv]
#     zmax_i = zmax_per_x[inv]
#     eta = (z - zmin_i) / (zmax_i - zmin_i)   # 0 at bottom, 1 at top, per column

#     # ---------- choose evenly spaced eta levels to plot ----------
#     eta_targets = np.linspace(0, 1, n_show)

#     # ---------- interpolate theta(eta_target) within each column, build grid ----------
#     theta_grid = np.full((n_show, n_cols), np.nan)
#     mz_grid = np.full((n_show, n_cols), np.nan)
#     mzn_grid = np.full((n_show, n_cols), np.nan)
#     mnorm_grid = np.full((n_show, n_cols), np.nan)

#     for j in range(n_cols):
#         col_mask = (inv == j)
#         eta_col = eta[col_mask]
#         th_col = theta[col_mask]
#         mz_col = mz[col_mask]
#         mzn_col = mzn[col_mask]
#         mnorm_col = mnorm[col_mask]
#         order = np.argsort(eta_col)
#         eta_col, th_col, mz_col, mzn_col, mnorm_col = eta_col[order], th_col[order], mz_col[order], mzn_col[order], mnorm[order]
#         # np.interp needs increasing x; also leaves NaN outside if you want strictness
#         theta_grid[:, j] = np.interp(eta_targets, eta_col, th_col)
#         mz_grid[:, j] = np.interp(eta_targets, eta_col, mz_col)
#         mzn_grid[:, j] = np.interp(eta_targets, eta_col, mzn_col)
#         mnorm_grid[:, j] = np.interp(eta_targets, eta_col, mnorm_col)

#     return unique_x, theta_grid, mz_grid, mzn_grid, mnorm_grid, n_cols, eta_targets


def straighten_data(x, z, theta, mz, mzn, mnorm, n_show=10):
    # ---------- group points into x-columns ----------
    # unique_x[inv] reconstructs x; inv tells you which column each point belongs to
    unique_x, inv = np.unique(x, return_inverse=True)
    n_cols = len(unique_x)

    # ---------- find local top/bottom z per column ----------
    # This handles the fact that the slice imported from paraview may be tilted and sliced at an angle so the ..
    # ..min and max points may be at a slant.
    zmin_per_x = np.full(n_cols, np.inf)
    zmax_per_x = np.full(n_cols, -np.inf)
    np.minimum.at(zmin_per_x, inv, z)
    np.maximum.at(zmax_per_x, inv, z)

    # ---------- normalized thickness coordinate for every point ----------
    #  THIS IS THE STRAIGHTENING STEP, eta takes each column (which may not be the same length) and..
    # .. asign them a 0 to 1 for normalization 
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
        th_col = theta[col_mask]
        mz_col = mz[col_mask]
        mzn_col = mzn[col_mask]
        mnorm_col = mnorm[col_mask]
        order = np.argsort(eta_col)
        eta_col, th_col, mz_col, mzn_col, mnorm_col = eta_col[order], th_col[order], mz_col[order], mzn_col[order], mnorm[order]
        # np.interp needs increasing x; also leaves NaN outside if you want strictness
        theta_grid[:, j] = np.interp(eta_targets, eta_col, th_col)
        mz_grid[:, j] = np.interp(eta_targets, eta_col, mz_col)
        mzn_grid[:, j] = np.interp(eta_targets, eta_col, mzn_col)
        mnorm_grid[:, j] = np.interp(eta_targets, eta_col, mnorm_col)

    return unique_x, theta_grid, mz_grid, mzn_grid, mnorm_grid, n_cols, eta_targets



def _apply_smoothing_to_cfg(cfg, field_pairs, sigma):
    """
    Smooths every grid built by build_grids (theta_grid, M_mag, and both
    the normalized + un-normalized grid for every field pair) in place,
    stashing the pre-smoothed version under a '_unsmoothed' suffix.
    """
    grid_keys = ['theta_grid', 'M_mag']
    for field, field_n in field_pairs:
        grid_keys += [f'{field}_grid', f'{field_n}_grid']

    for key in grid_keys:
        if key in cfg:
            cfg[f'{key}_unsmoothed'] = cfg[key]
            cfg[key] = smooth_grid(cfg[key], sigma=sigma, axis=1)

    cfg['smoothed'] = True
    cfg['smooth_sigma'] = sigma
    return cfg


def build_grids(cfg, field_pairs, n_show, xrange=None, smooth=False, sigma=2):
    """
    Core grid-building 

    Every field pair gets BOTH its un-normalized ('{field}_grid') and
    normalized ('{field_n}_grid') grid stored

    If smooth=True, a Gaussian filter (sigma along x only) is applied to
    every resulting grid (normalized AND un-normalized) immediately, so
    every function called afterwards on this cfg sees smoothed data.
    """

    if xrange is not None:
        xmin, xmax = xrange
        mask = (cfg['x'] >= xmin) & (cfg['x'] <= xmax)
        out = dict(sample=cfg['sample'], coord=cfg['coord'])
        x_src, z_src, theta_src, mnorm_src = cfg['x'][mask], cfg['z'][mask], cfg['theta'][mask], cfg['mnorm'][mask]
        field_source = {f: cfg[f][mask] for pair in field_pairs for f in pair}
    else:
        out = cfg
        x_src, z_src, theta_src, mnorm_src = cfg['x'], cfg['z'], cfg['theta'], cfg['mnorm']
        field_source = {f: cfg[f] for pair in field_pairs for f in pair}

    for field, field_n in field_pairs:
        unique_x, theta_grid, grid, grid_n, mnorm_grid, n_cols, eta_targets = straighten_data(
            x_src, z_src, theta_src, field_source[field], field_source[field_n], mnorm_src, n_show=n_show
        )

        out.setdefault('unique_x', unique_x)
        out.setdefault('theta_grid', theta_grid)
        out.setdefault('n_cols', n_cols)
        out.setdefault('eta_targets', eta_targets)
        out.setdefault('M_mag', mnorm_grid)

        out[f'{field}_grid'] = grid
        out[f'{field_n}_grid'] = grid_n

    if smooth:
        _apply_smoothing_to_cfg(out, field_pairs, sigma)
    else:
        out['smoothed'] = False
        out['smooth_sigma'] = None

    return out


# def straighten_and_store(cfg, field_pairs, n_show, smooth=False, sigma=2):
#     """Full-profile version: builds grids in place on cfg (which already
#     holds the raw 'x'/'z'/'mx'/... arrays from get_slice)."""
#     return build_grids(cfg, field_pairs, n_show, xrange=None, smooth=smooth, sigma=sigma)


# def slice_straighten_and_store(cfg, xrange, field_pairs, n_show, smooth=False, sigma=2):
#     """Cropped domain-wall version: builds grids into a fresh dict limited
#     to the given x-range."""
#     return build_grids(cfg, field_pairs, n_show, xrange=xrange, smooth=smooth, sigma=sigma)


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

def gradient(cfg, field_pairs, colors, plot=True):
    """
    Computes d(field)/dx for BOTH the un-normalized and normalized grid of
    every field pair (no more 'normalized' selector -- both are always
    computed and stored). Assumes cfg[field+'_grid'] / cfg[field_n+'_grid']
    already reflect smoothed data if smoothing was requested upstream.

    Stores:
        d{field}_dx    (from the un-normalized grid)
        d{field_n}_dx  (from the normalized grid)
    """
    dx = cfg['unique_x'][1] - cfg['unique_x'][0]
    cfg.setdefault('dx', dx)

    for field, field_n in field_pairs:
        cfg[f'd{field}_dx'] = np.gradient(cfg[f'{field}_grid'], dx, axis=1)
        cfg[f'd{field_n}_dx'] = np.gradient(cfg[f'{field_n}_grid'], dx, axis=1)

    if plot:
        plot_profile_gradient(cfg, field_pairs, colors)

    return cfg


def hyperbolic(x, M0, x0, w):
    M = M0*np.tanh(((x - x0))/w)
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
    w = np.std(x) * 2  # start with this for now... doesn't really mean anything physically.
    guess = [M0, x0, w]
    fit = curve_fit(hyperbolic, x, y, p0=guess)

    return fit[0], fit[1]


def hyperbolic_noM0(x, x0, w):
    M0 = 0.1-(-0.1)
    M = M0*np.tanh(((x - x0))/w)
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
    w = np.std(x) * 2  # start with this for now... doesn't really mean anything physically.
    guess = [x0, w]
    fit = curve_fit(hyperbolic_noM0, x, y, p0=guess)

    return fit[0], fit[1]


def solve_for_a(x, y, height, slope):
    # Define the function to find roots for
    def equation(a):
        return height * np.tanh((x + 25 * a)/slope) - height * np.tanh((x - 25 * a)/slope) - y

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
    pixel_size_nm = 8  # voxel size in nm
    k = center
    min_x, max_x = subsect_x
    y_lo, y_hi = subsect_y

    colors = plt.cm.plasma(np.linspace(0, 1, len(shift_z)))

    linecut_z = []
    dw_width_bloch = []
    dw_center = []

    fig = plt.figure(figsize=(21, 6))
    ax1 = fig.add_subplot(1, 4, 1)
    cut = ax1.imshow(recon[k, :, :], cmap='viridis')
    ax1.set_title(f'Gamma = {Gamma}, YX')
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(center-10, center+10)
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
        dw_width_bloch.append(hyperbolic_fit[2])  # might need to be converted into nm
        dw_center.append(hyperbolic_fit[1])

        ax1.plot([p0[1], p1[1]], [p0[0], p1[0]], color=colors[i], label=f'shift {z} nm')

        # Overlay linecut in second subplot
        ax2.plot(x_fit, linecut_fit, color=colors[i], label=f'shift {z} nm', marker='o')
        ax2.plot(x_fit, hyperbolic_plot, color=colors[i], linewidth=3, linestyle='--', label=f'Hyperbolic Fit {sample}, {Gamma},\n DW width = {hyperbolic_fit[2]:.2f} nm')

    ax3.plot(shift_z, np.abs(dw_width_bloch), marker='o')
    ax4.plot(shift_z, dw_center, marker='o')

    plt.tight_layout()
    plt.show()

    return linecut_z, dw_width_bloch, dw_center

def plot_YZ_linecut(recon, subsect_x, subsect_y, shift_z, center, sample, Gamma):

    pixel_size_nm = 8  # voxel size in nm
    k = center
    min_x, max_x = subsect_x
    y_lo, y_hi = subsect_y

    colors = plt.cm.plasma(np.linspace(0, 1, len(shift_z)))

    linecut_z = []
    dw_width_bloch = []
    dw_center = []

    fig = plt.figure(figsize=(21, 6))
    ax1 = fig.add_subplot(1, 4, 1)
    cut = ax1.imshow(recon[:, :, k], cmap='viridis')
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
        dw_width_bloch.append(hyperbolic_fit[2])  # might need to be converted into nm
        dw_center.append(hyperbolic_fit[1])

        ax1.plot([p0[1], p1[1]], [p0[0], p1[0]], color=colors[i], label=f'shift {z} nm')

        # Overlay linecut in second subplot
        ax2.plot(x_fit, linecut_fit, color=colors[i], label=f'shift {z} nm', marker='o')
        ax2.plot(x_fit, hyperbolic_plot, color=colors[i], linewidth=3, linestyle='--', label=f'Hyperbolic Fit {sample}, {Gamma},\n DW width = {hyperbolic_fit[2]:.2f} nm')

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
    ax.legend(fontsize=8, ncol=2, title=r'fraction of local thickness', loc='upper left')
    ax.set_title(f"Magnitude \n full {cfg['coord']}-slice profile \n {cfg['sample']}")


def plot_profile_gradient(cfg, field_pairs, colors):
    """
    For each field pair, shows a 2x2 grid:
        [ un-normalized profile | un-normalized gradient ]
        [ normalized profile    | normalized gradient    ]
    so both versions are visible together at all times.
    """
    smooth_note = " (smoothed)" if cfg.get('smoothed') else ""

    for field, field_n in field_pairs:
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        for j, eta_v in enumerate(cfg['eta_targets']):
            ax[0, 0].plot(cfg['unique_x'], cfg[f'{field}_grid'][j, :], color=colors[j], lw=1.5,
                          label=fr'{eta_v*100:.0f}%')
            ax[0, 1].plot(cfg['unique_x'], cfg[f'd{field}_dx'][j, :], color=colors[j], lw=1.5,
                          label=fr'{eta_v*100:.0f}%')
            ax[1, 0].plot(cfg['unique_x'], cfg[f'{field_n}_grid'][j, :], color=colors[j], lw=1.5,
                          label=fr'{eta_v*100:.0f}%')
            ax[1, 1].plot(cfg['unique_x'], cfg[f'd{field_n}_dx'][j, :], color=colors[j], lw=1.5,
                          label=fr'{eta_v*100:.0f}%')

        ax[0, 0].set_ylabel(r'$M$')
        ax[0, 0].set_xlabel('x (pix)')
        ax[0, 0].legend(fontsize=8, ncol=2, title='fraction of local thickness', loc='upper right')
        ax[0, 0].set_title(f"{field} \n full {cfg['coord']}-DW profile{smooth_note}\n {cfg['sample']}")

        ax[0, 1].set_xlabel('x (pix)')
        ax[0, 1].set_ylabel(r'$dM(x)$/dx')
        ax[0, 1].legend(fontsize=8, ncol=2, title='fraction of local thickness', loc='upper right')
        ax[0, 1].set_title(f"{field} \n full {cfg['coord']}-DW Gradient{smooth_note}\n {cfg['sample']}")

        ax[1, 0].set_ylabel(r'$M / M_{norm}$')
        ax[1, 0].set_xlabel('x (pix)')
        ax[1, 0].legend(fontsize=8, ncol=2, title='fraction of local thickness', loc='upper right')
        ax[1, 0].set_title(f"{field_n} \n full {cfg['coord']}-DW profile{smooth_note}\n {cfg['sample']}")

        ax[1, 1].set_xlabel('x (pix)')
        ax[1, 1].set_ylabel(r'$dM(x)$/dx')
        ax[1, 1].legend(fontsize=8, ncol=2, title='fraction of local thickness', loc='upper right')
        ax[1, 1].set_title(f"{field_n} \n full {cfg['coord']}-DW Gradient{smooth_note}\n {cfg['sample']}")

        plt.tight_layout()
        plt.show()


def plot_full_plot(cfg, field_pairs, colors):
    ## Look at full slice, normalized and unnormalized

    for field, field_n in field_pairs:

        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        for j, eta_v in enumerate(cfg['eta_targets']):
            ax[0].plot(cfg['unique_x'], cfg[field + '_grid'][j, :], color=colors[j], lw=1.5, label=fr'{eta_v*100:.0f}%')

        ax[0].set_ylabel(r'$M$')
        ax[0].set_xlabel('x (pix)')
        ax[0].legend(fontsize=8, ncol=2, title=r'fraction of local thickness', loc='upper left')
        ax[0].set_title(f"{field} \n full {cfg['coord']}-slice profile \n {cfg['sample']}")

        for j, eta_v in enumerate(cfg['eta_targets']):
            ax[1].plot(cfg['unique_x'], cfg[field_n + '_grid'][j, :], color=colors[j], lw=1.5, label=fr'{eta_v*100:.0f}%')

        ax[1].set_ylabel(r'$M / M_{norm}$')
        ax[1].set_xlabel('x (pix)')
        ax[1].legend(fontsize=8, ncol=2, title=r'fraction of local thickness', loc='upper left')
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
    }

    psi = cfg['psi']

    if np.isscalar(psi):
        for ax, comp in zip(axes, ['dMz_dx', 'dMy_dx', 'dMx_dx']):
            ax.plot(x0, cfg[comp], color=cfg['color'], label=labels[comp])
    else:
        # hybrid: one line per psi value, colored by cmap
        for ax, comp in zip(axes, ['dMz_dx', 'dMy_dx', 'dMx_dx']):
            for psi_val, color, values in zip(psi, cfg['colors'], cfg[comp]):
                ax.plot(x0, values, color=color)

    for ax, comp in zip(axes, ['dMz_dx', 'dMy_dx', 'dMx_dx']):
        ax.set_xlabel(r'$x$ (pix)')
        ax.set_ylabel(f'${comp[0]}{comp[1]}_{comp[2]}$')
        ax.set_title(fr'${comp[0]}{comp[1]}_{comp[2]}(x)$ {wall.capitalize()} Wall Gradient')
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
    }

    psi = cfg['psi']

    if np.isscalar(psi):
        for ax, comp in zip(axes, ['Mz', 'My', 'Mx']):
            ax.plot(x0, cfg[comp], color=cfg['color'], label=labels[comp])
    else:
        # hybrid: one line per psi value, colored by cmap
        for ax, comp in zip(axes, ['Mz', 'My', 'Mx']):
            for psi_val, color, values in zip(psi, cfg['colors'], cfg[comp]):
                ax.plot(x0, values, color=color)

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


# -----------------------------------------------------------------------
# SECTION 4: Psi (azimuthal angle) -- normalized AND un-normalized
# -----------------------------------------------------------------------
def calc_psi(cfg, plot=True, colors=None):
    """
    Computes psi (azimuthal in-plane angle) from BOTH the normalized and
    un-normalized grids, always storing both:

        normalized (default names, unchanged from before):
            mx_psi_grid, my_psi_grid, psi_grid
        un-normalized ('_raw' suffix):
            mx_psi_raw_grid, my_psi_raw_grid, psi_raw_grid
    """
    sin_theta = np.sin(cfg['theta_grid'])
    if np.any(np.isclose(sin_theta, 0)):
        print(f"Warning ({cfg.get('sample')}, {cfg.get('coord')}): sin(theta) near zero at some points — psi undefined there")

    def _psi_from(mx_grid, my_grid):
        mx_component = np.clip(mx_grid / sin_theta, -1, 1)
        my_component = np.clip(my_grid / sin_theta, -1, 1)
        mx_psi = np.arccos(mx_component)
        my_psi = np.arcsin(my_component)
        psi = np.arctan2(my_component, mx_component)  # quadrant-correct reference
        return mx_psi, my_psi, psi

    # normalized branch (default / unsuffixed names)
    cfg['mx_psi_grid'], cfg['my_psi_grid'], cfg['psi_grid'] = _psi_from(cfg['mxn_grid'], cfg['myn_grid'])

    # un-normalized branch ('_raw' suffix)
    cfg['mx_psi_raw_grid'], cfg['my_psi_raw_grid'], cfg['psi_raw_grid'] = _psi_from(cfg['mx_grid'], cfg['my_grid'])

    if plot:
        plot_psi(cfg, colors if colors is not None else plt.cm.viridis(np.linspace(0, 1, len(cfg['eta_targets']))))

    return cfg


def plot_psi(cfg, colors, normalized=True, n_bins=11):
    """Plots ψ computed from Mx, My, and atan2, for either the normalized
    (default) or un-normalized branch."""
    suffix = '' if normalized else '_raw'
    label_tag = 'normalized' if normalized else 'un-normalized'

    fig, ax = plt.subplots(1, 3, figsize=(25, 8))
    x_data = cfg['unique_x'] - cfg['unique_x'].mean()

    bin_edges = np.linspace(-1, 1, n_bins + 1)
    cmap = plt.cm.twilight_shifted
    bin_colors = cmap(np.linspace(0, 1, n_bins))

    for j, eta_v in enumerate(cfg['eta_targets']):
        ax[0].plot(x_data, cfg[f'mx_psi{suffix}_grid'][j, :] / np.pi, color=colors[j], lw=1.5,
                   marker='o', label=fr'ψ from $M_x$, {eta_v*100:.0f}%')
        ax[1].plot(x_data, cfg[f'my_psi{suffix}_grid'][j, :] / np.pi, color=colors[j], lw=1.5,
                   marker='s', label=fr'ψ from $M_y$, {eta_v*100:.0f}%')
        ax[2].plot(x_data, cfg[f'psi{suffix}_grid'][j, :] / np.pi, color=colors[j], lw=2.5,
                   marker='o', linestyle=':', label=fr'ψ (atan2), {eta_v*100:.0f}%')

    for a in ax:
        for k in range(n_bins):
            a.axhspan(bin_edges[k], bin_edges[k+1], color=bin_colors[k], alpha=0.15, zorder=0)

    ax[0].set_ylabel(r'ψx(x) (1/π)')
    ax[1].set_ylabel(r'ψy(x) (1/π)')
    ax[0].set_xlabel('x (pix)')
    ax[0].set_ylim(-1, 1)
    ax[0].axhline(0, color='cyan', label='Néel wall')
    ax[0].axhline(0.5, color='magenta', label='Bloch wall')
    ax[0].axhline(-0.5, color='magenta')
    ax[0].legend(fontsize=7, ncol=2, title='fraction of local thickness', loc='lower left')
    ax[0].set_title(f"Calculated ψ per x ({label_tag}) \n {cfg['sample']} {cfg['coord']}-slice")

    ax[1].set_xlabel('x (pix)')
    ax[1].set_ylim(-1, 1)
    ax[1].axhline(0, color='cyan', label='Néel wall')
    ax[1].axhline(0.5, color='magenta', label='Bloch wall')
    ax[1].axhline(-0.5, color='magenta')
    ax[1].legend(fontsize=7, ncol=2, title='fraction of local thickness', loc='lower left')
    ax[1].set_title(f"Calculated ψ per x ({label_tag}) \n {cfg['sample']} {cfg['coord']}-slice")

    ax[2].set_ylabel(r'ψ(x) (1/π)')
    ax[2].set_xlabel('x (pix)')
    ax[2].set_ylim(-1, 1)
    ax[2].axhline(0, color='cyan', label='Néel wall')
    ax[2].axhline(0.5, color='magenta', label='Bloch wall')
    ax[2].axhline(-0.5, color='magenta')
    ax[2].legend(fontsize=7, ncol=2, title='fraction of local thickness', loc='lower left')
    ax[2].set_title(f"Calculated ψ per x ({label_tag}) \n {cfg['sample']}")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------
# SECTION 5: Neel / Bloch wall-type fractions -- normalized AND un-normalized
# -----------------------------------------------------------------------
def calc_neel_bloch_stats(cfg, psi_tol=0.05, normalized=True):
    """
    Classifies each (eta, x) point as Neel-like (psi/pi ~ 0 or +/-1) or
    Bloch-like (psi/pi ~ +/-0.5), from the psi grid computed by calc_psi().

    normalized=True  -> reads cfg['psi_grid'] (from normalized components),
                         stores unsuffixed keys (matches original behavior).
    normalized=False -> reads cfg['psi_raw_grid'] (from un-normalized
                         components), stores '_raw'-suffixed keys.

    Stores (suffix = '' if normalized else '_raw'):
        neel_pct_overall{suffix}, bloch_pct_overall{suffix}
        neel_pct_per_eta{suffix}, bloch_pct_per_eta{suffix}
    """
    suffix = '' if normalized else '_raw'
    psi_key = 'psi_grid' if normalized else 'psi_raw_grid'
    psi_over_pi = cfg[psi_key] / np.pi

    n_slices = len(cfg['eta_targets'])
    neel_pct_per_eta = np.zeros(n_slices)
    bloch_pct_per_eta = np.zeros(n_slices)

    for j in range(n_slices):
        psi_slice = psi_over_pi[j, :]
        neel_mask = (np.abs(psi_slice) <= psi_tol) | (np.isclose(np.abs(psi_slice), 1, atol=psi_tol))
        bloch_mask = np.isclose(np.abs(psi_slice), 0.5, atol=psi_tol)
        neel_pct_per_eta[j] = 100 * neel_mask.sum() / psi_slice.size
        bloch_pct_per_eta[j] = 100 * bloch_mask.sum() / psi_slice.size

    neel_mask_all = (np.abs(psi_over_pi) <= psi_tol) | (np.isclose(np.abs(psi_over_pi), 1, atol=psi_tol))
    bloch_mask_all = np.isclose(np.abs(psi_over_pi), 0.5, atol=psi_tol)
    neel_pct_overall = 100 * neel_mask_all.sum() / psi_over_pi.size
    bloch_pct_overall = 100 * bloch_mask_all.sum() / psi_over_pi.size

    cfg[f'neel_pct_overall{suffix}'] = neel_pct_overall
    cfg[f'bloch_pct_overall{suffix}'] = bloch_pct_overall
    cfg[f'neel_pct_per_eta{suffix}'] = neel_pct_per_eta
    cfg[f'bloch_pct_per_eta{suffix}'] = bloch_pct_per_eta

    return cfg


# def plot_psi_bin_offset_w_highlight(cfg, select_psi='', normalized=True, n_bins=11, offset=1.5, psi_tol=0.05, colors=None):
#     """
#     Plots Mz/My/Mx (offset per eta slice), colored by psi bin, with
#     Neel-like points highlighted cyan and Bloch-like points highlighted
#     magenta. Expects calc_neel_bloch_stats(cfg, normalized=normalized) to
#     already have been run (so this function only plots, it doesn't
#     recompute the underlying statistics).
#     """

def plot_psi_bin_offset_w_highlight(cfg, select_psi='', n_bins=11, offset=1.5, psi_tol=0.05, colors=None, normalized=True):
    psi_key = select_psi + 'psi_grid'
    if psi_key not in cfg:
        cfg = calc_psi(cfg)

    suffix = '' if normalized else '_raw'
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(cfg['eta_targets'])))

    x_data = cfg['unique_x'] - cfg['unique_x'].mean()
    psi_key = select_psi + f'psi{suffix}_grid'
    psi_over_pi = cfg[psi_key] / np.pi

    bin_edges = np.linspace(-1, 1, n_bins + 1)
    cmap = plt.cm.twilight_shifted
    bin_colors = cmap(np.linspace(0, 1, n_bins))

    if normalized:
        x_k, y_k, z_k = 'mxn_grid', 'myn_grid', 'mzn_grid'
    else:
        x_k, y_k, z_k = 'mx_grid', 'my_grid', 'mz_grid'

    fig, ax = plt.subplots(1, 3, figsize=(26, 8))

    for j, eta_v in enumerate(cfg['eta_targets']):
        psi_slice = psi_over_pi[j, :]
        shift = j * offset

        mzn = cfg[z_k][j, :] + shift
        myn = cfg[y_k][j, :] + shift
        mxn = cfg[x_k][j, :] + shift

        ax[0].plot(x_data, mzn, c=colors[j], linewidth=1)
        ax[1].plot(x_data, myn, c=colors[j], linewidth=1)
        ax[2].plot(x_data, mxn, c=colors[j], linewidth=1)

        bin_idx = np.digitize(psi_slice, bin_edges[1:-1])
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        point_colors = bin_colors[bin_idx].copy()

        neel_mask = (np.abs(psi_slice) <= psi_tol) | (np.isclose(np.abs(psi_slice), 1, atol=psi_tol))
        bloch_mask = np.isclose(np.abs(psi_slice), 0.5, atol=psi_tol)
        point_colors[neel_mask] = mcolors.to_rgba('cyan')
        point_colors[bloch_mask] = mcolors.to_rgba('magenta')

        ax[0].scatter(x_data, mzn, c=point_colors, s=20, zorder=3)
        ax[1].scatter(x_data, myn, c=point_colors, s=20, zorder=3)
        ax[2].scatter(x_data, mxn, c=point_colors, s=20, zorder=3)

    neel_pct_overall = cfg.get(f'neel_pct_overall{suffix}')
    bloch_pct_overall = cfg.get(f'bloch_pct_overall{suffix}')
    neel_pct_per_eta = cfg.get(f'neel_pct_per_eta{suffix}')
    bloch_pct_per_eta = cfg.get(f'bloch_pct_per_eta{suffix}')

    ax[0].set_ylabel(r'$M_{z}$ (offset per slice)')
    ax[1].set_ylabel(r'$M_{y}$ (offset per slice)')
    ax[2].set_ylabel(r'$M_{x}$ (offset per slice)')
    for a in ax:
        a.set_xlabel('x (pix)')
        a.set_yticks([])

    if neel_pct_overall is not None:
        per_eta_handles = [
            plt.Line2D([0], [0], color=colors[j],
                       label=fr'{eta_v*100:.0f}%: Néel {neel_pct_per_eta[j]:.1f}% vs. Bloch {bloch_pct_per_eta[j]:.1f}%')
            for j, eta_v in enumerate(cfg['eta_targets'])
        ]
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10,
                       label=fr'Néel-like (ψ≈0,±1) — overall: {neel_pct_overall:.1f}%'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=10,
                       label=fr'Bloch-like (ψ≈±0.5) — overall: {bloch_pct_overall:.1f}%'),
        ] + list(reversed(per_eta_handles))
        ax[2].legend(handles=legend_handles, fontsize=15, loc='upper left',
                     bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    tag = 'normalized' if normalized else 'un-normalized'
    fig.suptitle(f"{cfg['sample']} {cfg['coord']}-slice, colored by {select_psi} ψ ({tag})")

    plt.tight_layout()
    plt.show()
    return cfg


def plot_neel_bloch_comparison(samples, sample_ids, normalized=True, offset=15):
    """
    Reproduces the original 4-panel (y per-eta / y overall bar / x per-eta /
    x overall bar) Neel-vs-Bloch summary figure, reading from either the
    normalized (default) or un-normalized branch of calc_neel_bloch_stats.
    """
    suffix = '' if normalized else '_raw'
    tag = 'normalized' if normalized else 'un-normalized'

    fig, ax = plt.subplots(2, 2, figsize=(16, 14))

    overall_neel, overall_bloch = [], []
    overall_neel_err, overall_bloch_err = [], []
    overall_neelx, overall_blochx = [], []
    overall_neel_errx, overall_bloch_errx = [], []

    for s_idx, sample in enumerate(sample_ids):
        shift = s_idx * offset

        for coord, a in zip(['y', 'x'], [ax[0, 0], ax[1, 0]]):
            cfg = samples[sample][coord]
            eta_pct = cfg['eta_targets'] * 100
            n_cols = cfg['n_cols']

            p_neel = cfg[f'neel_pct_per_eta{suffix}'] / 100
            p_bloch = cfg[f'bloch_pct_per_eta{suffix}'] / 100
            neel_err = np.sqrt(p_neel * (1 - p_neel) / n_cols) * 100
            bloch_err = np.sqrt(p_bloch * (1 - p_bloch) / n_cols) * 100

            a.errorbar(eta_pct, cfg[f'neel_pct_per_eta{suffix}'] + shift, yerr=neel_err,
                       color='cyan', marker='o', lw=1.5, capsize=3)
            a.errorbar(eta_pct, cfg[f'bloch_pct_per_eta{suffix}'] + shift, yerr=bloch_err,
                       color='magenta', marker='o', lw=1.5, capsize=3)

        for coord, a in zip(['y', 'x'], [ax[0, 0], ax[1, 0]]):
            cfg = samples[sample][coord]
            avg_plot = (cfg[f'neel_pct_per_eta{suffix}'][0] + shift + cfg[f'bloch_pct_per_eta{suffix}'][0] + shift) / 2
            a.axhline(avg_plot, color='k', lw=1, alpha=0.3, zorder=0)
            a.annotate(fr'{sample}', xy=(0.15, avg_plot + 2), xycoords=('axes fraction', 'data'),
                       xytext=(-8, 0), textcoords='offset points',
                       ha='right', va='center', fontsize=15, color='k')

        cfg_y = samples[sample]['y']
        n_total = cfg_y['n_cols'] * len(cfg_y['eta_targets'])
        p_neel_all = cfg_y[f'neel_pct_overall{suffix}'] / 100
        p_bloch_all = cfg_y[f'bloch_pct_overall{suffix}'] / 100
        overall_neel.append(cfg_y[f'neel_pct_overall{suffix}'])
        overall_bloch.append(cfg_y[f'bloch_pct_overall{suffix}'])
        overall_neel_err.append(np.sqrt(p_neel_all * (1 - p_neel_all) / n_total) * 100)
        overall_bloch_err.append(np.sqrt(p_bloch_all * (1 - p_bloch_all) / n_total) * 100)

        cfg_x = samples[sample]['x']
        n_totalx = cfg_x['n_cols'] * len(cfg_x['eta_targets'])
        p_neel_allx = cfg_x[f'neel_pct_overall{suffix}'] / 100
        p_bloch_allx = cfg_x[f'bloch_pct_overall{suffix}'] / 100
        overall_neelx.append(cfg_x[f'neel_pct_overall{suffix}'])
        overall_blochx.append(cfg_x[f'bloch_pct_overall{suffix}'])
        overall_neel_errx.append(np.sqrt(p_neel_allx * (1 - p_neel_allx) / n_totalx) * 100)
        overall_bloch_errx.append(np.sqrt(p_bloch_allx * (1 - p_bloch_allx) / n_totalx) * 100)

    ax[0, 0].set_title(f'y-slice ({tag})')
    ax[0, 0].set_xlabel('Fraction of local thickness (%)')
    ax[0, 0].set_ylabel('DW type (%)')

    color_handles = [
        plt.Line2D([0], [0], color='cyan', lw=2, label='% Néel'),
        plt.Line2D([0], [0], color='magenta', lw=2, label='% Bloch'),
    ]
    ax[0, 0].legend(handles=color_handles, fontsize=8, loc='upper left')

    x_pos = np.arange(len(sample_ids))
    width = 0.35
    ax[0, 1].bar(x_pos - width/2, overall_neel, width, yerr=overall_neel_err, capsize=4,
                 color='cyan', alpha=0.6, label='% Néel')
    ax[0, 1].bar(x_pos + width/2, overall_bloch, width, yerr=overall_bloch_err, capsize=4,
                 color='magenta', alpha=0.6, label='% Bloch')
    ax[0, 1].set_xticks(x_pos)
    ax[0, 1].set_xticklabels(sample_ids)
    ax[0, 1].set_xlabel('sample')
    ax[0, 1].set_ylabel('DW type (%)')
    ax[0, 1].set_title(f'Overall Néel/Bloch % per sample\ny-slice ({tag})')
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].set_title(f'x-slice ({tag})')
    ax[1, 0].set_xlabel('Fraction of local thickness (%)')
    ax[1, 0].set_ylabel('DW type (%)')
    ax[1, 0].legend(handles=color_handles, fontsize=8, loc='upper left')

    ax[1, 1].bar(x_pos - width/2, overall_neelx, width, yerr=overall_neel_errx, capsize=4,
                 color='cyan', alpha=0.6, label='% Néel')
    ax[1, 1].bar(x_pos + width/2, overall_blochx, width, yerr=overall_bloch_errx, capsize=4,
                 color='magenta', alpha=0.6, label='% Bloch')
    ax[1, 1].set_xticks(x_pos)
    ax[1, 1].set_xticklabels(sample_ids)
    ax[1, 1].set_xlabel('sample')
    ax[1, 1].set_ylabel('DW type (%)')
    ax[1, 1].set_title(f'Overall Néel/Bloch % per sample\nx-slice ({tag})')
    ax[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_neel_bloch_norm_vs_raw(samples, sample_ids, offset=15):
    """
    NEW: side-by-side comparison of the overall % Neel / % Bloch computed
    from the NORMALIZED vector components vs the UN-NORMALIZED components,
    for both the y- and x-slices. This is the final "does normalization
    change the wall-type classification" sanity check.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    x_pos = np.arange(len(sample_ids))
    width = 0.2

    for a, coord in zip(ax, ['y', 'x']):
        neel_norm, bloch_norm = [], []
        neel_raw, bloch_raw = [], []

        for sample in sample_ids:
            cfg = samples[sample][coord]
            neel_norm.append(cfg['neel_pct_overall'])
            bloch_norm.append(cfg['bloch_pct_overall'])
            neel_raw.append(cfg['neel_pct_overall_raw'])
            bloch_raw.append(cfg['bloch_pct_overall_raw'])

        a.bar(x_pos - 1.5*width, neel_norm, width, color='cyan', alpha=0.9, label='% Néel (normalized)')
        a.bar(x_pos - 0.5*width, neel_raw, width, color='cyan', alpha=0.4, hatch='//', label='% Néel (un-normalized)')
        a.bar(x_pos + 0.5*width, bloch_norm, width, color='magenta', alpha=0.9, label='% Bloch (normalized)')
        a.bar(x_pos + 1.5*width, bloch_raw, width, color='magenta', alpha=0.4, hatch='//', label='% Bloch (un-normalized)')

        a.set_xticks(x_pos)
        a.set_xticklabels(sample_ids)
        a.set_xlabel('sample')
        a.set_ylabel('DW type (%)')
        a.set_title(f'{coord}-slice: normalized vs un-normalized')
        a.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------
# SECTION 6: Spatial energy densities -- normalized AND un-normalized
# -----------------------------------------------------------------------
def calc_energies(cfg, coord, A=1.0, D=1.0, K=1.0):
    """
    Computes exchange, DMI, and PMA energy densities per x-position, per
    eta slice, for BOTH the normalized and un-normalized magnetization
    grids. Assumes cfg already has 'mx_grid'/'mxn_grid' etc. (and their
    'd..._dx' gradients from gradient()).

    Exchange: -A * (dm/dx)^2   [component depends on slice coord]
    DMI:       D * [m(x) x m(x+dx)]_component
    PMA:      -K * mz^2

    Normalized results stored as before (exchange_grid, dmi_grid, pma_grid).
    Un-normalized results stored with a '_raw' suffix.
    """
    def _compute(mx, my, mz, dm_dx):
        if coord == 'x':
            my_shift = np.roll(my, -1, axis=1)
            mz_shift = np.roll(mz, -1, axis=1)
            cross = my * mz_shift - mz * my_shift
        elif coord == 'y':
            mx_shift = np.roll(mx, -1, axis=1)
            mz_shift = np.roll(mz, -1, axis=1)
            cross = mz * mx_shift - mx * mz_shift
        else:
            raise ValueError(f"Unrecognized coord '{coord}' — expected 'x' or 'y'")

        exchange_grid = -A * (dm_dx ** 2)
        dmi_grid = D * cross
        dmi_grid[:, -1] = np.nan  # last column wraps via np.roll — not physically valid
        pma_grid = -K * (mz ** 2)
        return exchange_grid, dmi_grid, pma_grid

    dm_dx_key = 'dmxn_dx' if coord == 'x' else 'dmyn_dx'
    dm_dx_raw_key = 'dmx_dx' if coord == 'x' else 'dmy_dx'

    # normalized branch (default / unsuffixed names, unchanged from before)
    exchange_grid, dmi_grid, pma_grid = _compute(
        cfg['mxn_grid'], cfg['myn_grid'], cfg['mzn_grid'], cfg[dm_dx_key]
    )
    cfg['exchange_grid'] = exchange_grid
    cfg['dmi_grid'] = dmi_grid
    cfg['pma_grid'] = pma_grid

    # un-normalized branch ('_raw' suffix)
    exchange_raw_grid, dmi_raw_grid, pma_raw_grid = _compute(
        cfg['mx_grid'], cfg['my_grid'], cfg['mz_grid'], cfg[dm_dx_raw_key]
    )
    cfg['exchange_raw_grid'] = exchange_raw_grid
    cfg['dmi_raw_grid'] = dmi_raw_grid
    cfg['pma_raw_grid'] = pma_raw_grid

    cfg['A'], cfg['D'], cfg['K'] = A, D, K

    return cfg


def plot_energies(cfg, sample, coord='y', offset=None, normalized=True, colors=None):
    """
    cfg is the grid-dict that calc_energies() was run on directly (e.g.
    crop1[sample][coord]['dw']) -- not a nested samples/crop1 wrapper.
    `sample`/`coord` are only used for the figure title.
    """
    tag = 'normalized' if normalized else 'un-normalized'

    if normalized:
        energy_keys = ['pma_grid', 'exchange_grid', 'dmi_grid']
    else:
        energy_keys = ['pma_raw_grid', 'exchange_raw_grid', 'dmi_raw_grid']
    offset_shift = [1, 20, 4]
    energy_labels = [r'PMA', r'Exchange', r'DMI']

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))

    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(cfg['eta_targets'])))
    x_data = cfg['unique_x'] - cfg['unique_x'].mean()
    linestyle = '-'

    for j, eta_v in enumerate(cfg['eta_targets']):
        for a, key, ct in zip(ax, energy_keys, enumerate(offset_shift)):
            shift = j * (offset if offset is not None else 0) / offset_shift[ct[0]]
            a.axhline(shift, color=colors[j], lw=1, alpha=0.3, zorder=0)
            a.annotate(fr'{eta_v*100:.0f}%', xy=(0.15, shift-(0.15/offset_shift[ct[0]])), xycoords=('axes fraction', 'data'),
                       xytext=(-8, 0), textcoords='offset points',
                       ha='right', va='center', fontsize=15, color=colors[j])

            values = cfg[key][j, :] + shift
            a.plot(x_data, values, color=colors[j], marker='o', linestyle=linestyle, lw=1.5)

    for a, label in zip(ax, energy_labels):
        a.set_xlabel('x (pix)')
        a.set_ylabel('Energy density (a.u.)')
        a.set_title(label)
        if offset is not None:
            a.set_yticks([])

    fig.suptitle(f'Calculated Energy Densities Across Slice ({tag}) \n {sample} {coord}-slice')
    plt.tight_layout()
    plt.show()
