# --- Cell: imports ---

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import zoom

from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import marching_cubes
import pyvista as pv

import matplotlib.ticker as mticker
import pandas as pd

#######################################################################
# Visualize 3D Volume and Cropped Volume 

#######################################################################
def crop_volume_by_thickness(sample, film_thickness_nm, voxel_size_nm = 8., verbose=True):
    """
    Crop a loaded sample dict in z to the true film thickness, symmetric
    around `center1` (a z-voxel index into the ORIGINAL, uncropped volume).

    Parameters
    ----------
    sample : dict
        Output of load_vti_sample().
    film_thickness_nm : float
        True material thickness in physical units (same units as z_coords).
    dz : float or None
        Voxel spacing in z. If None, computed from sample['z_coords']
        (robust to slightly non-uniform rectilinear spacing via median diff).

    Returns
    -------
    dict : same structure as `sample`, all z-axis arrays cropped, plus:
        'crop_z_index_range'      : (z_lo, z_hi) indices into the ORIGINAL volume
        'crop_n_voxels_z'         : number of z voxels kept
        'crop_dz'                 : voxel spacing used
        'crop_target_thickness_nm': requested thickness
        'crop_actual_thickness_nm': n_voxels * dz (won't exactly match target
                                     -- voxel count must be an integer)
        'crop_center1'            : center1 used
    """
    nz, ny, nx = sample['shape']
    print(nz, ny, nx)
    z_coords = sample['z_coords']
    print(z_coords)

    center =  z_coords[(nz//2)]
    print(center)

    n_vox = film_thickness_nm / voxel_size_nm
    print(n_vox)

    half = n_vox / 2.0
    z_lo = int(round(center - half))
    z_hi = int(round(center + half))  # exclusive
    print(z_lo, z_hi)

    z_lo_c, z_hi_c = max(z_lo, 0), min(z_hi, nz)
    if (z_lo_c, z_hi_c) != (z_lo, z_hi):
        print(f"[crop_volume_by_thickness] warning: requested crop "
              f"[{z_lo}:{z_hi}] exceeds volume bounds [0:{nz}] -- "
              f"clamped to [{z_lo_c}:{z_hi_c}]")
    z_lo, z_hi = z_lo_c, z_hi_c #+ 1
    n_crop = z_hi - z_lo

    cropped = dict(sample)  # shallow copy; array entries overwritten below
    for key, val in sample.items():
        if not isinstance(val, np.ndarray):
            continue
        if key == 'z_coords':
            cropped[key] = val[z_lo:z_hi].copy()
        elif val.ndim == 3 and val.shape == (nz, ny, nx):
            cropped[key] = val[z_lo:z_hi, :, :].copy()
        elif val.ndim == 4 and val.shape[:3] == (nz, ny, nx):
            cropped[key] = val[z_lo:z_hi, :, :, :].copy()
        # else: leave untouched (e.g. x_coords, y_coords -- unaffected by a z-crop)

    print(n_crop, ny, nx)
    cropped['shape'] = (n_crop, ny, nx)
    cropped['dims'] = (nx, ny, n_crop)
    cropped['crop_z_index_range'] = (z_lo, z_hi)
    cropped['crop_n_voxels_z'] = n_crop
    cropped['crop_target_thickness_nm'] = film_thickness_nm
    cropped['crop_actual_thickness_nm'] = n_crop * voxel_size_nm
    cropped['crop_center1'] = center

    if verbose:
        print(f"[{sample.get('name','sample')}] z crop: [{z_lo}:{z_hi}) -> "
              f"actual thickness = {n_crop*voxel_size_nm:.2f} nm "
              f"(target {film_thickness_nm} nm)")
        total_full = nz * ny * nx
        total_crop = n_crop * ny * nx
        print(f"    total voxels: {total_full} (full) -> {total_crop} (cropped), "
              f"{100*total_crop/total_full:.1f}% retained")

    return cropped



def crop_interpolate_volume_by_thickness(sample, film_thickness_nm, voxel_size_nm=8.,
                              target_voxel_size_nm=2., verbose=True):
    """
    Interpolate a loaded sample dict from `voxel_size_nm` down to
    `target_voxel_size_nm` (isotropic, all three axes), then crop in z to
    the true film thickness, symmetric around the volume's z-center index.

    Interpolating first means the crop boundary can land on a much finer
    grid (e.g. 2nm increments instead of 8nm), so the returned thickness
    can be much closer to `film_thickness_nm`.

    Parameters
    ----------
    sample : dict
        Output of load_vti_sample().
    film_thickness_nm : float
        True material thickness in physical units (same units as z_coords).
    voxel_size_nm : float
        Original (input) isotropic voxel spacing.
    target_voxel_size_nm : float
        Voxel spacing to interpolate to before cropping. Must evenly divide
        (or at least be commensurate with) voxel_size_nm for a clean zoom
        factor; e.g. 8 -> 2 gives a scale factor of 4.
    verbose : bool
        Print progress/diagnostics.

    Returns
    -------
    dict : same structure as `sample`, all spatial arrays interpolated to
        `target_voxel_size_nm` and cropped in z, plus:
        'crop_z_index_range'       : (z_lo, z_hi) indices into the INTERPOLATED volume
        'crop_n_voxels_z'          : number of z voxels kept (at target_voxel_size_nm)
        'crop_dz'                  : voxel spacing used for the crop (target_voxel_size_nm)
        'crop_orig_voxel_size_nm'  : voxel spacing before interpolation
        'crop_target_thickness_nm' : requested thickness
        'crop_actual_thickness_nm' : n_voxels * target_voxel_size_nm
        'crop_center1'             : center z-coordinate used
    """
    nz, ny, nx = sample['shape']
    scale = voxel_size_nm / target_voxel_size_nm

    if verbose:
        print(f"Original shape: {(nz, ny, nx)}, voxel size: {voxel_size_nm} nm")
        print(f"Interpolating to {target_voxel_size_nm} nm voxels (scale x{scale:g})...")

    # --- Step 1: interpolate everything (coords + data arrays) to the finer grid ---
    interpolated = dict(sample)
    for key, val in sample.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ('x_coords', 'y_coords', 'z_coords'):
            interpolated[key] = zoom(val, scale, order=1)
        elif val.ndim == 3 and val.shape == (nz, ny, nx):
            interpolated[key] = zoom(val, scale, order=1)
        elif val.ndim == 4 and val.shape[:3] == (nz, ny, nx):
            # don't interpolate a trailing non-spatial (e.g. vector/channel) axis
            interpolated[key] = zoom(val, (scale, scale, scale, 1), order=1)
        # else: leave untouched

    z_coords = interpolated['z_coords']
    nz_i = z_coords.shape[0]
    ny_i = interpolated['y_coords'].shape[0] if 'y_coords' in interpolated else ny
    nx_i = interpolated['x_coords'].shape[0] if 'x_coords' in interpolated else nx

    if verbose:
        print(f"Interpolated shape: {(nz_i, ny_i, nx_i)}")

    # --- Step 2: crop in z at the new (finer) resolution ---
    center = z_coords[nz_i // 2]
    if verbose:
        print(f"z-center: {center}")

    n_vox = film_thickness_nm / target_voxel_size_nm
    half = n_vox / 2.0
    z_lo = int(round((nz_i // 2) - half))
    z_hi = int(round((nz_i // 2) + half))  # exclusive

    z_lo_c, z_hi_c = max(z_lo, 0), min(z_hi, nz_i)
    if (z_lo_c, z_hi_c) != (z_lo, z_hi):
        print(f"[crop_volume_by_thickness] warning: requested crop "
              f"[{z_lo}:{z_hi}] exceeds volume bounds [0:{nz_i}] -- "
              f"clamped to [{z_lo_c}:{z_hi_c}]")
    z_lo, z_hi = z_lo_c, z_hi_c
    n_crop = z_hi - z_lo

    cropped = dict(interpolated)  # shallow copy; array entries overwritten below
    for key, val in interpolated.items():
        if not isinstance(val, np.ndarray):
            continue
        if key == 'z_coords':
            cropped[key] = val[z_lo:z_hi].copy()
        elif val.ndim == 3 and val.shape == (nz_i, ny_i, nx_i):
            cropped[key] = val[z_lo:z_hi, :, :].copy()
        elif val.ndim == 4 and val.shape[:3] == (nz_i, ny_i, nx_i):
            cropped[key] = val[z_lo:z_hi, :, :, :].copy()
        # else: leave untouched (e.g. x_coords, y_coords -- unaffected by a z-crop)

    cropped['shape'] = (n_crop, ny_i, nx_i)
    cropped['dims'] = (nx_i, ny_i, n_crop)
    cropped['crop_z_index_range'] = (z_lo, z_hi)
    cropped['crop_n_voxels_z'] = n_crop
    cropped['crop_dz'] = target_voxel_size_nm
    cropped['crop_orig_voxel_size_nm'] = voxel_size_nm
    cropped['crop_target_thickness_nm'] = film_thickness_nm
    cropped['crop_actual_thickness_nm'] = n_crop * target_voxel_size_nm
    cropped['crop_center1'] = center

    if verbose:
        print(f"[{sample.get('name', 'sample')}] z crop (at {target_voxel_size_nm} nm/vox): "
              f"[{z_lo}:{z_hi}) -> actual thickness = {n_crop * target_voxel_size_nm:.2f} nm "
              f"(target {film_thickness_nm} nm)")
        total_full = nz_i * ny_i * nx_i
        total_crop = n_crop * ny_i * nx_i
        print(f"    total voxels: {total_full} (interpolated, full) -> {total_crop} (cropped), "
              f"{100 * total_crop / total_full:.1f}% retained")

    return cropped



def plot_crop_verification(sample, cropped, field='Mz', y_index=None,
                            cmap='viridis', vmax=None):
    """
    Plot the center-y (x-z) slice of `field` before and after the z-crop,
    with color-coded reference lines marking the crop boundaries.
    """
    nz_full, ny, nx = sample['shape']
    if y_index is None:
        y_index = ny // 2

    full_slice = sample[field][:, y_index, :]
    crop_slice = cropped[field][:, y_index, :]

    z_lo, z_hi = cropped['crop_z_index_range']
    n_crop = cropped['crop_n_voxels_z']

    if vmax is None:
        vmax = np.nanmax(np.abs(full_slice))
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    boundary_colors = plt.cm.viridis(np.linspace(0, 1, 2))

    # --- before: full stack with crop boundaries marked ---
    im0 = axes[0].imshow(full_slice, origin='lower', aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax, extent=[0, nx, 0, nz_full])
    fig.colorbar(im0, ax=axes[0], label=f'${field}$')
    axes[0].set_title(f"Before crop\n{sample.get('name','')}  (y = {y_index})")
    axes[0].set_xlabel('x (voxel)')
    axes[0].set_ylabel('z (voxel)')

    for c, z_line, label in zip(boundary_colors, [z_lo, z_hi], ['crop lo', 'crop hi']):
        axes[0].axhline(z_line, color=c, lw=2, linestyle='--', zorder=3, label=label)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].annotate(
        f'N$_z$ = {n_crop} voxels\n(~{cropped["crop_actual_thickness_nm"]:.1f} nm)',
        xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10,
        ha='left', va='bottom',
        bbox=dict(boxstyle='round', fc='white', alpha=0.85))

    # --- after: cropped stack, same color lines at its own edges ---
    im1 = axes[1].imshow(crop_slice, origin='lower', aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax, extent=[0, nx, 0, n_crop])
    fig.colorbar(im1, ax=axes[1], label=f'${field}$')
    axes[1].set_title(f"After crop\n{sample.get('name','')}  (y = {y_index})")
    axes[1].set_xlabel('x (voxel)')
    axes[1].set_ylabel('z (voxel, cropped)')
    for c, z_line in zip(boundary_colors, [0, n_crop]):
        axes[1].axhline(z_line, color=c, lw=2, linestyle='--', zorder=3)

    plt.tight_layout()
    plt.show()

    total_full = nz_full * ny * nx
    total_crop = n_crop * ny * nx
    print(f"Total voxels: {total_full} (full volume) -> {total_crop} (cropped), "
          f"{n_crop} of {nz_full} z-slices kept "
          f"({100*n_crop/nz_full:.1f}%).")



import numpy as np
import matplotlib.pyplot as plt


def plot_crop_interpolation_verification(sample, cropped, field='Mz', y_index=None,
                            cmap='viridis', vmax=None):
    """
    Plot the (x-z) slice of `field` before and after crop+interpolation.

    `sample` is assumed to be at its original voxel size (e.g. 8nm) and
    `cropped` may be on a finer, interpolated grid (e.g. 2nm) -- as produced
    by crop_volume_by_thickness(). Because the two arrays can have different
    shapes/voxel sizes, everything is plotted in physical units (nm) rather
    than raw voxel index, so the crop boundaries drawn on both panels line
    up correctly regardless of resolution.
    """
    nz_full, ny_full, nx_full = sample['shape']
    n_crop, ny_crop, nx_crop = cropped['shape']

    # physical coordinate arrays (fall back to plain voxel index if absent)
    z_full = np.asarray(sample.get('z_coords', np.arange(nz_full)), dtype=float)
    x_full = np.asarray(sample.get('x_coords', np.arange(nx_full)), dtype=float)
    y_full = np.asarray(sample.get('y_coords', np.arange(ny_full)), dtype=float)

    z_crop = np.asarray(cropped.get('z_coords', np.arange(n_crop)), dtype=float)
    x_crop = np.asarray(cropped.get('x_coords', np.arange(nx_crop)), dtype=float)
    y_crop = np.asarray(cropped.get('y_coords', np.arange(ny_crop)), dtype=float)

    # voxel sizes -- prefer the metadata written by crop_volume_by_thickness,
    # fall back to inferring from the coordinate arrays themselves
    orig_dz = cropped.get('crop_orig_voxel_size_nm')
    if orig_dz is None:
        orig_dz = float(np.median(np.diff(z_full))) if len(z_full) > 1 else 1.0
    new_dz = cropped.get('crop_dz')
    if new_dz is None:
        new_dz = float(np.median(np.diff(z_crop))) if len(z_crop) > 1 else orig_dz

    def nearest_index(arr, value):
        return int(np.argmin(np.abs(arr - value)))

    # pick the same PHYSICAL y-location in both volumes (index may differ
    # between them if y was also interpolated)
    if y_index is None:
        y_index = ny_full // 2
    y_phys = y_full[y_index]
    y_index_crop = nearest_index(y_crop, y_phys)

    full_slice = sample[field][:, y_index, :]
    crop_slice = cropped[field][:, y_index_crop, :]

    # physical z-bounds of the kept crop region (voxel edges, not centers)
    if len(z_crop) > 0:
        z_lo_nm = z_crop[0] - new_dz / 2.0
        z_hi_nm = z_crop[-1] + new_dz / 2.0
    else:
        z_lo_nm = z_hi_nm = cropped.get('crop_center1', 0.0)

    if vmax is None:
        vmax = np.nanmax(np.abs(full_slice))
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    boundary_colors = plt.cm.viridis(np.linspace(0, 1, 2))

    # --- before: full stack at its native (coarser) resolution ---
    im0 = axes[0].imshow(full_slice, origin='lower', aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          extent=[x_full[0], x_full[-1], z_full[0], z_full[-1]])
    fig.colorbar(im0, ax=axes[0], label=f'${field}$')
    axes[0].set_title(f"Before crop ({orig_dz:g} nm/vox)\n"
                       f"{sample.get('name', '')}  (y $\\approx$ {y_phys:.1f} nm)")
    axes[0].set_xlabel('x (nm)')
    axes[0].set_ylabel('z (nm)')

    for c, z_line, label in zip(boundary_colors, [z_lo_nm, z_hi_nm], ['crop lo', 'crop hi']):
        axes[0].axhline(z_line, color=c, lw=2, linestyle='--', zorder=3, label=label)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].annotate(
        f'interpolated {orig_dz:g} nm $\\to$ {new_dz:g} nm/vox\n'
        f'N$_z$ (crop) = {n_crop} vox\n'
        f'(~{cropped["crop_actual_thickness_nm"]:.1f} nm)',
        xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10,
        ha='left', va='bottom',
        bbox=dict(boxstyle='round', fc='white', alpha=0.85))

    # --- after: cropped + interpolated stack, same physical boundary lines ---
    im1 = axes[1].imshow(crop_slice, origin='lower', aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax,
                          extent=[x_crop[0], x_crop[-1], z_crop[0], z_crop[-1]])
    fig.colorbar(im1, ax=axes[1], label=f'${field}$')
    axes[1].set_title(f"After crop + interp ({new_dz:g} nm/vox)\n"
                       f"{sample.get('name', '')}  (y $\\approx$ {y_crop[y_index_crop]:.1f} nm)")
    axes[1].set_xlabel('x (nm)')
    axes[1].set_ylabel('z (nm, cropped)')
    for c, z_line in zip(boundary_colors, [z_lo_nm, z_hi_nm]):
        axes[1].axhline(z_line, color=c, lw=2, linestyle='--', zorder=3)

    plt.tight_layout()
    plt.show()

    total_full = nz_full * ny_full * nx_full
    total_crop = n_crop * ny_crop * nx_crop
    print(f"Voxel size: {orig_dz:g} nm -> interpolated to {new_dz:g} nm")
    print(f"Shape: {sample['shape']} (full, {orig_dz:g} nm/vox) -> "
          f"{cropped['shape']} (cropped, {new_dz:g} nm/vox)")
    print(f"Total voxels: {total_full} (full) -> {total_crop} (cropped)")
    print(f"z-thickness kept: {cropped['crop_actual_thickness_nm']:.2f} nm "
          f"(target {cropped['crop_target_thickness_nm']:.2f} nm)")
    

def plot_three_slice_fancy(sample, field='Mz', x_index=None, y_index=None, z_index=None,
                      cmap='viridis', vmin=None, vmax=None):
    """
    Plot an x-slice (y-z plane), y-slice (x-z plane), and z-slice (x-y plane)
    of `field` from a sample dict, side by side in one figure.

    Parameters
    ----------
    sample : dict
        A sample dict with (nz, ny, nx)-shaped fields, e.g. samples[name]
        or samples[name]['cropped'].
    field : str
        Which field to view on all three slices: 'Mx', 'My', 'Mz',
        or 'magnitude' (aliases 'mag'/'Mag' accepted, case-insensitive).
    x_index, y_index, z_index : int or None
        Voxel index for each slice. Defaults to the center of that axis.
    cmap : str
        Colormap for all three panels (default 'viridis').
    vmin, vmax : float or None
        Shared color limits across all three panels. If None, computed
        from the min/max of the full `field` array (not just the slices
        shown), so panels stay on a consistent, comparable scale.

    Returns
    -------
    (fig, axes) : the created matplotlib figure and its 3 axes.
    """
    field_aliases = {'mag': 'magnitude', 'magnitude': 'magnitude',
                      'mx': 'Mx', 'my': 'My', 'mz': 'Mz'}
    key = field_aliases.get(field.lower())
    if key is None or key not in sample:
        raise KeyError(
            f"field '{field}' not recognized/available. "
            f"Expected one of: Mx, My, Mz, magnitude."
        )
    data = sample[key]  # shape (nz, ny, nx)
    nz, ny, nx = sample['shape']

    if x_index is None:
        x_index = nx // 2
    if y_index is None:
        y_index = ny // 2
    if z_index is None:
        z_index = nz // 2

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    slice_x = data[:, :, x_index]   # (nz, ny) -- y-z plane at fixed x
    slice_y = data[:, y_index, :]   # (nz, nx) -- x-z plane at fixed y
    slice_z = data[z_index, :, :]   # (ny, nx) -- x-y plane at fixed z

    LABEL_FONT = {'family': 'Times New Roman', 'weight': 'bold', 'size': 16}
    TITLE_FONT = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}
    TICK_FONT = {'family': 'Times New Roman', 'size': 13}
    CBAR_LABEL_FONT = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}

    def style_axis(ax):
        ax.set_xlabel(ax.get_xlabel(), fontdict=LABEL_FONT)
        ax.set_ylabel(ax.get_ylabel(), fontdict=LABEL_FONT)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(TICK_FONT['size'])

    def style_colorbar(cbar, label_text):
        cbar.set_label(label_text, fontdict=CBAR_LABEL_FONT)
        cbar.formatter = mticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(TICK_FONT['size'])
        cbar.ax.yaxis.get_offset_text().set_fontname('Times New Roman')
        cbar.ax.yaxis.get_offset_text().set_fontsize(TICK_FONT['size'])

    # Conversion factor: 1 pix = 8 nm
    PIX_TO_NM = 2 #8

    # Calculate extents in nm
    extent_x = [0, ny * PIX_TO_NM / 1000, 0, nz * PIX_TO_NM]  # y, z
    extent_y = [0, nx * PIX_TO_NM / 1000, 0, nz * PIX_TO_NM]  # x, z
    extent_z = [0, nx * PIX_TO_NM / 1000, 0, ny * PIX_TO_NM / 1000]  # x, y

    # Create figure with GridSpec for custom layout
    fig = plt.figure(figsize=(25,8)) ##################################################################################
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 2, 0.15], height_ratios=[1, 1], 
                        hspace=0.37, wspace=0.15)

    # Z slice (left, full square)
    ax_z = fig.add_subplot(gs[:, 0])
    im_z = ax_z.imshow(slice_z, origin='lower', aspect='equal', cmap=cmap,
                        vmin=vmin, vmax=vmax, extent=extent_z)
    ax_z.set_title(f"Z slice", fontdict=TITLE_FONT)
    ax_z.set_xlabel('x (μm)', fontdict=LABEL_FONT)
    ax_z.set_ylabel('y (μm)', fontdict=LABEL_FONT)
    #ax_z.axhline(y_index * PIX_TO_NM/1000, color='magenta', lw=5, linestyle='--', zorder=3)
    #ax_z.axvline(x_index * PIX_TO_NM/1000, color='darkorange', lw=5, linestyle='--', zorder=3)
    style_axis(ax_z)

    # X slice (top right, thin and long)
    ax_x = fig.add_subplot(gs[0, 1])
    im_x = ax_x.imshow(slice_x, origin='lower', aspect='auto', cmap=cmap,
                        vmin=vmin, vmax=vmax, extent=extent_x)
    ax_x.set_title(f"X slice", fontdict=TITLE_FONT, color='darkorange')
    ax_x.set_xlabel('y (μm)', fontdict=LABEL_FONT)
    ax_x.set_ylabel('z (nm)', fontdict=LABEL_FONT)
    style_axis(ax_x)

    # Y slice (bottom right, thin and long)
    ax_y = fig.add_subplot(gs[1, 1])
    im_y = ax_y.imshow(slice_y, origin='lower', aspect='auto', cmap=cmap,
                        vmin=vmin, vmax=vmax, extent=extent_y)
    ax_y.set_title(f"Y slice", fontdict=TITLE_FONT, color='magenta')
    ax_y.set_xlabel('x (μm)', fontdict=LABEL_FONT)
    ax_y.set_ylabel('z (nm)', fontdict=LABEL_FONT)
    style_axis(ax_y)


    # Colorbar for Z slice
    cbar_xy = fig.colorbar(im_x, ax=[ax_x, ax_y], orientation='vertical', 
                            pad=0.05, aspect=30)
    style_colorbar(cbar_xy, key)

    fig.suptitle(f"{key} Contrast at Volume Center", 
                fontfamily=LABEL_FONT['family'],
                fontweight=LABEL_FONT['weight'],
                fontsize=25)

    plt.show()

    return fig

#######################################################################
# Gaussian Smoothing (only in dedicated M value)

#######################################################################
def apply_gaussian_blur_and_view(sample, field='Mz', x_index=None, y_index=None, z_index=None,
                      cmap='viridis', vmin=None, vmax=None, sigma=None):
    """
    Plot an x-slice (y-z plane), y-slice (x-z plane), and z-slice (x-y plane)
    of `field` from a sample dict, side by side in one figure.

    Parameters
    ----------
    sample : dict
        A sample dict with (nz, ny, nx)-shaped fields, e.g. samples[name]
        or samples[name]['cropped'].
    field : str
        Which field to view on all three slices: 'Mx', 'My', 'Mz',
        or 'magnitude' (aliases 'mag'/'Mag' accepted, case-insensitive).
    x_index, y_index, z_index : int or None
        Voxel index for each slice. Defaults to the center of that axis.
    cmap : str
        Colormap for all three panels (default 'viridis').
    vmin, vmax : float or None
        Shared color limits across all three panels. If None, computed
        from the min/max of the full `field` array (not just the slices
        shown), so panels stay on a consistent, comparable scale.
    sigma : float, tuple of 3 floats, or None
        If not None, apply 3D Gaussian smoothing to the field before slicing
        (via scipy.ndimage.gaussian_filter), applied across all three axes
        (z, y, x). Pass a single float for isotropic smoothing in voxel
        units, or a (sigma_z, sigma_y, sigma_x) tuple for anisotropic
        smoothing, e.g. to account for non-cubic voxel spacing. The
        resulting smoothed volume is cached in
        sample['smoothed'][key][sigma] so repeated calls with the same
        sigma don't recompute it, and so it can be reused elsewhere in
        the notebook (e.g. sample['smoothed']['Mz'][1.0]).

    Returns
    -------
    (fig, axes) : the created matplotlib figure and its 3 axes.
    """
    field_aliases = {'mag': 'magnitude', 'magnitude': 'magnitude',
                      'mx': 'Mx', 'my': 'My', 'mz': 'Mz'}
    key = field_aliases.get(field.lower())
    if key is None or key not in sample:
        raise KeyError(
            f"field '{field}' not recognized/available. "
            f"Expected one of: Mx, My, Mz, magnitude."
        )

    data = sample[key]  # shape (nz, ny, nx), raw field

    if sigma is not None:
        sample.setdefault('smoothed', {}).setdefault(key, {})
        if sigma not in sample['smoothed'][key]:
            sample['smoothed'][key][sigma] = gaussian_filter(data, sigma=sigma)
        data = sample['smoothed'][key][sigma]

    nz, ny, nx = sample['shape']

    if x_index is None:
        x_index = nx // 2
    if y_index is None:
        y_index = ny // 2
    if z_index is None:
        z_index = nz // 2

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    slice_x = data[:, :, x_index]   # (nz, ny) -- y-z plane at fixed x
    slice_y = data[:, y_index, :]   # (nz, nx) -- x-z plane at fixed y
    slice_z = data[z_index, :, :]   # (ny, nx) -- x-y plane at fixed z

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    im0 = axes[0].imshow(slice_x, origin='lower', aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax, extent=[0, ny, 0, nz])
    axes[0].set_title(f"X slice (x = {x_index})", color='darkorange')
    axes[0].set_xlabel('y (voxel)')
    axes[0].set_ylabel('z (voxel)')
    fig.colorbar(im0, ax=axes[0], label=key)

    im1 = axes[1].imshow(slice_y, origin='lower', aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax, extent=[0, nx, 0, nz])
    axes[1].set_title(f"Y slice (y = {y_index})", color='magenta')
    axes[1].set_xlabel('x (voxel)')
    axes[1].set_ylabel('z (voxel)')
    fig.colorbar(im1, ax=axes[1], label=key)

    im2 = axes[2].imshow(slice_z, origin='lower', aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax, extent=[0, nx, 0, ny])
    axes[2].set_title(f"Z slice (z = {z_index})")
    axes[2].set_xlabel('x (voxel)')
    axes[2].set_ylabel('y (voxel)')
    axes[2].axhline(y_index, color='magenta', lw=2, linestyle='--', zorder=3)
    axes[2].axvline(x_index, color='darkorange', lw=2, linestyle='--', zorder=3)
    fig.colorbar(im2, ax=axes[2], label=key)

    title_suffix = f" (sigma={sigma})" if sigma is not None else " (raw)"
    fig.suptitle(f"{sample.get('name','')} -- {key}{title_suffix}")
    plt.tight_layout()
    plt.show()

    return fig, axes




#######################################################################
# Material Comparison Line Plots

#######################################################################
def plot_angle_vs_z(wall_df, bounds, angle_col='psi_chiral',  bloch_axis = None , neel_axis = None , title = None):
    df = wall_df.copy()
    z_lo, z_hi = bounds
    z_min = z_lo - z_lo
    z_max = z_hi - z_lo
   

    df['z_layer'] = df['iz'].round().astype(int)
    # print(df['iz'].min(), df['iz'].max())
    # print(df['z_layer'].value_counts().sort_index() if 'z_layer' in df else 'not yet computed')

    # Restrict to the physical film thickness, excluding any background/
    # padding layers from the tomography reconstruction.
    n_before = len(df)
    df = df[(df['z_layer'] >= z_min) & (df['z_layer'] <= z_max)]
    print(f"{len(df)} / {n_before} vertices fall within layers {z_min}-{z_max}")


    grouped = df.groupby('z_layer')[angle_col].agg(['mean', 'std', 'count'])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                fmt='o-', capsize=3)

    if bloch_axis== None:
        print('no axis')
    else:
        print(bloch_axis)
        for deg in bloch_axis: 
            ax.axhline(deg, color = 'magenta', label = 'Bloch Wall')
        for deg in neel_axis:
            ax.axhline(deg, color = 'cyan', label = 'Neel Wall')
    ax.set_xlabel('z voxel layer')
    ax.set_xticks(range(z_min, z_max))
    ax.set_ylabel(f'{angle_col} (deg)')
    ax.set_title(f'{angle_col} vs. thickness position \n {title}')
    # plt.legend()
    plt.tight_layout()
    plt.show()
    return grouped