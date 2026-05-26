"""
Functions.py
============
Shared utility functions for ptychography data processing and polarization alignment.

This module is imported in analysis notebooks as:
    import Functions as register

All functions are then called as register.function_name(...)

Author: Emma Bernard
"""

# ========================= IMPORTS =========================
import os
import datetime

import numpy as np
import tifffile
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

from scipy.fft import fftn, ifftn
from scipy.ndimage import fourier_shift, gaussian_filter
from scipy.ndimage import shift as img_shift
from numpy.fft import fftshift
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

from skimage.feature import graycomatrix, graycoprops
from skimage import io

from sklearn.cluster import KMeans
import scipy.ndimage as ndimage

# pystxm imports (required for scan metadata functions)
from pystxm_core.io.writeNX import stxm
from pystxm_core.stack import stack
from pystxm_core.image import image
from pystxm_core.utilities import h5_to_im

from skimage.transform import EuclideanTransform, warp
from skimage import measure, transform
from scipy import ndimage

# ===========================================================


# -----------------------------------------------------------------------
# SECTION 1: File Sorting and Metadata
# -----------------------------------------------------------------------

def sort_ptycho_images(folder_path):
    """
    Walk through a list of date-labeled folders and collect all valid
    ptychography HDF5 files.

    A file is considered valid if:
      - Its name ends with '_ccdframes_0_0.h5'

    Parameters
    ----------
    folder_path : list of str
        List of directory paths (one per acquisition date) to search.

    Returns
    -------
    filtered_scans : list of str
        Full paths to all files passing the filters.
    """
    filtered_scans = []

    for date in folder_path:
        for filename in os.listdir(date):
            # Must match the expected CCD frames filename pattern
            if not filename.endswith('_ccdframes_0_0.h5'):
                continue

            full_path = os.path.join(date, filename)
            size_kb = os.path.getsize(full_path) / 1024

            # Must be larger than 1000 KB to be a complete scan
            if size_kb <= 1000:
                continue

            filtered_scans.append(full_path)

        print(f"Found {len(filtered_scans)} qualifying scans:")

    return filtered_scans


def print_ptycho_metadata(recon_files):
    """
    Print scan metadata (scan number, angle, polarization) for each
    reconstructed ptychography file.

    Parameters
    ----------
    recon_files : list of str
        Full paths to '_ccdframes_0_0.h5' reconstruction files.
        A matching '.stxm' file must exist at the same path.
    """
    for i, filepath in enumerate(recon_files):
        stxm_filename = filepath.replace('_ccdframes_0_0.h5', '.stxm')
        s = stxm(stxm_file=stxm_filename)

        scan_angle = s.data['entry0']["motors"]["coarser"]
        polar = s.data['entry0']['motors']['polarization']
        scan_num = filepath.split('_')[1]
        energy = s.data['entry0']['motors']['beamline_energy']

        print(f"{i}. Scan {scan_num}; Angle = {scan_angle:.2f}; Polarization = {polar}")


# -----------------------------------------------------------------------
# SECTION 2: Polarization Alignment -- Image Registration
# -----------------------------------------------------------------------

def align_images_CC(img1, img2, bound):
    """
    Align two images using phase cross-correlation with subpixel refinement.

    The cross-correlation peak is found within a restricted window of size
    (2*bound x 2*bound) around the image center, limiting the search to
    physically reasonable shifts. Subpixel accuracy is achieved by fitting
    a 2D paraboloid to the peak neighborhood.

    Parameters
    ----------
    img1 : 2D np.ndarray
        Reference image (e.g., positive polarization).
    img2 : 2D np.ndarray
        Image to be aligned (e.g., negative polarization).
    bound : int
        Half-width (in pixels) of the search window around the image center.

    Returns
    -------
    translation : tuple of float
        Estimated (dy, dx) translation in pixels to shift img2 onto img1.
    cross_corr_window : 2D np.ndarray
        The cross-correlation surface within the search window (for inspection).
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Normalize both images to [0, 1] before cross-correlation
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    # Compute FFTs and normalized cross-power spectrum (phase correlation)
    f1 = fftn(img1)
    f2 = fftn(img2)
    eps = 1e-9
    cross_power_spectrum = f1 * np.conj(f2)
    cross_power_spectrum /= np.abs(cross_power_spectrum) + eps

    # Inverse FFT -> correlation surface; shift zero-frequency to center
    cross_corr = ifftn(cross_power_spectrum).real
    cross_corr = fftshift(cross_corr)

    # Restrict search to a window of size (2*bound x 2*bound) at image center
    height, width = cross_corr.shape
    cen_y, cen_x = height // 2, width // 2
    cross_corr_window = cross_corr[cen_y - bound:cen_y + bound,
                                   cen_x - bound:cen_x + bound]

    # Locate the peak within the window
    max_idx = np.unravel_index(np.argmax(np.abs(cross_corr_window)),
                               cross_corr_window.shape)
    peak_val = cross_corr_window[max_idx]

    # Subpixel refinement: fit a 2D paraboloid around the peak
    if peak_val > 0:
        dy_sub, dx_sub = _fit_paraboloid_2d(cross_corr_window, max_idx, size=3)
    else:
        dy_sub, dx_sub = _fit_paraboloid_2d(-cross_corr_window, max_idx, size=3)

    # Convert window-local peak to shift relative to image center
    y_peak = max_idx[0] + dy_sub - cross_corr_window.shape[0] // 2
    x_peak = max_idx[1] + dx_sub - cross_corr_window.shape[1] // 2

    translation = (y_peak, x_peak)
    return translation, cross_corr_window



def rotate_images_CC(mask_pos, mask_neg, image_to_rotate, angle='', sample='', Gamma='', plot = True): 
    '''
    Always shifting the CPL wrt CPR so image to rotate should be CPL 
    '''
    # cpr, cpl: 2D binary masks (numpy arrays)
    # Assume bright dots are 1 and background 0
    cpr_bin = mask_pos > 0
    cpl_bin = mask_neg > 0
    
    lab_cpr = measure.label(cpr_bin)
    lab_cpl = measure.label(cpl_bin)
    
    props_cpr = measure.regionprops(lab_cpr)
    props_cpl = measure.regionprops(lab_cpl)
    
    pts_cpr = np.array([p.centroid for p in props_cpr])[:, ::-1]  # (x, y)
    pts_cpl = np.array([p.centroid for p in props_cpl])[:, ::-1]  # (x, y)
    
    # Keep correspondence consistent
    pts_cpr = pts_cpr[np.lexsort((pts_cpr[:,1], pts_cpr[:,0]))]
    pts_cpl = pts_cpl[np.lexsort((pts_cpl[:,1], pts_cpl[:,0]))]
    
    # New API (no deprecation warning)
    tform = transform.EuclideanTransform.from_estimate(pts_cpl, pts_cpr)  # CPL -> CPR
    
    rotation_deg = np.degrees(tform.rotation)
    theta = np.deg2rad(rotation_deg)       # your computed rotation in degrees
    trans =  tform.translation
    
    print("Rotation (deg):", rotation_deg)
    print("Translation (px):", trans)
    
    tform2 = EuclideanTransform(rotation=theta, translation=(trans[0], trans[1])) 

    ## translate the mask
    cpl_mask_aligned = warp(
        cpl_bin.astype(float),
        inverse_map=tform2.inverse,
        order=0,
        preserve_range=True,
        mode='constant',
        cval=0
    ).astype(mask_neg.dtype)


    ## translate the image

    aligned_image = warp(
        image_to_rotate,
        inverse_map=tform2.inverse,
        order=0,
        preserve_range=True,
        mode='constant',
        cval=0
    ).astype(image_to_rotate.dtype)

    if plot == True: 
        # Plot fiducial masks before and after CC
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
        
        diff_trans = mask_neg.astype(int) - mask_pos.astype(int)
        axes[0].imshow(diff_trans, norm=mpl.colors.Normalize(vmin=-0.01, vmax=0.01))
        axes[0].set_title(f"Masked shifted CPL - CPR, \n Sample {sample}, {Gamma} @ {angle:.2f}")
        
        axes[1].imshow(cpl_mask_aligned, norm=mpl.colors.Normalize(vmin=-0.01, vmax=0.01))
        axes[1].set_title(f"rotated and shifted CPL, \n Sample {sample}, {Gamma} @ {angle:.2f}")
        
        diff_trans_rot = cpl_mask_aligned.astype(int) - mask_pos.astype(int)
        axes[2].imshow(diff_trans_rot, norm=mpl.colors.Normalize(vmin=-0.01, vmax=0.01))
        axes[2].set_title(f"Masked shifted/rotated CPL - CPR, \n Sample {sample}, {Gamma} @ {angle:.2f}")
        
        plt.tight_layout(); plt.show()

    return aligned_image, cpl_mask_aligned


    
def _fit_paraboloid_2d(corr, peak, size=5):
    """
    Fit a 2D paraboloid to the neighborhood around a correlation peak for
    subpixel localization.

    Parameters
    ----------
    corr : 2D np.ndarray
        Correlation surface (or its negative).
    peak : tuple of int
        (row, col) index of the peak in `corr`.
    size : int
        Side length of the fitting patch (should be odd).

    Returns
    -------
    y_peak, x_peak : float
        Subpixel offsets of the paraboloid vertex from `peak`.
    """
    y0, x0 = peak
    half = size // 2

    # Pad to avoid edge effects
    padded = np.pad(corr, half, mode='wrap')
    patch = padded[y0:y0 + size, x0:x0 + size]

    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    x = x.flatten()
    y = y.flatten()
    z = patch.flatten()

    # Fit: f(x,y) = ax^2 + by^2 + cxy + dx + ey + f
    A = np.column_stack((x**2, y**2, x * y, x, y, np.ones_like(x)))
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c, d, e, _ = coeffs

    denom = 4 * a * b - c**2
    if denom == 0:
        return 0.0, 0.0  # Degenerate case -- return no subpixel correction

    x_peak = (c * e - 2 * b * d) / denom
    y_peak = (c * d - 2 * a * e) / denom
    return y_peak, x_peak



def xmcd_peak_separation(intensities, bins=200, smooth_sigma=2.0):
    """
    Compute contrast as peak separation for a bimodal XMCD intensity distribution.
    
    Parameters
    ----------
    intensities : array-like
        Flat array of pixel intensity values.
    bins : int
        Number of histogram bins.
    smooth_sigma : float
        Gaussian smoothing applied to histogram before peak finding.

    Returns
    -------
    dict with keys:
        peak_positions   : (float, float) — intensity values of the two peaks
        peak_separation  : float          — |μ₁ − μ₂|
        d_prime          : float          — separation / (σ₁ + σ₂), SNR-normalized
    """
    intensities = np.asarray(intensities).flatten()

    # Build and smooth histogram
    counts, edges = np.histogram(intensities, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    smoothed = gaussian_filter1d(counts.astype(float), sigma=smooth_sigma)

    # Find peaks — require some minimum prominence to avoid noise
    prominence_threshold = smoothed.max() * 0.1
    peaks, props = find_peaks(smoothed, prominence=prominence_threshold)

    if len(peaks) < 2:
        raise ValueError(f"Expected 2 peaks, found {len(peaks)}. "
                         "Try adjusting bins or smooth_sigma.")

    # Take the two most prominent peaks
    prominences = props["prominences"]
    top2_idx = np.argsort(prominences)[-2:]
    top2_peaks = sorted(peaks[top2_idx])  # left peak first

    mu1, mu2 = centers[top2_peaks[0]], centers[top2_peaks[1]]
    peak_separation = abs(mu2 - mu1)

    # Estimate per-peak std by fitting a window around each peak
    def peak_std(peak_idx, window=20):
        lo = max(0, peak_idx - window)
        hi = min(len(centers), peak_idx + window)
        weights = smoothed[lo:hi]
        c = centers[lo:hi]
        mu = np.average(c, weights=weights)
        return np.sqrt(np.average((c - mu) ** 2, weights=weights))

    sigma1 = peak_std(top2_peaks[0])
    sigma2 = peak_std(top2_peaks[1])
    d_prime = peak_separation / (sigma1 + sigma2) if (sigma1 + sigma2) > 0 else np.inf

    # return {
    #     "peak_positions":  (mu1, mu2),
    #     "peak_separation": peak_separation,
    #     "d_prime":         d_prime,
    # }

    return peak_separation


def snr_db(arr: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio in dB, treating mean as signal and std as noise.
    SNR(dB) = 20 * log10(mean / std)
    A higher SNR means low contrast (signal dominates, little variation).
    A lower SNR means more texture/contrast relative to the mean level.
    """
    std = np.std(arr)
    mean = np.mean(arr)
    if std == 0 or mean == 0:
        return float("inf") if std == 0 else float("-inf")
    return float(20 * np.log10(mean / std))
    
# -----------------------------------------------------------------------
# SECTION 3: Normalization
# -----------------------------------------------------------------------

def auto_normalize_by_fiducial(image, c, bins=100,
                                peak_prominence=0.000001,
                                window_width=0.1,
                                plot_hist=False):
    """
    Normalize an image by the mean intensity of its high-intensity fiducial
    marker (e.g., a gold bead or hole).

    The fiducial is identified as the rightmost (highest-intensity) peak in
    the smoothed pixel intensity histogram.

    Parameters
    ----------
    image : 2D np.ndarray
        Input amplitude or intensity image.
    c : str
        Matplotlib color string used for histogram range lines.
    bins : int
        Number of histogram bins.
    peak_prominence : float
        Minimum prominence for histogram peak detection.
    window_width : float
        Half-width of the intensity window around the fiducial peak
        (in the same units as image intensity).
    plot_hist : bool
        If True, display a diagnostic histogram plot. Set to False to
        suppress output and reduce memory usage during batch processing.

    Returns
    -------
    normalized_image : 2D np.ndarray
        Image divided by the fiducial mean intensity.
    fiducial_mean : float
        Mean intensity of pixels identified as the fiducial.
    """
    flat = image.flatten()
    hist_vals, bin_edges = np.histogram(flat, bins=bins, density=True)
    smoothed = gaussian_filter1d(hist_vals, sigma=2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(smoothed, prominence=peak_prominence)

    if len(peaks) < 2:
        raise ValueError("Could not find at least two peaks in histogram. "
                         "Try lowering peak_prominence.")

    # Fiducial = rightmost (highest-intensity) peak
    fid_peak_idx = peaks[np.argmax(bin_centers[peaks])]
    fid_peak_center = bin_centers[fid_peak_idx]

    range_min = fid_peak_center - window_width
    range_max = fid_peak_center + window_width

    mask = (flat >= range_min) & (flat <= range_max)
    mask_img = (image >= range_min) & (image <= range_max)
    n_pixels = int(np.sum(mask))
    fiducial_vals = flat[mask]

    if fiducial_vals.size == 0:
        raise ValueError("No pixels found in the selected fiducial intensity range.")

    fiducial_mean = np.mean(fiducial_vals)
    normalized_image = image / fiducial_mean

    if plot_hist:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # --- Left panel: full histogram on log y-axis -----------------------
        # Log scale compresses the dominant main peak so the small fiducial
        # peak is no longer dwarfed.  Clip zeros to 1e-6 to avoid log(0).
        ax1.plot(bin_centers, np.clip(hist_vals, 1e-6, None),
                 alpha=0.3, label='Raw histogram')
        ax1.plot(bin_centers, np.clip(smoothed,  1e-6, None),
                 color='black', label='Smoothed')
        ax1.axvline(range_min,        color=c,     linestyle='--', linewidth=1,
                    label=f'Range min ({range_min:.3f})')
        ax1.axvline(range_max,        color=c,     linestyle='--', linewidth=1,
                    label=f'Range max ({range_max:.3f})')
        ax1.axvline(fid_peak_center,  color='red', linestyle='--', linewidth=1.5,
                    label=f'Fiducial peak ({fid_peak_center:.3f})')
        ax1.set_yscale('log')
        ax1.set_title("Full histogram (log scale)")
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Density (log)")
        ax1.legend(fontsize=8)

        # --- Right panel: zoomed to the fiducial peak region ----------------
        # Show only the window around the detected peak so its shape is clear.
        zoom_pad = window_width * 4      # show 4x the normalization window
        zoom_lo  = max(bin_centers[0],  fid_peak_center - zoom_pad)
        zoom_hi  = min(bin_centers[-1], fid_peak_center + zoom_pad)
        zoom_mask = (bin_centers >= zoom_lo) & (bin_centers <= zoom_hi)

        ax2.plot(bin_centers[zoom_mask], hist_vals[zoom_mask],
                 alpha=0.35, label='Raw histogram')
        ax2.plot(bin_centers[zoom_mask], smoothed[zoom_mask],
                 color='black', linewidth=1.5, label='Smoothed')
        # ax2.axvspan(range_min, range_max, color=c, alpha=0.12,
        #             label='Normalization window')
        ax2.axvline(range_min,       color=c,     linestyle='--', linewidth=1)
        ax2.axvline(range_max,       color=c,     linestyle='--', linewidth=1)
        ax2.axvline(fid_peak_center, color='red', linestyle='--', linewidth=1.5)
        # ax2.annotate(f'peak = {fid_peak_center:.3f}',
        #              xy=(fid_peak_center, smoothed[fid_peak_idx]),
        #              xytext=(fid_peak_center + zoom_pad * 0.3,
        #                      smoothed[fid_peak_idx] * 1.15),
        #              arrowprops=dict(arrowstyle='->', color='red'),
        #              color='red', fontsize=9)
        ax2.set_title("Fiducial peak (zoomed)")
        ax2.set_xlabel("Intensity")
        ax2.set_ylabel("Density")
        ax2.legend(fontsize=8)

        plt.suptitle("Fiducial Detection Diagnostic", fontsize=11)
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        norm_CC = mpl.colors.Normalize(vmin=0, vmax=1)
        show = axes.imshow(mask_img, norm = norm_CC)
        axes.set_title(f'max peak normalization, masked area')
        fig.colorbar(show)
        plt.tight_layout()
        plt.show()

    return normalized_image, fiducial_mean, n_pixels

def auto_normalize_by_fiducial_kmeans(image, fiduc_num, factor,
                                      plot=False, sample='', Gamma=''):
    """
    Normalize an image by the mean intensity of its fiducial hole pixels,
    using KMeans cluster masks directly (no bounding boxes).

    Pipeline
    --------
    1. Threshold bright pixels at (max_valid / factor)
    2. Run KMeans on their (y, x) coordinates to assign each bright pixel
       to one of fiduc_num clusters
    3. Compute the combined mean of all clustered bright pixels
    4. Divide the image by that mean

    Parameters
    ----------
    image : 2D np.ndarray
        Raw amplitude image. Fiducial holes must appear as bright regions.
    fiduc_num : int
        Number of fiducial holes expected (= number of KMeans clusters).
    factor : float
        Intensity threshold factor. Pixels above (max_valid / factor) are
        treated as fiducial candidates. Typical value: 1.25.
    plot : bool
        If True, display (1) the original image with the KMeans mask
        overlaid, and (2) a bar chart of per-cluster means, so you can
        verify the mask captures the fiducials correctly.
    sample : str
        Sample identifier used in plot titles.
    Gamma : str
        Gamma label used in plot titles.

    Returns
    -------
    normalized_image : 2D np.ndarray
        Image divided by fiducial_mean.
    fiducial_mean : float
        Mean intensity across all valid fiducial cluster pixels combined.
    n_pixels : int
        Total number of pixels included in the fiducial mask.
    """
    from sklearn.cluster import KMeans

    img_h, img_w = image.shape

    # --- Step 1: threshold to find bright candidate pixels ------------------
    finite_vals = image[np.isfinite(image) & (image > 0)]
    if finite_vals.size == 0:
        raise ValueError("Image contains no valid (finite, non-zero) pixels.")

    threshold = finite_vals.max() / factor
    mask_candidates = np.isfinite(image) & (image > threshold)
    candidate_coords = np.argwhere(mask_candidates)  # shape (N, 2): (y, x)

    if len(candidate_coords) < fiduc_num:
        raise ValueError(
            f"Only {len(candidate_coords)} pixels above threshold "
            f"(max / {factor} = {threshold:.4f}); cannot form {fiduc_num} clusters. "
            "Try increasing factor."
        )

    # --- Step 2: KMeans clustering on bright pixel coordinates --------------
    kmeans = KMeans(n_clusters=fiduc_num, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(candidate_coords)

    # Build a full-image label mask (unlabelled pixels = -1)
    label_mask = np.full((img_h, img_w), -1, dtype=int)
    label_mask[candidate_coords[:, 0], candidate_coords[:, 1]] = labels

    # Binary mask of all pixels contributing to fiducial_mean
    mask_img = label_mask >= 0  # True for any pixel assigned to a cluster

    # --- Step 3: compute per-cluster and combined means ---------------------
    cluster_means = []
    all_vals = []

    for k in range(fiduc_num):
        cluster_pixels = image[label_mask == k]
        valid = cluster_pixels[np.isfinite(cluster_pixels) & (cluster_pixels > 0)]
        if valid.size == 0:
            print(f"  Warning: cluster {k} has no valid pixels — skipping.")
            continue
        cluster_means.append(float(np.mean(valid)))
        all_vals.extend(valid.tolist())

    if len(all_vals) == 0:
        raise ValueError("All clusters were empty after filtering. Check your factor value.")

    fiducial_mean = float(np.mean(all_vals))
    n_pixels = len(all_vals)

    # --- Step 4: normalize --------------------------------------------------
    normalized_image = image / fiducial_mean

    # --- Diagnostic plot ----------------------------------------------------
    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: image + cluster mask overlay
        axes[0].imshow(image, cmap='viridis', origin='upper')

        # Build a semi-transparent colored overlay for each cluster
        cmap_clusters = plt.get_cmap('tab10')
        overlay = np.zeros((img_h, img_w, 4))  # RGBA
        for k in range(fiduc_num):
            color = cmap_clusters(k % 10)
            ys, xs = np.where(label_mask == k)
            overlay[ys, xs] = (*color[:3], 0.98)  # semi-transparent

        axes[0].imshow(overlay, origin='upper')

        # Mark centroids
        centroids = kmeans.cluster_centers_  # (y, x)
        for k, (cy, cx) in enumerate(centroids):
            axes[0].plot(cx, cy, 'w+', markersize=10, markeredgewidth=1.5)
            axes[0].text(cx + 4, cy, f"{k}: {cluster_means[k]:.2f}",
                         color='white', fontsize=8, va='center')

        axes[0].set_title(
            f"KMeans mask mean = {fiducial_mean:.4f}\n"
            f"Sample {sample}  {Gamma}"
        )

        # Right: per-cluster bar chart
        axes[1].bar(range(len(cluster_means)), cluster_means,
                    color=[mcolors.to_hex(plt.get_cmap('tab10')(k % 10))
                           for k in range(len(cluster_means))],
                    alpha=0.8, width=0.5)
        axes[1].axhline(fiducial_mean, color='red', linestyle='--',
                        linewidth=1.5,
                        label=f' mean = {fiducial_mean:.4f}')
        axes[1].set_xticks(range(len(cluster_means)))
        axes[1].set_xticklabels([f"Cluster {k}" for k in range(len(cluster_means))])
        axes[1].set_ylabel("Mean intensity")
        axes[1].set_title(f"Per-cluster mean and n_pixels = {n_pixels}")
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        plt.show()

        # -- Figure 2: binary mask -------------------------------------------
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
        norm_CC = mpl.colors.Normalize(vmin=0, vmax=1)
        show = ax2.imshow(mask_img, norm=norm_CC)
        ax2.set_title(
            f"KMeans mask \n {sample}  {Gamma}"
        )
        fig2.colorbar(show)
        plt.tight_layout()
        plt.show()

    return normalized_image, fiducial_mean, n_pixels, mask_img
    
    
# def auto_normalize_by_fiducial(image, c, bins=100,
#                                 peak_prominence=0.000001,
#                                 window_width=0.1,
#                                 plot_hist=False):
#     """
#     Normalize an image by the mean intensity of its high-intensity fiducial
#     marker (e.g., a gold bead or hole).

#     The fiducial is identified as the rightmost (highest-intensity) peak in
#     the smoothed pixel intensity histogram.

#     Parameters
#     ----------
#     image : 2D np.ndarray
#         Input amplitude or intensity image.
#     c : str
#         Matplotlib color string used for histogram range lines.
#     bins : int
#         Number of histogram bins.
#     peak_prominence : float
#         Minimum prominence for histogram peak detection.
#     window_width : float
#         Half-width of the intensity window around the fiducial peak
#         (in the same units as image intensity).
#     plot_hist : bool
#         If True, display a diagnostic histogram plot. Set to False to
#         suppress output and reduce memory usage during batch processing.

#     Returns
#     -------
#     normalized_image : 2D np.ndarray
#         Image divided by the fiducial mean intensity.
#     fiducial_mean : float
#         Mean intensity of pixels identified as the fiducial.
#     """
#     flat = image.flatten()
#     hist_vals, bin_edges = np.histogram(flat, bins=bins, density=True)
#     smoothed = gaussian_filter1d(hist_vals, sigma=2)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#     peaks, _ = find_peaks(smoothed, prominence=peak_prominence)

#     if len(peaks) < 2:
#         plt.figure(figsize=(8, 4))
#         plt.plot(bin_centers, hist_vals, label='Histogram', alpha=0.3)
#         plt.plot(bin_centers, smoothed, label='Smoothed Histogram', color='black')
#         plt.title("Histogram, no Fiducials detected")
#         plt.xlabel("Intensity")
#         plt.ylabel("Density")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
#         raise ValueError("Could not find at least two peaks in histogram. "
#                          "Try lowering peak_prominence.")

#     # Fiducial = rightmost (highest-intensity) peak
#     fid_peak_idx = peaks[np.argmax(bin_centers[peaks])]
#     fid_peak_center = bin_centers[fid_peak_idx]

#     range_min = fid_peak_center - window_width
#     range_max = fid_peak_center + window_width

#     mask = (flat >= range_min) & (flat <= range_max)
#     fiducial_vals = flat[mask]

#     if fiducial_vals.size == 0:
#         raise ValueError("No pixels found in the selected fiducial intensity range.")

#     # fiducial_mean = np.mean(fiducial_vals)
#     fiducial_mean = fid_peak_center
#     normalized_image = image / fiducial_mean

#     if plot_hist:
#         plt.figure(figsize=(8, 4))
#         plt.plot(bin_centers, hist_vals, label='Histogram', alpha=0.3)
#         plt.plot(bin_centers, smoothed, label='Smoothed Histogram', color='black')
#         plt.axvline(range_min, color=c, linestyle='--', label='Range min')
#         plt.axvline(range_max, color=c, linestyle='--', label='Range max')
#         plt.axvline(fid_peak_center, color='red', linestyle='--', label='Fiducial peak')
#         plt.title("Histogram and Fiducial Detection")
#         plt.xlabel("Intensity")
#         plt.ylabel("Density")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#     return normalized_image, fiducial_mean

# def auto_normalize_by_fiducial_kmeans(image, fiduc_num, factor,
#                                      half_size=15,
#                                      plot=False, sample='', Gamma=''):
#     """
#     Normalize an image by the mean intensity of its fiducial hole pixels.

#     Fiducial positions are detected automatically each call using
#     find_fiducial_points (KMeans clustering on bright pixels), so the
#     normalization adapts to per-scan fiducial drift without requiring
#     fixed coordinates.

#     Pipeline
#     --------
#     1. find_fiducial_points  ->  (y, x) centroid for each hole
#     2. Crop a (2*half_size x 2*half_size) patch around each centroid
#     3. Average valid (finite, non-zero) pixels across all crops
#     4. Divide image by that mean

#     Parameters
#     ----------
#     image : 2D np.ndarray
#         Raw amplitude image. Holes must appear as bright regions
#         (passed directly to find_fiducial_points without inversion).
#     fiduc_num : int
#         Number of fiducial holes expected in the image.
#     factor : float
#         Intensity threshold factor passed to find_fiducial_points.
#         Pixels above (max_valid / factor) are treated as hole candidates.
#         Typical value: 1.25.
#     half_size : int
#         Half-width in pixels of the square crop around each fiducial
#         centroid. Set once in your configuration cell. Default: 15.
#     plot : bool
#         If True, display the image with detected fiducial boxes overlaid,
#         and print per-box means for consistency checking.
#     sample : str
#         Sample identifier for plot titles.
#     Gamma : str
#         Gamma label for plot titles.

#     Returns
#     -------
#     normalized_image : 2D np.ndarray
#         Image divided by fiducial_mean.
#     fiducial_mean : float
#         Mean intensity across all valid fiducial crop pixels combined.
#     fiduc_points : np.ndarray, shape (fiduc_num, 2)
#         Detected fiducial centroids (y, x) from find_fiducial_points.
#         Useful for downstream inspection or drift tracking.
#     box_means : list of float
#         Mean intensity of each individual fiducial crop. If one box
#         differs significantly from the others it may be clipping onto
#         the membrane edge or a padding boundary.
#     """
#     # --- Step 1: detect fiducial positions -----------------------------------
#     fiduc_points = find_fiducial_points(
#         image, fiduc_num, factor, verbose=False
#     )

#     if np.all(np.isnan(fiduc_points)):
#         raise ValueError(
#             "find_fiducial_points returned all NaN — no fiducials detected. "
#             "Try adjusting factor or check that holes are bright in this image."
#         )

#     # --- Step 2: crop around each centroid and collect pixel values ----------
#     all_fiducial_vals = []
#     box_means = []
#     valid_points = []

#     img_h, img_w = image.shape

#     for i, (cy, cx) in enumerate(fiduc_points):
#         if np.isnan(cy) or np.isnan(cx):
#             print(f"  Warning: fiducial {i} centroid is NaN — skipping.")
#             continue

#         cy, cx = int(round(cy)), int(round(cx))

#         # Clamp box to image bounds so edge fiducials don't raise an error
#         y0 = max(0,     cy - half_size)
#         y1 = min(img_h, cy + half_size)
#         x0 = max(0,     cx - half_size)
#         x1 = min(img_w, cx + half_size)

#         crop = image[y0:y1, x0:x1]
#         n_pixels = int(np.sum(crop))

#         # Exclude padding zeros and non-finite values within the crop
#         valid = crop[np.isfinite(crop) & (crop > 0)]

#         if valid.size == 0:
#             print(f"  Warning: crop around fiducial {i} at ({cy}, {cx}) "
#                   f"contains no valid pixels — skipping.")
#             continue

#         box_mean = np.mean(valid)
#         box_means.append(box_mean)
#         all_fiducial_vals.extend(valid.tolist())
#         valid_points.append((i, cy, cx, y0, y1, x0, x1))

#     if len(all_fiducial_vals) == 0:
#         raise ValueError(
#             "All fiducial crops were empty or invalid. "
#             "Check that half_size is appropriate and that fiducials are "
#             "not near the image border."
#         )

#     # --- Step 3: combined mean and sanity check ------------------------------
#     fiducial_mean = np.mean(all_fiducial_vals)

#     # # Warn if individual boxes disagree — may indicate one crop has drifted
#     # # onto the membrane edge or a zero-padding region
#     # if len(box_means) > 1:
#     #     spread = np.max(box_means) - np.min(box_means)
#     #     if spread / fiducial_mean > 0.05:
#     #         print(f"  Warning: fiducial box means span {spread:.4f} "
#     #               f"({100 * spread / fiducial_mean:.1f}% of combined mean). "
#     #               f"Individual means: {[f'{m:.4f}' for m in box_means]}. "
#     #               f"Consider raising half_size or inspecting with plot=True.")

#     # --- Step 4: normalize ---------------------------------------------------
#     normalized_image = image / fiducial_mean

#     # --- Diagnostic plot -----------------------------------------------------
#     if plot:
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#         # Left: image with fiducial boxes overlaid
#         axes[0].imshow(image, cmap='viridis')
#         for j, (i, cy, cx, y0, y1, x0, x1) in enumerate(valid_points):
#             rect = plt.Rectangle(
#                 (x0, y0), x1 - x0, y1 - y0,
#                 linewidth=1.5, edgecolor='red', facecolor='none'
#             )
#             axes[0].add_patch(rect)
#             axes[0].text(
#                 x1 + 3, (y0 + y1) / 2,
#                 f"{j}: {box_means[j]:.2f}",
#                 color='red', fontsize=8, va='center'
#             )
#         # axes[0].set_title(
#         #     f"Fiducial crops — combined mean = {fiducial_mean:.4f}\n"
#         #     f"Sample {sample}, {Gamma}"
#         # )

#         # Right: bar chart of per-box means with combined mean marked
#         axes[1].bar(range(len(box_means)), box_means, color='steelblue',
#                     alpha=0.7, width=0.5)
#         axes[1].axhline(fiducial_mean, color='red', linestyle='--',
#                         linewidth=1.5, label=f'Combined mean = {fiducial_mean:.4f}')
#         axes[1].set_xticks(range(len(box_means)))
#         axes[1].set_xticklabels([f"Box {j}" for j in range(len(box_means))])
#         # axes[1].set_ylabel("Mean intensity")
#         axes[1].set_title("Per-fiducial means")
#         axes[1].legend(fontsize=9)

#         # plt.suptitle(f"auto_normalize_by_fiducial_crop diagnostic", fontsize=11)
#         plt.tight_layout()
#         plt.show()

#     return normalized_image, fiducial_mean, fiduc_points, box_means, n_pixels


def auto_normalize_by_fiducial_threshold(image, c, bins=100,
                                stdev_threshold = 5,
                                plot_hist=False):
    """
    Updates to auto_normalization include Normalizing an image by the mean intensity of its masked fiducial
    marker.

    The fiducial is identified as the rightmost (highest-intensity) peak in
    the intensity histogram. A crop around this fiducial is taken to help view the histogram

    Parameters
    ----------
    image : 2D np.ndarray
        Input amplitude or intensity image.
    c : str
        Matplotlib color string used for histogram range lines.
    bins : int
        Number of histogram bins.
    stdev_threshold : float 
        How large of an intensity window around the fiducial peak to use for plotting
        (in the same units as image intensity).
    plot_hist : bool
        If True, display a diagnostic histogram plot. Set to False to
        suppress output and reduce memory usage during batch processing.

    Returns
    -------
    normalized_image : 2D np.ndarray
        Image divided by the fiducial mean intensity.
    fiducial_mean : float
        Mean intensity of pixels identified as the fiducial.
    """
    # create histogram of pixel intensities
    valid_mask = np.isfinite(image) & (image > 0)
    flat = image[valid_mask].flatten()

    flat = image.flatten()
    hist_vals, bin_edges = np.histogram(flat, bins=bins, density=True) 
    x_axis = (bin_edges[:-1] + bin_edges[1:]) / 2  

    # Find peak in histogram
    peak_prominence = 0.01
    peak, props = find_peaks(hist_vals, prominence=peak_prominence)

    if len(peak) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_axis, hist_vals, alpha=0.4, label='Histogram')
        ax.set_yscale('log')
        ax.set_title("No peak found")
        ax.set_xlabel("Intensity")
        ax.legend()
        plt.tight_layout()
        plt.show()
        raise ValueError("No peaks found in histogram. Try lowering peak_prominence.")

    # assume domain peak is the lowest peak
    peak_idx = peak[np.argmin(x_axis[peak])]
    peak_center_x = x_axis[peak_idx]

    # Define threshold within some window of the domain peak
    fit_window = len(x_axis) // 4
    low = max(0, peak_idx - fit_window)
    hi = min(len(x_axis), peak_idx + fit_window)

    try: #fitting a gaussian to the peak

        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
        p0 = [hist_vals[peak_idx], peak_center_x, (x_axis[-1] - x_axis[0]) / 10]
        popt, _ = curve_fit(gaussian, x_axis[low:hi], hist_vals[low:hi], p0=p0, maxfev=5000)
        _, fitted_mu, fitted_sigma = popt
        fitted_sigma = abs(fitted_sigma)

        smoothed = gaussian(x_axis, popt[0], popt[1], popt[2])
        
    except Exception as e: # If fit fails, print error and exit function
        raise ValueError(f"Gaussian fit failed")

    # threshold = mean + N * sigma
    threshold = fitted_mu + stdev_threshold * fitted_sigma

    # Build binary mask: 1 where image exceeds threshold, 0 elsewhere
    mask = (image > threshold) & valid_mask
    fiducial_vals = image[mask]
    n_pixels = int(np.sum(mask))
    
    if n_pixels == 0:
        raise ValueError(
            f"No pixels above threshold ({threshold:.4f}). "
            f"Try lowering stdev_threshold (currently {stdev_threshold})."
        )

    ## normalize by fiducial mean taken from the mask
    fiducial_mean = np.mean(fiducial_vals)


    if fiducial_mean <= threshold:
        raise ValueError(
            f"fiducial_mean ({fiducial_mean:.4f}) is not above threshold "
            f"({threshold:.4f}). This means the mask captured the wrong pixels. "
            f"Check that stdev_threshold ({stdev_threshold}) is high enough to "
            f"clear the sample distribution tail, or inspect the histogram with "
            f"plot_hist=True."
        )
        
    normalized_image = image / fiducial_mean

            
    if plot_hist:
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, hist_vals, label='Histogram', alpha=0.3)
        plt.plot(x_axis, smoothed, label='Gaussian Fit', color='black')
        plt.axvline(threshold, color = c, linestyle='--', label=f'Threshold (μ + {stdev_threshold}σ = {threshold:.4f})')
        plt.axvline(fitted_mu, color='red', linestyle=':', label=f'Peak μ = {fitted_mu:.4f}')
        plt.axvline(fiducial_mean, color='green', linestyle=':', label=f'normalization constant = {fiducial_mean:.4f}')
        plt.title("Histogram and Threshold Detection")
        plt.xlabel("Intensity")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, hist_vals, label='Histogram', alpha=0.3)
        plt.axvline(fiducial_mean, color='green', linestyle=':', label=f'normalization constant = {fiducial_mean:.4f}')
        plt.axvline(threshold, color = c, linestyle='--', label=f'Threshold (μ + {stdev_threshold}σ = {threshold:.4f})')
        plt.yscale('log')
        plt.xlim(threshold, )
        # plt.ylim(0, 0.005)
        plt.title("Histogram and Threshold Detection")
        plt.xlabel("Intensity")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()


        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        norm_CC = mpl.colors.Normalize(vmin=0, vmax=1)
        show = axes.imshow(mask, norm = norm_CC)
        axes.set_title(f'threshold normalization, masked area')
        fig.colorbar(show)
        plt.tight_layout()
        plt.show()

        


    return normalized_image, fiducial_mean, n_pixels, mask, threshold, n_pixels


#################################################################################################

def auto_normalize_charge(image, c, bins=100,
                           peak_prominence=0.0001,
                           window_width=0.1,
                           plot_hist=False):
    """
    Normalize a charge contrast image by the mean intensity of the
    lowest-intensity (leftmost) histogram peak.

    Parameters
    ----------
    image : 2D np.ndarray
        Charge contrast image (finite values only are used).
    c : str
        Matplotlib color string for histogram range lines.
    bins : int
        Number of histogram bins.
    peak_prominence : float
        Minimum prominence for peak detection.
    window_width : float
        Half-width of the intensity window around the detected peak.
    plot_hist : bool
        If True, display a diagnostic histogram. Set to False during
        batch processing to reduce memory usage.

    Returns
    -------
    range_min, range_max : float
        Intensity range used for normalization.
    normalized_image : 2D np.ndarray
        Image divided by the charge contrast peak mean.
    fiducial_mean : float
        Mean intensity of the pixels used for normalization.
    """
    flat = image[np.isfinite(image)].flatten()
    hist_vals, bin_edges = np.histogram(flat, bins=bins, density=True)
    smoothed = gaussian_filter1d(hist_vals, sigma=2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(smoothed, prominence=peak_prominence)

    if len(peaks) < 2:
        raise ValueError("Could not find at least two peaks in histogram. "
                         "Try lowering peak_prominence.")

    # Charge contrast reference = leftmost (lowest-intensity) peak
    fid_peak_idx = peaks[np.argmin(bin_centers[peaks])]
    fid_peak_center = bin_centers[fid_peak_idx]

    range_min = fid_peak_center - window_width
    range_max = fid_peak_center + window_width

    mask = (flat >= range_min) & (flat <= range_max)
    fiducial_vals = flat[mask]

    if fiducial_vals.size == 0:
        raise ValueError("No pixels found in the selected charge contrast range.")

    fiducial_mean = np.mean(fiducial_vals)
    normalized_image = image / fiducial_mean

    if plot_hist:
        plt.figure(figsize=(8, 4))
        plt.plot(bin_centers, hist_vals, label='Raw Histogram', alpha=0.3)
        plt.plot(bin_centers, smoothed, label='Smoothed Histogram', color='black')
        plt.axvline(range_min, color=c, linestyle='--', label='Auto range min')
        plt.axvline(range_max, color=c, linestyle='--', label='Auto range max')
        plt.axvline(fid_peak_center, color='red', linestyle='--',
                    label='Detected charge peak')
        plt.title("Histogram and Charge Contrast Detection")
        plt.xlabel("Intensity")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return range_min, range_max, normalized_image, fiducial_mean


def auto_normalize_OD_charge(image, c, bins=100,
                              peak_prominence=0.0001,
                              window_width=0.15,
                              plot_hist=False):
    """
    Normalize an optical density (OD) charge contrast image by the mean
    of the highest-intensity histogram peak.

    Unlike auto_normalize_charge, this targets the rightmost peak, which
    corresponds to the dominant OD signal rather than the background.

    Parameters
    ----------
    image : 2D np.ndarray
        OD-converted charge contrast image.
    c : str
        Matplotlib color string for histogram lines.
    bins : int
        Number of histogram bins.
    peak_prominence : float
        Minimum prominence for peak detection. A diagnostic plot is shown
        automatically if fewer than one peak is found.
    window_width : float
        Half-width of the intensity window around the detected peak.
    plot_hist : bool
        If True, display a diagnostic histogram. Set to False during
        batch processing to reduce memory usage.

    Returns
    -------
    range_min, range_max : float
        Intensity range around the detected peak.
    normalized_image : 2D np.ndarray
        Image divided by the peak mean.
    fiducial_mean : float
        Mean intensity used for normalization.
    """
    flat = image[np.isfinite(image)].flatten()
    hist_vals, bin_edges = np.histogram(flat, bins=bins, density=True)
    smoothed = gaussian_filter1d(hist_vals, sigma=2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    peaks, _ = find_peaks(smoothed, prominence=peak_prominence)

    if len(peaks) < 1:
        # Show a diagnostic plot to help the user tune peak_prominence
        plt.figure(figsize=(8, 4))
        plt.plot(bin_centers, hist_vals, label='Raw Histogram', alpha=0.3)
        plt.plot(bin_centers, smoothed, label='Smoothed Histogram', color='black')
        plt.title(f"Histogram -- Peak Finder FAILED "
                  f"(prominence = {peak_prominence})")
        plt.xlabel("Intensity")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()
        raise ValueError("Could not find any peaks in histogram. "
                         "Try lowering peak_prominence.")

    # OD charge reference = rightmost (highest-intensity) peak
    fid_peak_idx = peaks[np.argmax(bin_centers[peaks])]
    fid_peak_center = bin_centers[fid_peak_idx]

    range_min = fid_peak_center - window_width
    range_max = fid_peak_center + window_width

    mask = (flat >= range_min) & (flat <= range_max)
    mask_img = (image >= range_min) & (image <= range_max)
    fiducial_vals = flat[mask]

    if fiducial_vals.size == 0:
        raise ValueError("No pixels found in the selected OD charge range.")

    masked_pixels = image[mask_img]

    fiducial_mean = np.mean(fiducial_vals)
    normalized_image = image / fiducial_mean

    if plot_hist:
        plt.figure(figsize=(8, 4))
        plt.plot(bin_centers, hist_vals, label='Raw Histogram', alpha=0.3)
        plt.plot(bin_centers, smoothed, label='Smoothed Histogram', color='black')
        plt.axvline(range_min, color=c, linestyle='--', label='Auto range min')
        plt.axvline(range_max, color=c, linestyle='--', label='Auto range max')
        plt.axvline(fid_peak_center, color='red', linestyle='--',
                    label='Detected OD peak')
        plt.title("Histogram and OD Charge Contrast Detection")
        plt.xlabel("Intensity")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return range_min, range_max, normalized_image, fiducial_mean


# -----------------------------------------------------------------------
# SECTION 4: Masking and Error Metrics
# -----------------------------------------------------------------------

def mask_holes_to_one(img_pos, img_neg, factor):
    """
    Create binary masks marking high-intensity pixels (holes / fiducials)
    as 1 and all other pixels as 0.

    Pixels above `factor * image_max` are considered part of a hole. This
    mask is used to focus cross-correlation on fiducial features only.

    Parameters
    ----------
    img_pos : 2D np.ndarray
        Normalized positive-polarization image.
    img_neg : 2D np.ndarray
        Normalized negative-polarization image.
    factor : float
        Fraction of the maximum intensity above which a pixel is classified
        as a hole (e.g., 0.7 masks the top 30% of intensity range).

    Returns
    -------
    first_image : 2D np.ndarray
        Binary mask for img_pos (1 where pixel > factor * max).
    second_image : 2D np.ndarray
        Binary mask for img_neg (1 where pixel > factor * max).
    """
    first_image = np.zeros_like(img_pos)
    first_image[img_pos > img_pos.max() * factor] = 1

    second_image = np.zeros_like(img_neg)
    second_image[img_neg > img_neg.max() * factor] = 1

    return first_image, second_image


def calc_NRMSE(input_array, ref_array):
    """
    Calculate the Normalized Root Mean Square Error (NRMSE) between two
    images.

    NRMSE = RMSE / (max(ref) - min(ref))

    A value of 0 indicates perfect agreement; larger values indicate
    greater misalignment.

    Parameters
    ----------
    input_array : np.ndarray
        Array to evaluate (e.g., aligned negative-polarization OD).
    ref_array : np.ndarray
        Reference array (e.g., positive-polarization OD).

    Returns
    -------
    NRMSE : float
        Normalized error metric (dimensionless).
    """
    NRMSE = (np.sqrt(np.mean((ref_array - input_array) ** 2)) /
             (np.max(ref_array) - np.min(ref_array)))
    return NRMSE


# -----------------------------------------------------------------------
# SECTION 5: Background Fitting
# -----------------------------------------------------------------------

def background_fcn(coords, a, b, c):
    """
    Planar background model: f(x, y) = a*x + b*y + c.

    Used internally by fit_background as the model passed to curve_fit.

    Parameters
    ----------
    coords : tuple of np.ndarray
        (x, y) coordinate arrays (raveled).
    a, b, c : float
        Plane coefficients.

    Returns
    -------
    np.ndarray
        Raveled plane values at each (x, y) coordinate.
    """
    x, y = coords
    return a * x + b * y + c


def fit_background(image):
    """
    Fit a planar (first-order) background to an image and return it.

    The fitted plane can be subtracted from the image to remove slow
    intensity gradients across the field of view.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image.

    Returns
    -------
    B : 2D np.ndarray
        Fitted planar background with the same shape as `image`.
    """
    ny, nx = image.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x_flat = x.ravel()
    y_flat = y.ravel()
    flat_img = image.ravel()

    popt, _ = curve_fit(background_fcn, (x_flat, y_flat), flat_img)
    a,b,c = popt
    slope = a*x + b*y
    return slope


# -----------------------------------------------------------------------
# SECTION 6: Cosine Fit (for angle calibration)
# -----------------------------------------------------------------------

def cos_fit(x, A, yo, phase):
    """
    Cosine model for fitting NRMSE vs. tilt angle.

    The period is fixed by the data range: b = pi / range(x).
    The fit identifies the angle of maximum magnetic contrast,
    which is then set as the calibrated zero angle.

    Parameters
    ----------
    x : np.ndarray
        Tilt angle values (in radians).
    A : float
        Amplitude of the cosine.
    yo : float
        Vertical offset.
    phase : float
        Phase shift (radians).

    Returns
    -------
    np.ndarray
        Cosine model evaluated at each value of x.
    """
    b_fix = np.pi / np.ptp(x)
    return A * np.cos(b_fix * x + phase) + yo


# -----------------------------------------------------------------------
# SECTION 7: Power Spectral Density (PSD) Quality Check
# -----------------------------------------------------------------------

def compute_fft(image, plot=False, label='', angles_pospol=None, k=None):
    """
    Normalize an image and compute its 2D Power Spectral Density (PSD).

    The image is min-max normalized before the FFT so that PSD values are
    comparable across scans with different absolute intensities.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image patch (e.g., a cropped region of charge contrast).
    plot : bool
        If True, display the FFT and PSD images.
    label : str
        Label string used in plot titles (e.g., 'Sample 452, Gamma90').
    angles_pospol : np.ndarray or None
        Angle array, used only for plot titles when plot=True.
    k : int or None
        Scan index, used only for plot titles when plot=True.

    Returns
    -------
    psd : 2D np.ndarray
        Power spectral density (squared magnitude of centered FFT).
    """
    image = (image - image.min()) / (image.max() - image.min())

    F = fftn(image)
    F_shifted = fftshift(F)
    psd = np.abs(F_shifted) ** 2

    if plot and angles_pospol is not None and k is not None:
        norm_vis = mpl.colors.Normalize(vmin=0, vmax=50)
        norm_psd = mpl.colors.Normalize(vmin=0, vmax=10000)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
        fft_plot = axes[0].imshow(F_shifted.real, cmap='hot', norm=norm_vis)
        fig.colorbar(fft_plot, ax=axes[0])
        axes[0].set_title(f"FFT -- {label} @ {angles_pospol[k]:.2f}")
        psd_plot = axes[1].imshow(psd, cmap='hot', norm=norm_psd)
        fig.colorbar(psd_plot, ax=axes[1])
        axes[1].set_title(f"PSD -- {label} @ {angles_pospol[k]:.2f}")
        plt.tight_layout()
        plt.show()

    return psd


def radial_average(psd):
    """
    Compute the radially averaged 1D profile of a 2D PSD.

    Pixels at the same distance from the image center are averaged together,
    reducing the 2D PSD to a 1D radial power spectrum.

    Parameters
    ----------
    psd : 2D np.ndarray
        Power spectral density image (must be square or nearly square).

    Returns
    -------
    radialprofile : 1D np.ndarray
        Mean PSD value at each integer radial distance from the center.
    """
    y, x = np.indices(psd.shape)
    center = np.array([(x.max() - x.min()) / 2.0,
                       (y.max() - y.min()) / 2.0])
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(int)

    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile


def compare_CC(unaligned, aligned, sub_b, plot=False,
               band=(2, 6), label1='', label2 = '', label3 = '', angles_pospol=None, k=None):
    """
    Compare the radially averaged PSD of unaligned, aligned, and
    background-subtracted charge contrast images within a spatial
    frequency band.

    A higher in-band PSD power indicates more structured (non-noise)
    signal. Scans with in-band power above the noise threshold are
    flagged for review.

    Parameters
    ----------
    unaligned : 2D np.ndarray
        Cropped unaligned charge contrast patch.
    aligned : 2D np.ndarray
        Cropped aligned charge contrast patch.
    sub_b : 2D np.ndarray
        Cropped background-subtracted (or substitute) patch.
    plot : bool
        If True, display the radial PSD profiles.
    band : tuple of int
        (fmin, fmax) spatial frequency range (pixels) to integrate.
    label : str
        Label string for plot titles.
    angles_pospol : np.ndarray or None
        Angle array for plot titles.
    k : int or None
        Scan index for plot titles.

    Returns
    -------
    band_u : float
        Integrated PSD power of the unaligned image in `band`.
    band_a : float
        Integrated PSD power of the aligned image in `band`.
    band_sb : float
        Integrated PSD power of the background-subtracted image in `band`.
    """
    psd_u = compute_fft(unaligned, plot=plot, label=label1,
                        angles_pospol=angles_pospol, k=k)
    psd_a = compute_fft(aligned, plot=plot, label=label2,
                        angles_pospol=angles_pospol, k=k)
    psd_sb = compute_fft(sub_b, plot=plot, label=label3,
                         angles_pospol=angles_pospol, k=k)

    rad_u = radial_average(psd_u)
    rad_a = radial_average(psd_a)
    rad_sb = radial_average(psd_sb)

    if plot:
        fig, axes = plt.subplots(1, 1, figsize=(10.5, 3.5))
        axes.plot(rad_u, label=f'{label1}')
        axes.plot(rad_a, label=f'{label2}')
        axes.plot(rad_sb, label=f'{label3}')
        axes.legend()
        #title = f"Radial PSD -- {label}"
        # if angles_pospol is not None and k is not None:

        axes.set_title(f" Radial integration @ {angles_pospol[k]:.2f}")
        axes.set_xlim(band)
        axes.set_ylim(0, 2000)
        plt.tight_layout()
        plt.show()

    fmin, fmax = band
    band_u = np.sum(rad_u[fmin:fmax])
    band_a = np.sum(rad_a[fmin:fmax])
    band_sb = np.sum(rad_sb[fmin:fmax])

    return band_u, band_a, band_sb


def generate_uniform_noise_image(shape, low, high):
    """
    Generate a uniform noise image as a noise baseline for PSD comparison.

    Parameters
    ----------
    shape : tuple of int
        Image shape (height, width).
    low, high : float
        Lower and upper bounds of the uniform distribution.

    Returns
    -------
    np.ndarray
        Uniform random noise image.
    """
    return np.random.uniform(low, high, size=shape)


def generate_gaussian_noise_image(shape, mean, stdev):
    """
    Generate a Gaussian noise image as a noise baseline for PSD comparison.

    Parameters
    ----------
    shape : tuple of int
        Image shape (height, width).
    mean : float
        Mean of the Gaussian distribution.
    stdev : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    np.ndarray
        Gaussian random noise image.
    """
    return np.random.normal(loc=mean, scale=stdev, size=shape)


def PSD_Check(image_stack, region, center_y, center_x, half_size,
              image_stack_pospol, avg_mean, avg_var, band=(2, 6), set_plot = False):
    """
    Evaluate every scan in a stack for domain signal in the CC using PSD analysis.

    For each scan, a small patch is cropped at the specified region and its
    in-band PSD power is compared against three noise thresholds derived
    from synthetic uniform and Gaussian noise images.

    Scans exceeding a threshold are considered to have structured signal
    (i.e., potential alignment or magnetic artifacts) and are flagged.

    Parameters
    ----------
    image_stack : 3D np.ndarray
        Stack of charge contrast images (n_scans, height, width).
    region : int
        Index into center_y / center_x defining the patch location.
    center_y, center_x : np.ndarray
        Arrays of y and x patch center coordinates.
    half_size : int
        Half-width of the square patch to crop.
    image_stack_pospol : 3D np.ndarray
        Positive polarization stack (used only for shape / loop count).
    avg_mean, avg_var : float
        Mean and variance of the aligned charge contrast (from regional
        analysis), used to parameterize the noise models.
    band : tuple of int
        (fmin, fmax) spatial frequency band to integrate.

    Returns
    -------
    band_vals : 1D np.ndarray
        In-band PSD power for each scan.
    scans_to_check_uniform : list of int
        Scan indices exceeding the uniform noise threshold.
    scans_to_check_gauss : list of int
        Scan indices exceeding the Gaussian noise threshold.
    scans_to_check_avg : list of int
        Scan indices exceeding the average of the two noise thresholds.
    """
    # Build noise baselines from the observed mean and variance
    patch_shape = (half_size * 2, half_size * 2)
    noise_uniform = generate_uniform_noise_image(
        patch_shape,
        avg_mean - np.sqrt(avg_var),
        avg_mean + np.sqrt(avg_var)
    )
    noise_gauss = generate_gaussian_noise_image(
        patch_shape, avg_mean, np.sqrt(avg_var)
    )
    band_noise_unif, band_noise_gauss, _ = compare_CC(
        noise_uniform, noise_gauss, noise_gauss, plot=set_plot, band=band
    )
    band_noise_avg = np.mean([band_noise_unif, band_noise_gauss])

    band_vals = np.zeros(image_stack_pospol.shape[0])
    scans_to_check_uniform = []
    scans_to_check_gauss = []
    scans_to_check_avg = []

    for k in range(image_stack_pospol.shape[0]):
        cy, cx = center_y[region], center_x[region]
        cropped = image_stack[k][cy - half_size:cy + half_size,
                                 cx - half_size:cx + half_size]
        band_vals[k], _, _ = compare_CC(cropped, cropped, cropped,
                                        plot=set_plot, band=band)

        if band_vals[k] > band_noise_unif:
            scans_to_check_uniform.append(k)
        if band_vals[k] > band_noise_gauss:
            scans_to_check_gauss.append(k)
        if band_vals[k] > band_noise_avg:
            scans_to_check_avg.append(k)

    return band_vals, scans_to_check_uniform, scans_to_check_gauss, scans_to_check_avg


# -----------------------------------------------------------------------
# SECTION 8: Plotting Utilities
# -----------------------------------------------------------------------

def plot_XMCD_CC_normalizeCC(XMCD, CC, angle, sample, Gamma,
                              FOV_x=4.0, FOV_y=4.0):
    """
    Display XMCD and charge contrast images side-by-side with physical
    axis labels in micrometers.

    The charge contrast display range is determined automatically by
    auto_normalize_OD_charge.

    Parameters
    ----------
    XMCD : 2D np.ndarray
        XMCD image (neg - pos OD).
    CC : 2D np.ndarray
        Charge contrast image (neg + pos OD).
    angle : float
        Tilt angle in degrees (used in title).
    sample : str
        Sample identifier (used in title).
    Gamma : str
        Gamma label, e.g. 'Gamma90' (used in title).
    FOV_x, FOV_y : float
        Field of view in micrometers for axis scaling.
    """
    color_cc = 'green'
    range_min, range_max, _, _ = auto_normalize_OD_charge(
        CC, color_cc, plot_hist=False
    )

    norm_XMCD = mpl.colors.Normalize(vmin=-0.4, vmax=0.4)
    norm_charge = mpl.colors.Normalize(vmin=range_min, vmax=range_max)

    extent = [0, FOV_x, 0, FOV_y]
    ticks_um = np.arange(0, FOV_x + 0.1, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axes[0].imshow(XMCD, norm=norm_XMCD, extent=extent, origin='lower')
    axes[0].set_title(f"XMCD\nSample {sample}, {Gamma} @ {angle:.2f}")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[0].set_xticks(ticks_um)
    axes[0].set_yticks(ticks_um)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(CC, norm=norm_charge, extent=extent, origin='lower')
    axes[1].set_title(f"Charge Contrast\nSample {sample}, {Gamma} @ {angle:.2f}")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    axes[1].set_xticks(ticks_um)
    axes[1].set_yticks(ticks_um)
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


def plot_XMCD_ASYM(XMCD, ASYM, angle, sample, Gamma,
                              FOV_x=4.0, FOV_y=4.0):
    """
    Display XMCD and charge contrast images side-by-side with physical
    axis labels in micrometers.

    The charge contrast display range is determined automatically by
    auto_normalize_OD_charge.

    Parameters
    ----------
    XMCD : 2D np.ndarray
        XMCD image (neg - pos OD).
    ASYM : 2D np.ndarray
        XMCD /  CC
    angle : float
        Tilt angle in degrees (used in title).
    sample : str
        Sample identifier (used in title).
    Gamma : str
        Gamma label, e.g. 'Gamma90' (used in title).
    FOV_x, FOV_y : float
        Field of view in micrometers for axis scaling.
    """
    norm_XMCD = mpl.colors.Normalize(vmin=-0.4, vmax=0.4)

    extent = [0, FOV_x, 0, FOV_y]
    ticks_um = np.arange(0, FOV_x + 0.1, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axes[0].imshow(XMCD, norm=norm_XMCD, extent=extent, origin='lower')
    axes[0].set_title(f"XMCD\nSample {sample}, {Gamma} @ {angle:.2f}")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[0].set_xticks(ticks_um)
    axes[0].set_yticks(ticks_um)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(ASYM, norm=norm_XMCD, extent=extent, origin='lower')
    axes[1].set_title(f"Asymmetry\nSample {sample}, {Gamma} @ {angle:.2f}")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    axes[1].set_xticks(ticks_um)
    axes[1].set_yticks(ticks_um)
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


def plot_XMCD_crosscorr(unaligned, aligned, crosscorr, angle, sample, Gamma):
    """
    Display three panels: unaligned XMCD, aligned XMCD, and the
    cross-correlation surface used to compute the alignment shift.

    Parameters
    ----------
    unaligned : 2D np.ndarray
        XMCD difference map before alignment.
    aligned : 2D np.ndarray
        XMCD difference map after alignment.
    crosscorr : 2D np.ndarray
        Cross-correlation window from align_images_CC.
    angle : float
        Tilt angle in degrees (used in title).
    sample : str
        Sample identifier (used in title).
    Gamma : str
        Gamma label (used in title).
    """
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    im0 = axes[0].imshow(unaligned, norm=norm)
    axes[0].set_title(f"XMCD Unaligned\nSample {sample}, {Gamma} @ {angle:.2f}")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(aligned, norm=norm)
    axes[1].set_title(f"XMCD Aligned\nSample {sample}, {Gamma} @ {angle:.2f}")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(crosscorr, cmap='hot')
    axes[2].set_title("Cross-Correlation Surface")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


def plot_CLCR(CL, CR, angle, sample, Gamma):
    """
    Display left-circular (CL) and right-circular (CR) polarization images
    side-by-side.

    Parameters
    ----------
    CL : 2D np.ndarray
        Left-circular polarization image.
    CR : 2D np.ndarray
        Right-circular polarization image.
    angle : float
        Tilt angle in degrees (used in title).
    sample : str
        Sample identifier (used in title).
    Gamma : str
        Gamma label (used in title).
    """
    norm_im = mpl.colors.Normalize(vmin=0, vmax=1.45)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axes[0].imshow(CL, norm=norm_im)
    axes[0].set_title(f"Left Polarization (CL)\nSample {sample}, {Gamma} @ {angle:.2f}")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(CR, norm=norm_im)
    axes[1].set_title(f"Right Polarization (CR)\nSample {sample}, {Gamma} @ {angle:.2f}")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


def plot_histogram(vals_neg, vals_pos, bin_size, color_neg, color_pos,
                   angle, sample, Gamma, label1= ' ', label2 = ' ', title=''):
    """
    Plot overlapping intensity histograms for negative and positive
    polarization images.

    Parameters
    ----------
    vals_neg : 1D np.ndarray
        Flattened pixel values from the negative-polarization image.
    vals_pos : 1D np.ndarray
        Flattened pixel values from the positive-polarization image.
    color_neg, color_pos : str
        Matplotlib color strings for each histogram.
    angle : float
        Tilt angle in degrees (used in title).
    sample : str
        Sample identifier (used in title).
    Gamma : str
        Gamma label (used in title).
    title : str
        Optional subtitle (e.g., 'Un-normalized', 'Normalized').
    """
    plt.figure()
    plt.hist(vals_neg, bins=bin_size, color=color_neg, label=f'{label1}', alpha=0.7)
    plt.hist(vals_pos, bins=bin_size, color=color_pos, label=f'{label2}', alpha=0.7)
    plt.xlabel('Intensity')
    plt.ylabel('Pixel Count')
    # plt.ylim(0, 1000)
    plt.legend()
    plt.title(f"Sample {sample}, {Gamma} @ {angle:.2f}\n{title}")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------
# SECTION 9: Alpha Alignment — Fiducial Centroid Tracking
# -----------------------------------------------------------------------

def calculate_centroid(image):
    """
    Compute the intensity-weighted centroid of an image.

    The centroid is the center of mass using pixel intensities as weights.
    Used to find the average position of fiducial hole clusters.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image (any dtype; converted to float64 internally).

    Returns
    -------
    y_centroid, x_centroid : float
        Intensity-weighted centroid coordinates (row, col).
    """
    image = np.array(image, dtype=np.float64)
    y_indices, x_indices = np.indices(image.shape)
    total_intensity = np.sum(image)
    x_centroid = np.sum(x_indices * image) / total_intensity
    y_centroid = np.sum(y_indices * image) / total_intensity
    return y_centroid, x_centroid


def calculate_centroid_points(points):
    """
    Compute the simple (unweighted) centroid of a set of 2D coordinate points.

    Used to find the geometric center of a set of fiducial hole positions
    returned by find_fiducial_points.

    Parameters
    ----------
    points : iterable of (y, x) tuples
        Fiducial point coordinates, e.g. a set of (y, x) tuples from KMeans.

    Returns
    -------
    centroid_y, centroid_x : float
        Arithmetic mean of y and x coordinates. Returns None if points is empty.
    """
    if not points:
        return None

    sum_y = 0.0
    sum_x = 0.0
    num_points = len(points)

    for y, x in points:
        sum_y += y
        sum_x += x

    return sum_y / num_points, sum_x / num_points


def find_fiducial_points(image, fiduc_num, factor, verbose=False):
    """
    Locate fiducial holes in an image using intensity thresholding and KMeans
    clustering, then compute the centroid of each cluster.

    The algorithm:
    1. Masks pixels above (max_valid / 1.25) as candidate hole pixels.
    2. Clusters those pixels into `fiduc_num` groups with KMeans.
    3. Returns the centroid of each cluster as a fiducial point.

    Parameters
    ----------
    image : 2D np.ndarray
        Image to search for fiducials (typically inverted charge contrast,
        e.g. (CC - 2) * -1 so that holes appear bright).
    fiduc_num : int
        Number of fiducial holes expected in the image.
    verbose : bool
        If True, display a diagnostic figure showing clustered pixels and
        detected fiducial positions overlaid on the hole mask.

    Returns
    -------
    fiduc_points : np.ndarray, shape (fiduc_num, 2)
        Centroid (y, x) coordinates for each detected fiducial hole.
        Returns an array of NaN if no valid pixels are found or if fewer
        than `fiduc_num` candidate pixels exist.
    """
    factor = factor #1.25

    # Only use finite (non-NaN, non-Inf) pixels for thresholding
    valid_mask = np.isfinite(image)
    if not np.any(valid_mask):
        return np.full((fiduc_num, 2), np.nan)

    hole_thresh = np.nanmax(image[valid_mask]) / factor
    hole_mask = np.zeros_like(image, dtype=np.uint8)
    hole_mask[(image > hole_thresh) & valid_mask] = 1

    k_data = np.column_stack(np.where(hole_mask > 0))  # shape (N, 2)
    if k_data.shape[0] < fiduc_num:
        print('Not enough points to cluster')
        return np.full((fiduc_num, 2), np.nan)

    kmeans = KMeans(n_clusters=fiduc_num, n_init=5)
    kmeans.fit(k_data)

    if verbose:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    fiduc_points = []
    for i in range(fiduc_num):
        cluster = k_data[kmeans.labels_ == i]
        if cluster.size == 0:
            fiduc_points.append((np.nan, np.nan))
            continue

        y_data = cluster[:, 0]
        x_data = cluster[:, 1]
        temp_mask = np.zeros_like(image, dtype=np.uint8)
        temp_mask[y_data, x_data] = 1

        y_c, x_c = calculate_centroid(temp_mask)
        fiduc_points.append((y_c, x_c))

        if verbose:
            axes[0].scatter(x_data, y_data)
            axes[0].plot(x_c, y_c, '+k')

    fiduc_points = np.array(fiduc_points)

    if verbose:
        axes[0].set_title("Clustered fiducials")
        axes[1].imshow(hole_mask)
        axes[1].plot(fiduc_points[:, 1], fiduc_points[:, 0], '+r')
        plt.tight_layout()
        plt.show()

    return fiduc_points




def perspectiveTransform_alignment(first_image, first_points, second_image,
                                   second_points, verbose=False):
    """
    Align two images using a perspective (homography) transform estimated
    from matched fiducial points.

    Fiducial points from each image are matched by nearest-neighbor distance,
    then OpenCV's getPerspectiveTransform is used to compute a 3x3 homography
    and warpPerspective is applied to the second image.

    Parameters
    ----------
    first_image : 2D np.ndarray
        Reference image (the target alignment frame).
    first_points : np.ndarray, shape (4, 2)
        Fiducial (y, x) coordinates in the reference image.
    second_image : 2D np.ndarray
        Image to be transformed.
    second_points : np.ndarray, shape (4, 2)
        Fiducial (y, x) coordinates in the second image.
    verbose : bool
        If True, print point-by-point transform residuals and show
        before/after difference maps.

    Returns
    -------
    transformed_image : 2D np.ndarray
        second_image after perspective alignment to first_image.
    """
    src_points = []
    dst_points = []
    for point in first_points:
        dists = np.linalg.norm(second_points - point, axis=1)
        index = np.argmin(dists)
        src_points.append(point)
        dst_points.append(second_points[index])

    src_points = np.array(src_points, dtype='float32')
    dst_points = np.array(dst_points, dtype='float32')

    A = cv2.getPerspectiveTransform(src_points, dst_points)

    src_pts_reshaped = src_points.reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(src_pts_reshaped, A).reshape(-1, 2)

    if verbose:
        for i in range(len(dst_points)):
            print(f"Target: {dst_points[i]}, Transformed: {transformed[i]}, "
                  f"Diff: {np.abs(dst_points[i] - transformed[i])}")

    height, width = first_image.shape
    transformed_image = cv2.warpPerspective(
        second_image, A, (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )

    orig_error  = calc_NRMSE(second_image, first_image)
    align_error = calc_NRMSE(transformed_image, first_image)

    if verbose:
        original_diffMap = second_image - first_image
        aligned_diffMap  = transformed_image - first_image
        norm = mpl.colors.Normalize(
            vmin=np.nanmin(original_diffMap),
            vmax=np.nanmax(original_diffMap)
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_diffMap, norm=norm)
        axes[0].set_title(f"Error Before ({orig_error*100:.2f}%)")
        axes[1].imshow(aligned_diffMap, norm=norm)
        axes[1].set_title(f"Error After ({align_error*100:.2f}%)")
        plt.tight_layout()
        plt.show()

    return transformed_image


# -----------------------------------------------------------------------
# SECTION 10: Alpha Alignment — Curve Fitting Models
# -----------------------------------------------------------------------

def sin_fit(x, A, yo, phase):
    """
    Sinusoidal model for fitting fiducial centroid y-position vs. tilt angle.

    The y-position of the fiducial centroid traces a sinusoid as the sample
    tilts, because the sample shifts vertically in projection. The period is
    fixed to the data range: b = pi / range(x).

    Parameters
    ----------
    x : np.ndarray
        Tilt angle values (in radians).
    A : float
        Amplitude of the sine.
    yo : float
        Vertical offset (mean centroid position).
    phase : float
        Phase shift (radians).

    Returns
    -------
    np.ndarray
        Sinusoid model evaluated at each value of x.
    """
    b_fix = np.pi / np.ptp(x)
    return A * np.sin(b_fix * x + phase) + yo


def lin_fit(x, m, xo):
    """
    Linear model for fitting fiducial centroid x-position vs. tilt angle.

    The x-position of the centroid typically drifts linearly across the
    tilt series (slow stage drift). Fitting and correcting this removes
    systematic x-offset from all scans.

    Parameters
    ----------
    x : np.ndarray
        Tilt angle values (in radians).
    m : float
        Slope (pixels per radian).
    xo : float
        Intercept (pixel offset at x=0).

    Returns
    -------
    np.ndarray
        Linear model evaluated at each value of x.
    """
    return m * x + xo


# -----------------------------------------------------------------------
# SECTION 11: Alpha Alignment — Plotting Utilities
# -----------------------------------------------------------------------

def plot_CC_crosscorr(first_CC, second_CC, crosscorr, angle1, angle2, sample, Gamma):
    """
    Display three panels: first CC image, second CC image, and the
    cross-correlation surface used to compute the alpha alignment shift.

    Parameters
    ----------
    first_CC : 2D np.ndarray
        Reference charge contrast image (scan k).
    second_CC : 2D np.ndarray
        Target charge contrast image (scan k+1, before shift).
    crosscorr : 2D np.ndarray
        Cross-correlation window from align_images_CC.
    angle1 : float
        Tilt angle of the reference scan (degrees).
    angle2 : float
        Tilt angle of the target scan (degrees).
    sample : str
        Sample identifier (used in title).
    Gamma : str
        Gamma label (used in title).
    """
    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    im0 = axes[0].imshow(first_CC, norm=norm)
    axes[0].set_title(f"CC First\nSample {sample}, {Gamma} @ {angle1:.2f}")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(second_CC, norm=norm)
    axes[1].set_title(f"CC Second\nSample {sample}, {Gamma} @ {angle2:.2f}")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(crosscorr, cmap='hot')
    axes[2].set_title("Cross-Correlation Surface")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()


def plot_centroid_fits(angles_deg, y_centroid, x_centroid,
                       y_fit, x_fit, y_fit_label, x_fit_label,
                       sample, Gamma, color_list, ylim=None):
    """
    Plot fiducial centroid y and x positions vs. tilt angle, overlaid
    with their fitted curves.

    Parameters
    ----------
    angles_deg : np.ndarray
        Tilt angle values in degrees (x-axis).
    y_centroid, x_centroid : array-like
        Measured centroid positions (pixels) for each scan.
    y_fit, x_fit : np.ndarray
        Fitted curve values at each angle.
    y_fit_label, x_fit_label : str
        Legend labels for the fit curves (include fit equation).
    sample : str
        Sample identifier (used in title).
    Gamma : str
        Gamma label (used in title).
    ylim : tuple of (float, float) or None
        Optional y-axis limits, e.g. (220, 320).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(angles_deg, y_centroid, color=color_list[0], marker='o',
            linewidth=0.5, label='y centroid')
    ax.plot(angles_deg, y_fit,      color=color_list[1],
            linewidth=2, label=y_fit_label)
    ax.plot(angles_deg, x_centroid, color=color_list[2], marker='o',
            linewidth=0.5, label='x centroid')
    ax.plot(angles_deg, x_fit,      color=color_list[3],
            linewidth=2, label=x_fit_label)
    ax.set_xlabel('Tilt angle (deg)')
    ax.set_ylabel('Centroid position (pixels)')
    ax.set_title(f"Fiducial Centroid vs. Tilt\nSample {sample}, {Gamma}")
    ax.legend(loc='upper left', fontsize=9)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()


# ============================ Other ============================
def ID_angles_to_delete(angles_negpol, angles_pospol):
    print("=" * 60)
    print(f"Angle inspection -- Sample {sample}, {Gamma}")
    print("=" * 60)
    
    suggested_del_neg  = []   # negpol-only candidates
    suggested_del_both = []   # both-stack candidates
    
    n_scans = angles_pospol.shape[0] - 1

    ct = 0
    for k in range(n_scans):
        angle_diff = angles_negpol[k] - angles_pospol[k]
    
        # Issue 1: pos/neg angle mismatch at this index
        if abs(angle_diff) > 0.5:
            if ct == 0: 
                ct = ct + 1
                k0 = k-1
                print(f"  [MISMATCH]  k={(k0):3d} | "
                      f"pos={angles_pospol[k0]:.2f}, neg={angles_negpol[k0]:.2f}, "
                      f"diff={angle_diff:.2f} -- consider deleting from negpol or pospol")
                print(f"  [MISMATCH]  k={k:3d} | "
                      f"pos={angles_pospol[k]:.2f}, neg={angles_negpol[k]:.2f}, "
                      f"diff={angle_diff:.2f} -- consider deleting from negpol or pospol")
            else: 
                print(f"  [MISMATCH]  k={k:3d} | "
                      f"pos={angles_pospol[k]:.2f}, neg={angles_negpol[k]:.2f}, "
                      f"diff={angle_diff:.2f} -- consider deleting from negpol or pospol")
            suggested_del_neg.append(k)
    
        # Issue 2: repeated angle in consecutive negpol scans
        if k < n_scans - 1:
            step = angles_negpol[k + 1] - angles_negpol[k]
            if abs(step) < 0.5:
                print(f"  [REPEAT]    k={k:3d}->{k+1} | "
                      f"neg[{k}]={angles_negpol[k]:.2f}, "
                      f"neg[{k+1}]={angles_negpol[k+1]:.2f}, "
                      f"step={step:.2f} -- consider deleting both")
                suggested_del_both.append(k)
        
    
    if not suggested_del_neg and not suggested_del_both:
        delete = False
        print("  No issues detected. All angle pairs appear consistent.")
    else:
        delete = True
        print(f"\nSuggested deletions:")
        print(f"  From negpol or pospol : {suggested_del_neg[0]}")
        print(f"  From both stacks : {suggested_del_both}")
        print("\nEdit and apply deletions in the next cell.")

def apply_deletion(angles_negpol, angles_pospol, image_stack_negpol, image_stack_pospol, del_neg_only, del_pos_only, del_both):
    # --- Apply deletions ---------------------------------------------------------
    print("=" * 60)
    print("Applying deletions...")
    
    if del_neg_only:
        print(f"  Removing {del_neg_only} from negpol "
              f"(angles: {angles_negpol[del_neg_only].tolist()})")
        angles_negpol      = np.delete(angles_negpol,      del_neg_only)
        image_stack_negpol = np.delete(image_stack_negpol, del_neg_only, axis=0)
    
    if del_pos_only:
        print(f"  Removing {del_pos_only} from pospol "
              f"(angles: {angles_pospol[del_pos_only].tolist()})")
        angles_pospol      = np.delete(angles_pospol,      del_pos_only)
        image_stack_pospol = np.delete(image_stack_pospol, del_pos_only, axis=0)
    
    if del_both:
        print(f"  Removing {del_both} from BOTH stacks")
        angles_negpol      = np.delete(angles_negpol,      del_both)
        image_stack_negpol = np.delete(image_stack_negpol, del_both, axis=0)
        angles_pospol      = np.delete(angles_pospol,      del_both)
        image_stack_pospol = np.delete(image_stack_pospol, del_both, axis=0)
    
    # --- Verify shapes match after deletions ------------------------------------
    print("\nPost-deletion shapes:")
    print(f"  pospol stack : {image_stack_pospol.shape}")
    print(f"  negpol stack : {image_stack_negpol.shape}")
    print(f"  pospol angles: {angles_pospol.shape}")
    print(f"  negpol angles: {angles_negpol.shape}")

    if angles_pospol.shape[0] == angles_negpol.shape[0]:
        delete = False
        print("\nAll shapes consistent. Ready to proceed.")
    else:
        delete = True
        ID_angles_to_delete(angles_negpol, angles_pospol)
        print("angle count does not match stack depth -- check deletion lists!")
    
    return angles_negpol, angles_pospol, image_stack_negpol, image_stack_pospol


################### Plotting ##########################

def _safe_peak_contrast(vals, bins=256, hist_range=None, prominence=None, distance=None):
    """
    Returns:
        contrast (float or None), n_peaks (int), peak_positions (np.ndarray)
    Contrast is abs difference between the two furthest detected peak centers.
    """
    vals = np.asarray(vals).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, 0, np.array([])

    counts, edges = np.histogram(vals, bins=bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Reasonable default prominence if not provided
    if prominence is None:
        prominence = 0.05 * np.max(counts) if np.max(counts) > 0 else 1

    peak_idx, _ = find_peaks(counts, prominence=prominence, distance=distance)
    n_peaks = len(peak_idx)

    if n_peaks < 2:
        return None, n_peaks, centers[peak_idx]

    peak_pos = np.sort(centers[peak_idx])
    contrast = float(peak_pos[-1] - peak_pos[0])  # two furthest peaks
    return contrast, n_peaks, peak_pos


# def plot_hist_compare_dict(
#     compare_dict,
#     bins=100,
#     alpha=0.5,
#     figsize=(12, 6),
#     show_mean_lines=True,
#     mean_line_colors=None,
#     hist_colors=None,
#     suptitle=None,
#     legend_fontsize=8
# ):
#     """
#     Plot histogram comparisons for multiple groups (e.g., CPL, CPR).

#     Parameters
#     ----------
#     compare_dict : dict
#         Structure:
#         {
#           "CPL": {"light off": img_array, "light on": img_array},
#           "CPR": {"light off": img_array, "light on": img_array},
#           ...
#         }
#     bins : int
#     alpha : float
#     figsize : tuple
#     show_mean_lines : bool
#     mean_line_colors : dict or None
#         e.g. {"light off": "orange", "light on": "blue"}
#     hist_colors : dict or None
#         e.g. {"light off": "#1f77b4", "light on": "#ff7f0e"}
#     suptitle : str or None
#     legend_fontsize : int
#     """
#     group_names = list(compare_dict.keys())
#     n_groups = len(group_names)

#     fig, axes = plt.subplots(1, n_groups, figsize=figsize, squeeze=False)
#     axes = axes[0]

#     for i, group in enumerate(group_names):
#         ax = axes[i]
#         condition_dict = compare_dict[group]  # e.g. {"light off": arr, "light on": arr}

#         for cond_name, img in condition_dict.items():
#             vals = np.asarray(img).ravel()
#             mean_val = np.mean(vals)
#             # contrast = xmcd_peak_separation(img)
#             # snr = snr_db(img)
#             lo, hi = np.percentile(vals, [0.1, 99.9])   # adjust as needed


#             hcolor = hist_colors.get(cond_name) if hist_colors else None
#             ax.hist(
#                 vals,
#                 bins=bins,
#                 alpha=alpha,
#                 color=hcolor,
#                 range=(lo, hi)
#                 #label=f"{cond_name}\ncontrast = {contrast:.3f}, snr = {snr:.3f}"
#             )

#             if show_mean_lines:
#                 mcolor = mean_line_colors.get(cond_name) if mean_line_colors else None
#                 ax.axvline(
#                     mean_val,
#                     color=mcolor,
#                     linestyle='-',
#                     linewidth=1.5,
#                     #label=f"mean = {mean_val:.2f}"
#                 )

#         ax.set_title(group)
#         ax.set_xlabel("Intensity")
#         ax.set_ylabel("Pixel Count")
#         ax.legend(title=group, fontsize=legend_fontsize)

#     if suptitle is not None:
#         fig.suptitle(suptitle)

#     plt.tight_layout()
#     plt.show()




def plot_hist_compare_dict(
    compare_dict,
    bins=100,
    alpha=0.5,
    figsize=(12, 12),
    show_mean_lines=True,
    mean_line_colors=None,
    hist_colors=None,
    suptitle=None,
    legend_fontsize=8,
    # robust/cropping options
    robust_range_pct=None,   # e.g. (1, 99)
    shared_range=True,
    report_clipped=False,
    # contrast peak detection options
    use_safe_peak_contrast=True,
    peak_prominence=None,
    peak_distance=None,
    contrast_hist_bins=256
):
    """
    compare_dict format:
    {
      "CPL": {"light off": arr, "light on": arr},
      "CPR": {"light off": arr, "light on": arr},
    }
    """
    group_names = list(compare_dict.keys())
    n_groups = len(group_names)

    # Build shared robust range if requested
    global_lohi = None
    if robust_range_pct is not None and shared_range:
        all_vals = []
        for g in group_names:
            for _, img in compare_dict[g].items():
                v = np.asarray(img).ravel()
                v = v[np.isfinite(v)]
                if v.size:
                    all_vals.append(v)
        if all_vals:
            all_concat = np.concatenate(all_vals)
            lo, hi = np.percentile(all_concat, robust_range_pct)
            global_lohi = (lo, hi)

    fig, axes = plt.subplots(1, n_groups, figsize=figsize, squeeze=False)
    axes = axes[0]

    for i, group in enumerate(group_names):
        ax = axes[i]
        condition_dict = compare_dict[group]

        # Per-group range if not shared
        group_lohi = None
        if robust_range_pct is not None and not shared_range:
            gvals = []
            for _, img in condition_dict.items():
                v = np.asarray(img).ravel()
                v = v[np.isfinite(v)]
                if v.size:
                    gvals.append(v)
            if gvals:
                gc = np.concatenate(gvals)
                lo, hi = np.percentile(gc, robust_range_pct)
                group_lohi = (lo, hi)

        hist_range = global_lohi if global_lohi is not None else group_lohi

        for cond_name, img in condition_dict.items():
            vals = np.asarray(img).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            mean_val = np.mean(vals)
            snr = snr_db(img)

            # Contrast logic
            if use_safe_peak_contrast:
                contrast, n_peaks, _ = _safe_peak_contrast(
                    vals,
                    bins=contrast_hist_bins,
                    hist_range=hist_range,
                    prominence=peak_prominence,
                    distance=peak_distance
                )
                if contrast is None:
                    contrast_txt = f"contrast = N/A (peaks found: {n_peaks})"
                else:
                    contrast_txt = f"contrast = {contrast:.3f}"
            else:
                # fallback to your existing method
                contrast = xmcd_peak_separation(img)
                contrast_txt = f"contrast = {contrast:.3f}"

            # Optional clipped counts for displayed histogram
            clip_txt = ""
            if report_clipped and hist_range is not None:
                lo, hi = hist_range
                n_low = int(np.sum(vals < lo))
                n_high = int(np.sum(vals > hi))
                #clip_txt = f"\nclipped low/high = {n_low}/{n_high}"

            hcolor = hist_colors.get(cond_name) if hist_colors else None
            ax.hist(
                vals,
                bins=bins,
                range=hist_range,
                alpha=alpha,
                color=hcolor,
                label=f"{cond_name}\n{contrast_txt}, snr = {snr:.3f}"
            )

            if show_mean_lines:
                mcolor = mean_line_colors.get(cond_name) if mean_line_colors else None
                ax.axvline(
                    mean_val,
                    color=mcolor,
                    linestyle='-',
                    linewidth=1.5,
                    label=f"mean = {mean_val:.2f}"
                )

        ax.set_title(group)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Pixel Count")
        ax.legend(title=group, fontsize=legend_fontsize)

    if suptitle is not None:
        fig.suptitle(suptitle)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------
# SECTION 12: Gamma Alpha alignment rotations
# -----------------------------------------------------------------------

#### Functions 
def preprocess_for_sift(img):
    """Convert any image dtype to uint8 grayscale for SIFT."""
    # Convert to grayscale if multichannel
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Handle complex arrays (e.g. from FFT)
    if np.iscomplexobj(img):
        img = np.abs(img)
    
    # Replace NaNs and Infs before normalization
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to 0–255 and convert to uint8
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:  # Avoid division by zero
        img = (img - img_min) / (img_max - img_min) * 255
    img = img.astype(np.uint8)

    return img

# def find_rotation_angle(img1, img2):
#     img1 = preprocess_for_sift(img1)
#     img2 = preprocess_for_sift(img2)

#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)

#     if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
#         print("Warning: Not enough keypoints found.")
#         return None

#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)

#     good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

#     if len(good_matches) < 4:
#         print(f"Warning: Only {len(good_matches)} good matches found — result may be unreliable.")
#         return None

#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#     M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

#     if M is None:
#         print("Warning: Could not estimate affine transform.")
#         return None

#     angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
#     return angle


def find_rotation_translation(img1, img2, ratio=0.75, ransac_thresh=3.0, min_inliers=2):
    img1 = preprocess_for_sift(img1)
    img2 = preprocess_for_sift(img2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("Warning: Not enough keypoints found.")
        return None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]

    if len(good_matches) < 4:
        print(f"Warning: Only {len(good_matches)} good matches found — result may be unreliable.")
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Get inliers from similarity model (rotation+scale+translation),
    # but we'll ignore scale and refit rigid transform using inliers.
    M_sim, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh
    )

    if M_sim is None or inliers is None:
        print("Warning: Could not estimate affine transform.")
        return None

    inlier_mask = inliers.ravel().astype(bool)
    if inlier_mask.sum() < min_inliers:
        print(f"Warning: Too few inliers ({inlier_mask.sum()}) to fit rigid transform.")
        return None

    src_in = src_pts[inlier_mask, 0, :]  # (N,2)
    dst_in = dst_pts[inlier_mask, 0, :]  # (N,2)

    # ---- Kabsch rigid transform (no scaling) ----
    c1 = src_in.mean(axis=0)
    c2 = dst_in.mean(axis=0)

    X = src_in - c1
    Y = dst_in - c2

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = c2 - R @ c1  # translation in pixel coords

    # ---- Angle convention fix ----
    # Your rotate_image() uses positive angles as CLOCKWISE.
    # From the rigid R (mapping src -> dst), the clockwise-positive angle is:
    angle_clockwise = -np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    # Full 2x3 rigid warp matrix to align img1 onto img2 coordinates
    M_rigid = np.array([
        [R[0, 0], R[0, 1], t[0]],
        [R[1, 0], R[1, 1], t[1]]
    ], dtype=np.float32)

    return angle_clockwise, (float(t[0]), float(t[1])), M_rigid, inlier_mask.sum()


def rotate_image_mask(img, angle_deg):
    """
    Rotate img by angle_deg (positive = clockwise, image/screen convention).
    Canvas is expanded to avoid clipping corners.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # cv2.getRotationMatrix2D is CCW-positive, so negate to match our convention
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, scale=1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Shift centre to the new canvas centre
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)
def to_display(img):
    """Normalise any dtype to float32 in [0, 1] for matplotlib."""
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img.astype(np.float32)


def compute_difference(img_a, img_b):
    """
    Signed difference (A - B) in float64.
    Centre-crops both images to the overlapping region if sizes differ.
    """
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)

    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h, w = min(ha, hb), min(wa, wb)

    ya, xa = (ha - h) // 2, (wa - w) // 2
    yb, xb = (hb - h) // 2, (wb - w) // 2

    return a[ya:ya+h, xa:xa+w] - b[yb:yb+h, xb:xb+w]