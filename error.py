
#################### Imports ########################
import os
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
#import dxchange
import time
import tifffile

from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar
from numpy.fft import fftn, fft2, fftshift
import tifffile



################## Resolution ########################
def resolution_at_threshold_bad(freq, fsc, threshold):
    ok = ~np.isnan(fsc)
    f, c = freq[ok], fsc[ok]
    below = np.where(c < threshold)[0]
    if len(below) == 0:
        return np.nan                      # never crosses: resolution beyond Nyquist
    i = below[0]
    if i == 0:
        return np.nan                      # already below threshold at lowest shell
    # linear interpolation between shells i-1 and i
    f_cross = f[i-1] + (threshold - c[i-1]) * (f[i] - f[i-1]) / (c[i] - c[i-1])
    return 1.0 / f_cross



def halfbit_threshold(n):
    """Half-bit threshold (van Heel & Schatz 2005) for n independent Fourier voxels per shell."""
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = 1.0 / np.sqrt(n)
        return (0.2071 + 1.9102 * s) / (1.2071 + 0.9102 * s)


def resolution_at_threshold(freq, fsc, threshold):
    """First downward crossing of fsc through threshold (scalar or per-shell array),
    linearly interpolated. Returns resolution = 1/f_cross, or NaN if no crossing."""
    thr = np.broadcast_to(np.asarray(threshold, dtype=float), fsc.shape)
    d = fsc - thr
    ok = ~np.isnan(d)
    f, d = freq[ok], d[ok]
    for i in range(1, len(d)):
        if d[i - 1] >= 0 and d[i] < 0:
            fc = f[i - 1] + d[i - 1] * (f[i] - f[i - 1]) / (d[i - 1] - d[i])
            return 1.0 / fc
    return np.nan



def fourier_shell_correlation_bad(vol1, vol2, voxel_size, n_shells=None, plot=True):

    print(f'n_shells preset = {n_shells}')
    vol1 = vol1.astype(np.float32); vol2 = vol2.astype(np.float32)
    assert vol1.shape == vol2.shape
    nz, ny, nx = vol1.shape

    # FFT of both bolumes
    F1 = np.fft.rfftn(vol1)
    F2 = np.fft.rfftn(vol2)

    # spatial frequency of x,y,z at voxel size
    kz = np.fft.fftfreq(nz, d=voxel_size)      
    ky = np.fft.fftfreq(ny, d=voxel_size)      
    kx = np.fft.rfftfreq(nx, d=voxel_size)     
    # spatial freq mag. 
    K = np.sqrt(kz[:, None, None]**2 + ky[None, :, None]**2 + kx[None, None, :]**2)

    # Nyquist
    k_max = 0.5 / voxel_size        

    if n_shells is None:
        n_shells = min(nz, ny, nx) // 2        # one shell per frequency step

    ## This is where the problem is because min nz //2  = 2.5 
    # n_shells = min(n_shells, min(nz, ny, nx) // 2)
    # print(f'n_shells = {n_shells}')

    edges = np.linspace(0, k_max, n_shells + 1)
    freq = 0.5 * (edges[:-1] + edges[1:])

    idx = (np.digitize(K.ravel(), edges) - 1)
    valid = (idx >= 0) & (idx < n_shells)
    idx = idx[valid]

    cross = (F1 * np.conj(F2)).real.ravel()[valid]
    p1 = (np.abs(F1)**2).ravel()[valid]
    p2 = (np.abs(F2)**2).ravel()[valid]

    num  = np.bincount(idx, weights=cross, minlength=n_shells)
    den1 = np.bincount(idx, weights=p1,    minlength=n_shells)
    den2 = np.bincount(idx, weights=p2,    minlength=n_shells)

    with np.errstate(invalid='ignore', divide='ignore'):
        fsc = num / np.sqrt(den1 * den2)
    fsc[(den1 == 0) | (den2 == 0)] = np.nan    # empty shells -> NaN, not 0

    res_05   = resolution_at_threshold_bad(freq, fsc, 0.5)
    res_0143 = resolution_at_threshold_bad(freq, fsc, 0.143)
    print(f"Resolution (FSC=0.5):   {res_05:.2f}")
    print(f"Resolution (FSC=0.143): {res_0143:.2f}")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(freq, fsc, lw=2)
        plt.axhline(0.5, ls='--', color='k', label=f'0.5, res = {res_05:.2f}')
        plt.axhline(0.143, ls=':', color='k', label=f'0.143, res = {res_0143:.2f}')
        if not np.isnan(res_05):   plt.axvline(1/res_05, ls='--', color='k')
        if not np.isnan(res_0143): plt.axvline(1/res_0143, ls=':', color='k')
        plt.xlabel('Spatial frequency (1 / nm)'); plt.ylabel('FSC')
        plt.ylim(-0.1, 1.05); plt.legend(); plt.tight_layout(); plt.show()

    return freq, fsc, res_05, res_0143


def frc_stack_w_window(v1, v2, voxel_size, plot=True):
    v1 = v1.astype(np.float64); v2 = v2.astype(np.float64)
    nz, ny, nx = v1.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    F1 = np.fft.fft2((v1 - v1.mean(axis=(1, 2), keepdims=True)) * win)
    F2 = np.fft.fft2((v2 - v2.mean(axis=(1, 2), keepdims=True)) * win)

    ky = np.fft.fftfreq(ny, d=voxel_size)
    kx = np.fft.fftfreq(nx, d=voxel_size)
    K = np.hypot(ky[:, None], kx[None, :])

    n_shells = min(ny, nx) // 2
    edges = np.linspace(0, 0.5 / voxel_size, n_shells + 1)
    freq = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(K.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < n_shells)
    idx = idx[ok]

    num  = (F1 * F2.conj()).real.sum(axis=0).ravel()[ok]
    den1 = (np.abs(F1)**2).sum(axis=0).ravel()[ok]
    den2 = (np.abs(F2)**2).sum(axis=0).ravel()[ok]
    num  = np.bincount(idx, num,  n_shells)
    den1 = np.bincount(idx, den1, n_shells)
    den2 = np.bincount(idx, den2, n_shells)

    with np.errstate(invalid='ignore', divide='ignore'):
        fsc = num / np.sqrt(den1 * den2)
    fsc[(den1 == 0) | (den2 == 0)] = np.nan

    res05, res0143 = (resolution_at_threshold(freq, fsc, t) for t in (0.5, 0.143))
    print(f"Resolution (0.5): {res05:.2f}   (0.143): {res0143:.2f}")
    if plot:
        plt.plot(freq, fsc); plt.axhline(0.5, ls='--', c='k'); plt.axhline(0.143, ls=':', c='k')
        plt.xlabel('Spatial frequency (1/nm)'); plt.ylabel('FRC'); plt.ylim(-0.1, 1.05); plt.show()
    return freq, fsc, res05, res0143


def frc_stack_no_window(v1, v2, voxel_size, plot=True, sample = 'call sample', title = 'call sample'):
    v1 = v1.astype(np.float64); v2 = v2.astype(np.float64)
    nz, ny, nx = v1.shape

    # One FFT per z slice 
    F1 = np.fft.fft2(v1)
    print(F1.shape)
    F2 = np.fft.fft2(v2)

    ky = np.fft.fftfreq(ny, d=voxel_size)
    kx = np.fft.fftfreq(nx, d=voxel_size)

    K = np.sqrt(ky[:, None]**2 + kx[None, :]**2)

    k_max = 0.5 / voxel_size

    n_shells = min(ny, nx) // 2
    print(f'no window number of shells = {n_shells}')
    edges = np.linspace(0, k_max, n_shells + 1)
    freq = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(K.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < n_shells)
    idx = idx[ok]

    # In this step, sum over z for F1 * F2 
    num  = (F1 * F2.conj()).real.sum(axis=0).ravel()[ok]
    den1 = (np.abs(F1)**2).sum(axis=0).ravel()[ok]
    den2 = (np.abs(F2)**2).sum(axis=0).ravel()[ok]
    num  = np.bincount(idx, num,  n_shells)
    den1 = np.bincount(idx, den1, n_shells)
    den2 = np.bincount(idx, den2, n_shells)

    with np.errstate(invalid='ignore', divide='ignore'):
        fsc = num / np.sqrt(den1 * den2)
    fsc[(den1 == 0) | (den2 == 0)] = np.nan


    # Number of voxels per ring * nz 
    n = np.bincount(idx, minlength=n_shells).astype(float) #* nz

    hb = halfbit_threshold(n)
    res = {
        "0.5":      resolution_at_threshold(freq, fsc, 0.5),
        "0.143":    resolution_at_threshold(freq, fsc, 0.143),
        "half-bit": resolution_at_threshold(freq, fsc, hb),
    }

    if plot:
        # plt.figure(figsize=(10, 6))          # add this line first

        plt.plot(freq, fsc, lw=2, label=f'FRC {title}')
        plt.plot(freq, hb, c="C3", lw=1.5, label=f"half-bit = {res["half-bit"]:.1f} nm")
        plt.axhline(0.5, ls="--", c="k", lw=1, label = f'0.5 = {res["0.5"]:.1f} nm')
        plt.axhline(0.143, ls=":", c="k", lw=1, label = f'0.143 = {res["0.143"]:.1f} nm')
        for (k, v), c in zip(res.items(), ['k', 'k', 'C3']):
            if not np.isnan(v):
                plt.axvline(1 / v, c=c, lw=0.8, ls='--' if k == '0.5' else ':')


        plt.xlabel("Spatial frequency (1 / nm)"); plt.ylabel(f' FRC {title}')
        plt.ylim(-0.1, 1.05); plt.legend(loc="upper right")

        plt.title(f'{title} \n {sample}' )
        


    return freq, fsc, n, res



def _local_mean(x, size):
    if size <= 1:
        return x
    try:
        from scipy.ndimage import uniform_filter
        return uniform_filter(x, size=size, mode="wrap")
    except ImportError:
        return x
    
def plot_fourier_planes(a, b, voxel_size, res=None, kz_index=0, smooth=7):
    """What the FSC sees: |F|^2 of each (preprocessed) volume and the local phase coherence,
    on the plane kz = fftfreq(nz)[kz_index]. Full plane (kx and ky both +/-), origin centred."""
    nz, ny, nx = a.shape
    Fa = np.fft.fft2(np.fft.fft(a, axis=0)[kz_index])
    Fb = np.fft.fft2(np.fft.fft(b, axis=0)[kz_index])
    Fa, Fb = np.fft.fftshift(Fa), np.fft.fftshift(Fb)
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=voxel_size))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=voxel_size))
    dk_x, dk_y = kx[1] - kx[0], ky[1] - ky[0]
    extent = [kx[0] - dk_x / 2, kx[-1] + dk_x / 2, ky[0] - dk_y / 2, ky[-1] + dk_y / 2]

    P1, P2 = np.abs(Fa) ** 2, np.abs(Fb) ** 2
    num = _local_mean((Fa * Fb.conj()).real, smooth)
    den = np.sqrt(_local_mean(P1, smooth) * _local_mean(P2, smooth))
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = num / den

    vmax = np.log10(max(P1.max(), P2.max()))
    vmin = vmax - 8
    kz_val = np.fft.fftfreq(nz, d=voxel_size)[kz_index]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for axi, P, t in [(ax[0], P1, "Volume 1"), (ax[1], P2, "Volume 2")]:
        im = axi.imshow(np.log10(P + 1e-30), extent=extent, origin="lower",
                        cmap="viridis", vmin=vmin, vmax=vmax)
        axi.set_title(f"{t}: log10 |F|²   (kz = {kz_val:.3g})")
        plt.colorbar(im, ax=axi, fraction=0.046)
    im = ax[2].imshow(coh, extent=extent, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    ax[2].set_title(f"local coherence (box {smooth}×{smooth})")
    plt.colorbar(im, ax=ax[2], fraction=0.046)

    th = np.linspace(0, 2 * np.pi, 400)
    for axi in ax:
        axi.set_xlabel(r"$k_x$ (1/nm)"); axi.set_ylabel(r"$k_y$ (1/nm)")
        axi.add_patch(plt.Rectangle((extent[0], extent[2]), extent[1] - extent[0],
                                    extent[3] - extent[2], fill=False, ec="none"))
    if res is not None:
        for key, ls, c in [("0.5", "--", "k"), ("0.143", ":", "k"), ("half-bit", "-.", "m")]:
            r = res.get(key, np.nan)
            if not np.isnan(r):
                ax[2].plot(np.cos(th) / r, np.sin(th) / r, ls=ls, c=c, lw=1, label=key)
        ax[2].legend(loc="upper right", fontsize=8)
    ax[2].set_xlim(extent[0], extent[1]); ax[2].set_ylim(extent[2], extent[3])
    plt.tight_layout(); plt.show()


def _tukey(n, alpha):
    """Tapered-cosine window: flat in the middle, cosine taper over a fraction alpha of the length."""
    if alpha <= 0:
        return np.ones(n)
    x = np.linspace(0, 1, n)
    w = np.ones(n)
    L = alpha / 2
    lo, hi = x < L, x > 1 - L
    w[lo] = 0.5 * (1 + np.cos(np.pi * (x[lo] / L - 1)))
    w[hi] = 0.5 * (1 + np.cos(np.pi * ((x[hi] - 1 + L) / L)))
    return w


def fourier_shell_correlation(vol1, vol2, voxel_size, n_shells=None,
                            window_xy=True, taper_z=False, tuckey_alpha = 0.5, 
                              plot=True, plot_fourier=True, sample = '452', title = 'Please indicate vector'):
    """3D FSC between two volumes (nz, ny, nx).

    Preprocessing (this is what fixes the inflated FSC):
      window_xy     : Hann window in y and x (suppresses edge-leakage streaks)
      taper_z       : gentle taper along z, [.25,.75,1,.75,.25] for nz=5 (optional)
    Returns freq, fsc, n_per_shell, resolutions(dict with 0.5, 0.143, halfbit).
    """
    a = vol1.astype(np.float64).copy()
    b = vol2.astype(np.float64).copy()
    assert a.shape == b.shape, "Volumes must have same shape"
    nz, ny, nx = a.shape


    w = np.ones((nz, ny, nx))
    if window_xy:
        w = w * np.outer(_tukey(ny, tuckey_alpha), _tukey(nx, tuckey_alpha))[None]
        #w = w * np.outer(np.hanning(ny), np.hanning(nx))[None]
    if taper_z and nz > 2:
        w = w * np.hanning(nz + 2)[1:-1][:, None, None]
    a *= w; b *= w

    F1 = np.fft.rfftn(a)
    F2 = np.fft.rfftn(b)

    # FULL axes for z and y (only x is halved by rfftn)
    kz = np.fft.fftfreq(nz, d=voxel_size)
    ky = np.fft.fftfreq(ny, d=voxel_size)
    kx = np.fft.rfftfreq(nx, d=voxel_size)
    K = np.sqrt(kz[:, None, None]**2 + ky[None, :, None]**2 + kx[None, None, :]**2)

    k_max = 0.5 / voxel_size                      # in-plane Nyquist, 
    if n_shells is None:
        n_shells = min(ny, nx) // 2
    edges = np.linspace(0, k_max, n_shells + 1)
    freq = 0.5 * (edges[:-1] + edges[1:])

    idx = np.digitize(K.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < n_shells)
    idx = idx[ok]

    cross = (F1 * np.conj(F2)).real.ravel()[ok]
    num  = np.bincount(idx, weights=cross, minlength=n_shells)
    p1 = (np.abs(F1)**2).ravel()[ok]
    p2 = (np.abs(F2)**2).ravel()[ok]

    den1 = np.bincount(idx, weights=p1,    minlength=n_shells)
    den2 = np.bincount(idx, weights=p2,    minlength=n_shells)

    n = np.bincount(idx, minlength=n_shells)

    with np.errstate(invalid="ignore", divide="ignore"):
        fsc = num / np.sqrt(den1 * den2)
    fsc[(den1 == 0) | (den2 == 0)] = np.nan          # empty shells -> NaN, not 0

    hb = halfbit_threshold(n)
    res = {
        "0.5": resolution_at_threshold(freq, fsc, 0.5),
        "0.143": resolution_at_threshold(freq, fsc, 0.143),
        "half-bit": resolution_at_threshold(freq, fsc, hb),
    }

    for k, v in res.items():
        print(f"Resolution (FSC={k}): {v:.2f}")

    if plot:
        plt.figure(figsize=(6.5, 4.2))
        plt.plot(freq, fsc, lw=2) #, label="FSC")
        plt.plot(freq, hb, c="C3", lw=1.5, label=f"half-bit = {res["half-bit"]:.1f} nm")
        plt.axhline(0.5, ls="--", c="k", lw=1, label = f'0.5 = {res["0.5"]:.1f} nm')
        plt.axhline(0.143, ls=":", c="k", lw=1, label = f'0.143 = {res["0.143"]:.1f} nm')
        for (k, v), c in zip(res.items(), ["k", "k", "C3"]):
            if not np.isnan(v):
                plt.axvline(1 / v, c=c, lw=0.8, ls="--" if k == "0.5" else ":")
        # txt = "\n".join(f"{k}: {v:.1f}" for k, v in res.items())
        # plt.text(0.97, 0.95, txt, transform=plt.gca().transAxes, ha="right", va="top")
        plt.xlabel("Spatial frequency (1 / nm)"); plt.ylabel(title)
        plt.ylim(-0.1, 1.05); plt.legend(loc="lower left")

        plt.title(f'{title} \n {sample}' )

        plt.tight_layout(); plt.show()

    if plot_fourier:
        plot_fourier_planes(a, b, voxel_size, res=res, kz_index=0, smooth=7)


    return freq, fsc, n, res




################## Simulation GT error ########################