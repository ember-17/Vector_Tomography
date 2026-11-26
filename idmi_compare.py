"""
Discrete vs. continuum interfacial DMI on a 3D magnetization volume.

Array convention: Mx, My, Mz have shape (Nx, Ny, Nz), indexed [x, y, z].
Each z-layer is treated independently (interfacial DMI uses in-plane bonds only).
Change the axis arguments below if your layout differs.

Discrete form (each bond counted once):
    E = -sum_<ij> D_ij . (m_i x m_j),   D_ij = D * (z_hat x u_ij)
      x-bond (u = +x):  E_b = -D * (m_iz*m_jx - m_ix*m_jz)
      y-bond (u = +y):  E_b = +D * (m_iy*m_jz - m_iz*m_jy)

Continuum form (same sign convention as your code):
    e = D * [ mx*dmz/dx - mz*dmx/dx + my*dmz/dy - mz*dmy/dy ]
"""
import numpy as np


def _prepare(Mx, My, Mz, normalize):
    """normalize: True -> unit vectors, a number -> divide by Ms, False/None -> as is."""
    m = [np.asarray(a, dtype=float) for a in (Mx, My, Mz)]
    if normalize is True:
        norm = np.sqrt(sum(a**2 for a in m))
        norm = np.where(norm == 0, 1.0, norm)
        m = [a / norm for a in m]
    elif normalize not in (False, None):
        m = [a / float(normalize) for a in m]
    return m


def dmi_discrete(Mx, My, Mz, D=1.0, normalize=True, periodic=(False, False),
                 axes=(0, 1)):
    """
    Discrete iDMI energy from nearest-neighbour bonds in the film plane.

    Returns a dict with:
      'bond_x', 'bond_y' : energy of the bond (i, i+1) stored at index i
                           (zero where the bond does not exist for open boundaries)
      'site'             : per-cell energy, each bond split half/half between
                           its two cells (sums to the total)
      'total'            : sum of all bond energies
    """
    mx, my, mz = _prepare(Mx, My, Mz, normalize)
    ax, ay = axes

    def nb(a, axis):  # value of the +1 neighbour along axis (periodic wrap)
        return np.roll(a, -1, axis=axis)

    # x-bonds: D_ij = D * y_hat
    e_bx = -D * (mz * nb(mx, ax) - mx * nb(mz, ax))
    # y-bonds: D_ij = -D * x_hat
    e_by = D * (my * nb(mz, ay) - mz * nb(my, ay))

    # remove the wrapped bond for open boundaries
    if not periodic[0]:
        idx = [slice(None)] * e_bx.ndim
        idx[ax] = -1
        e_bx[tuple(idx)] = 0.0
    if not periodic[1]:
        idx = [slice(None)] * e_by.ndim
        idx[ay] = -1
        e_by[tuple(idx)] = 0.0

    site = 0.5 * (e_bx + np.roll(e_bx, 1, axis=ax)
                  + e_by + np.roll(e_by, 1, axis=ay))

    return {"bond_x": e_bx, "bond_y": e_by, "site": site,
            "total": e_bx.sum() + e_by.sum()}


def dmi_continuum(Mx, My, Mz, D=1.0, normalize=True, periodic=(False, False),
                  spacing=(1.0, 1.0), axes=(0, 1)):
    """
    Continuum iDMI energy density (your L_xz_x + L_yz_y) with central differences.
    With spacing=(1, 1) derivatives are in grid units, which makes it directly
    comparable to dmi_discrete using the same D.
    """
    mx, my, mz = _prepare(Mx, My, Mz, normalize)
    ax, ay = axes

    def d(a, axis, h, per):
        if per:
            return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / (2 * h)
        return np.gradient(a, h, axis=axis)  # one-sided at the edges

    dmz_dx, dmx_dx = d(mz, ax, spacing[0], periodic[0]), d(mx, ax, spacing[0], periodic[0])
    dmz_dy, dmy_dy = d(mz, ay, spacing[1], periodic[1]), d(my, ay, spacing[1], periodic[1])

    L_xz_x = mx * dmz_dx - mz * dmx_dx
    L_yz_y = my * dmz_dy - mz * dmy_dy
    return D * (L_xz_x + L_yz_y)


def compare(Mx, My, Mz, D=1.0, normalize=True, periodic=(False, False), axes=(0, 1)):
    """Compare per-cell and total energies of the two forms (grid units)."""
    disc = dmi_discrete(Mx, My, Mz, D, normalize, periodic, axes)
    cont = dmi_continuum(Mx, My, Mz, D, normalize, periodic, (1.0, 1.0), axes)

    # Away from open edges the two per-cell maps should agree to round-off:
    # splitting each bond half/half reproduces the central difference exactly.
    sl = [slice(None)] * disc["site"].ndim
    for ax, per in zip(axes, periodic):
        if not per:
            sl[ax] = slice(1, -1)
    sl = tuple(sl)

    diff = disc["site"][sl] - cont[sl]
    out = {
        "total_discrete": disc["total"],
        "total_continuum": cont.sum(),
        "max_abs_diff_interior": np.abs(diff).max(),
        "rel_rms_diff_interior": np.sqrt((diff**2).mean())
                                 / (np.sqrt((cont[sl]**2).mean()) + 1e-30),
    }
    return out, disc, cont


if __name__ == "__main__":
    # Smooth Neel-skyrmion-like test texture, 64 x 64 x 3 layers
    N, nz = 64, 3
    x, y = np.meshgrid(np.arange(N) - N / 2, np.arange(N) - N / 2, indexing="ij")
    r = np.hypot(x, y) + 1e-9
    theta = 2 * np.arctan(np.exp((r - 12) / 4))   # profile
    mx = np.sin(theta) * x / r
    my = np.sin(theta) * y / r
    mz = -np.cos(theta)
    Mx, My, Mz = [np.repeat(a[:, :, None], nz, axis=2) for a in (mx, my, mz)]

    for per in [(False, False), (True, True)]:
        res, _, _ = compare(Mx, My, Mz, D=1.0, periodic=per)
        print("periodic =", per)
        for k, v in res.items():
            print(f"  {k}: {v:.6e}")
