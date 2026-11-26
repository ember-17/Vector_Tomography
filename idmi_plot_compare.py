"""
Visual comparison of discrete vs. continuum iDMI using your plot_energy_maps_fancy.

Requires idmi_compare.py (previous file) in the same folder.
Your volumes are indexed [z, y, x], so the x axis is 2 and the y axis is 1.
"""
import numpy as np
from idmi_compare import dmi_discrete, dmi_continuum

AXES_ZYX = (2, 1)  # (x axis, y axis) for arrays shaped (nz, ny, nx)


def build_idmi_maps(Mx, My, Mz, D=1.0, normalize=True, periodic=(False, False),
                    continuum=None, axes=AXES_ZYX):
    """
    Returns a dict ready for plot_energy_maps_fancy.

    continuum : optional array (nz, ny, nx). Pass your own Jacobian-based result
                (the `idmi` from your code) to test it against the discrete form.
                If None, the central-difference continuum form is computed here.
    D         : scalar, or an array broadcastable to (nz, ny, nx).
    """
    disc = dmi_discrete(Mx, My, Mz, D, normalize, periodic, axes)["site"]
    if continuum is None:
        continuum = dmi_continuum(Mx, My, Mz, D, normalize, periodic,
                                  spacing=(1.0, 1.0), axes=axes)
    continuum = np.asarray(continuum, dtype=float)
    assert continuum.shape == disc.shape, (continuum.shape, disc.shape)

    return {
        "shape": disc.shape,                 # (nz, ny, nx)
        "iDMI_continuum": continuum,
        "iDMI_discrete": disc,
        "iDMI_diff": disc - continuum,       # discrete - continuum
    }


def plot_idmi_comparison(maps, plot_fn, axis="z", index=None,
                         cmap="turbo", diff_cmap="RdBu_r"):
    """
    Calls plot_fn (your plot_energy_maps_fancy) three times:
    continuum and discrete share the same color limits; the difference
    map uses a symmetric scale around zero.
    Returns {key: figure}.
    """
    lo = min(maps["iDMI_continuum"].min(), maps["iDMI_discrete"].min())
    hi = max(maps["iDMI_continuum"].max(), maps["iDMI_discrete"].max())
    m = max(np.abs(maps["iDMI_diff"]).max(), 1e-30)

    figs = {}
    for key in ("iDMI_continuum", "iDMI_discrete"):
        figs[key] = plot_fn(maps, key, axis=axis, index=index,
                            cmap=cmap, vmin=lo, vmax=hi)
    figs["iDMI_diff"] = plot_fn(maps, "iDMI_diff", axis=axis, index=index,
                                cmap=diff_cmap, vmin=-m, vmax=m)
    return figs


if __name__ == "__main__":
    # Logic test with a stub in place of the real plot function
    nz, ny, nx = 4, 48, 64
    X, Y = np.meshgrid(np.arange(nx) - nx / 2, np.arange(ny) - ny / 2)
    r = np.hypot(X, Y) + 1e-9
    th = 2 * np.arctan(np.exp((r - 10) / 4))
    mx, my, mz = np.sin(th) * X / r, np.sin(th) * Y / r, -np.cos(th)
    Mx, My, Mz = [np.repeat(a[None], nz, axis=0) for a in (mx, my, mz)]

    calls = []
    def stub(data, key, axis, index, cmap, vmin, vmax):
        calls.append((key, cmap, round(float(vmin), 4), round(float(vmax), 4)))
        assert data["shape"] == data[key].shape
        return key

    maps = build_idmi_maps(Mx, My, Mz, D=1.0, periodic=(True, True))
    plot_idmi_comparison(maps, stub, index=2)
    for c in calls:
        print(c)
