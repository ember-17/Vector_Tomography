"""
vti_magnetic_import.py

Import VTK vector-field reconstructions of magnetic domains / domain walls
(any of: XML ImageData .vti, XML RectilinearGrid .vtr, or legacy-format
.vtk containing an image/structured-points or rectilinear-grid dataset),
and store each sample in a dictionary of numpy arrays suitable for
downstream physical-quantity calculations (magnitude, orientation angles,
divergence "charge", curl, etc.).

Format is auto-detected from file CONTENT, not the extension, since legacy
VTK files are very often saved with a .vtk (or even .vti) extension:
  - XML VTK    : file starts with '<?xml' or '<VTKFile'
  - Legacy VTK : file starts with '# vtk DataFile Version...'
  - gzip       : raises with instructions to decompress first

.vti/ImageData = uniform grid, defined by a single origin + spacing.
.vtr/RectilinearGrid = grid defined by three independent 1D coordinate
    arrays (X, Y, Z), which may be non-uniformly spaced along each axis.
Everything downstream (dict layout, array names) is identical either way.

Requires: vtk, numpy   (pip install vtk numpy)

Usage:
    from vti_magnetic_import import load_vti_sample, inspect_vti

    inspect_vti("sample1.vtk")                 # see what's inside first
    samples = {}
    samples["sample1"] = load_vti_sample("sample1.vtk", vector_array="Magnetization")
"""

import os
import numpy as np
import vtk
from vtk.util import numpy_support


# --------------------------------------------------------------------------
# 0. Format detection + reader selection.
#    Handles VTK's two unrelated file "families" that both commonly end up
#    with .vtk/.vti/.vtr extensions:
#      - XML VTK  (starts with '<?xml' / '<VTKFile', e.g. real .vti/.vtr)
#      - Legacy VTK (starts with '# vtk DataFile Version...', ASCII or binary)
#    Detection is done by sniffing file content, not trusting the extension,
#    since legacy files are very often saved with a .vtk (or even .vti)
#    extension by various tools.
# --------------------------------------------------------------------------
def _sniff_format(path):
    with open(path, "rb") as f:
        head = f.read(256)
    if not head:
        raise ValueError(f"'{path}' is empty (0 bytes).")
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    stripped = head.lstrip()
    if stripped.startswith((b"<?xml", b"<VTKFile")):
        return "xml"
    if stripped.startswith(b"# vtk DataFile"):
        return "legacy"
    raise ValueError(
        f"'{path}' doesn't look like an XML VTK or legacy VTK file. "
        f"First bytes: {head[:80]!r}"
    )


def _read_grid(path):
    """
    Return (vtk_dataset, kind) where kind is 'image' or 'rectilinear'.
    Auto-detects XML vs legacy VTK format from file content.
    """
    fmt = _sniff_format(path)

    if fmt == "gzip":
        raise ValueError(
            f"'{path}' is gzip-compressed. Decompress it first, e.g.:\n"
            f"    import gzip, shutil\n"
            f"    with gzip.open(r'{path}', 'rb') as f_in, "
            f"open('{path}.decompressed', 'wb') as f_out:\n"
            f"        shutil.copyfileobj(f_in, f_out)\n"
            f"then load the decompressed file."
        )

    if fmt == "xml":
        ext = os.path.splitext(path)[1].lower()
        if ext == ".vtr":
            reader = vtk.vtkXMLRectilinearGridReader()
            kind = "rectilinear"
        else:
            # default to ImageData reader for .vti and anything unrecognized;
            # if it's actually XML RectilinearGrid content this will simply
            # fail to parse — rename to .vtr or pass through explicitly.
            reader = vtk.vtkXMLImageDataReader()
            kind = "image"
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput(), kind

    # fmt == "legacy"
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(path)
    reader.Update()
    dataset = reader.GetOutput()
    class_name = dataset.GetClassName()
    if class_name in ("vtkImageData", "vtkStructuredPoints"):
        kind = "image"
    elif class_name == "vtkRectilinearGrid":
        kind = "rectilinear"
    else:
        raise ValueError(
            f"'{path}' is a legacy VTK file but contains a '{class_name}' "
            f"dataset, which this loader doesn't handle (expects "
            f"ImageData/StructuredPoints or RectilinearGrid)."
        )
    return dataset, kind


def _get_coords(dataset, kind):
    """
    Return 1D coordinate arrays (x, y, z) in physical units, and the point
    dimensions (nx, ny, nz), regardless of whether the source was .vti or .vtr.
    """
    dims = dataset.GetDimensions()
    nx, ny, nz = dims
    if kind == "image":
        origin = dataset.GetOrigin()
        spacing = dataset.GetSpacing()
        x = origin[0] + spacing[0] * np.arange(nx)
        y = origin[1] + spacing[1] * np.arange(ny)
        z = origin[2] + spacing[2] * np.arange(nz)
    else:  # rectilinear
        x = numpy_support.vtk_to_numpy(dataset.GetXCoordinates())
        y = numpy_support.vtk_to_numpy(dataset.GetYCoordinates())
        z = numpy_support.vtk_to_numpy(dataset.GetZCoordinates())
    return x, y, z, dims


# --------------------------------------------------------------------------
# 1. Inspection helper — run this FIRST on a new file to see what arrays,
#    dimensions, spacing etc. it actually contains before assuming names.
# --------------------------------------------------------------------------
def inspect_vti(path):
    """Print a summary of arrays / geometry contained in a .vti or .vtr file."""
    dataset, kind = _read_grid(path)
    x, y, z, dims = _get_coords(dataset, kind)

    print(f"File: {path}  (kind: {kind})")
    print(f"  Dimensions (nx, ny, nz): {dims}")
    if kind == "image":
        print(f"  Spacing:                 {dataset.GetSpacing()}")
        print(f"  Origin:                  {dataset.GetOrigin()}")
    else:
        def _span(a):
            diffs = np.diff(a)
            uniform = np.allclose(diffs, diffs[0]) if len(diffs) else True
            return f"[{a.min():.4g}, {a.max():.4g}], " \
                   f"{'uniform' if uniform else 'NON-uniform'} spacing"
        print(f"  X coords: {_span(x)}")
        print(f"  Y coords: {_span(y)}")
        print(f"  Z coords: {_span(z)}")

    for label, data in [("Point data", dataset.GetPointData()),
                         ("Cell data", dataset.GetCellData())]:
        n = data.GetNumberOfArrays()
        print(f"  {label} ({n} arrays):")
        for i in range(n):
            arr = data.GetArray(i)
            print(f"    - '{arr.GetName()}': "
                  f"{arr.GetNumberOfComponents()} component(s), "
                  f"{arr.GetNumberOfTuples()} tuples, "
                  f"dtype={arr.GetDataTypeAsString()}")
    return dataset


# --------------------------------------------------------------------------
# 2. Core loader — reads one .vti file into a flat dictionary of
#    physically-meaningful numpy arrays, all reshaped to (nz, ny, nx)
#    so array indexing matches [z, y, x] like the coordinate grids.
# --------------------------------------------------------------------------
def load_vti_sample(path, vector_array=None, scalar_arrays=None,
                     use_cell_data=False, sample_name=None):
    """
    Load a single .vti file into a dictionary.

    Parameters
    ----------
    path : str
        Path to the .vti file.
    vector_array : str or None
        Name of the 3-component vector array to treat as the magnetization
        (or displacement) field, e.g. "Magnetization", "M", "vectors".
        If None, the function will auto-pick the first 3-component array
        it finds (and tell you which one it picked).
    scalar_arrays : list[str] or None
        Names of any additional scalar arrays to pull in as-is
        (e.g. a confidence/quality map, a mask, reconstructed density).
    use_cell_data : bool
        Set True if your arrays live on cell data rather than point data
        (inspect_vti will tell you which).
    sample_name : str or None
        Optional label stored inside the dict (defaults to the filename).

    Returns
    -------
    dict with keys:
        'name'        : sample name
        'path'        : source file path
        'dims'        : (nx, ny, nz) grid dimensions
        'spacing'     : (dx, dy, dz) voxel size
        'origin'      : (x0, y0, z0)
        'shape'       : (nz, ny, nx)  -- numpy array shape convention used below
        'X','Y','Z'   : coordinate grids, shape (nz, ny, nx), physical units
        'Mx','My','Mz': vector components, shape (nz, ny, nx)
        'M_vec'       : raw vector array, shape (nz, ny, nx, 3)
        'magnitude'   : |M|, shape (nz, ny, nx)
        'mask'        : boolean array, True where magnitude > 0 (valid voxels)
        plus any requested scalar_arrays under their own array name.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    img, kind = _read_grid(path)
    x, y, z, dims = _get_coords(img, kind)   # x,y,z are 1D physical-unit coord arrays
    nx, ny, nz = dims

    data_source = img.GetCellData() if use_cell_data else img.GetPointData()
    if use_cell_data:
        # cell data grid is (nx-1, ny-1, nz-1); use cell-center coordinates
        x = 0.5 * (x[:-1] + x[1:]) if len(x) > 1 else x
        y = 0.5 * (y[:-1] + y[1:]) if len(y) > 1 else y
        z = 0.5 * (z[:-1] + z[1:]) if len(z) > 1 else z
        nx, ny, nz = max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1)

    # --- pick / validate the vector array -------------------------------
    if vector_array is None:
        for i in range(data_source.GetNumberOfArrays()):
            arr = data_source.GetArray(i)
            if arr.GetNumberOfComponents() == 3:
                vector_array = arr.GetName()
                print(f"[load_vti_sample] No vector_array given — "
                      f"auto-selected '{vector_array}'")
                break
        if vector_array is None:
            raise ValueError(
                "No 3-component array found. Run inspect_vti(path) and "
                "pass scalar_arrays=[...] / vector_array explicitly."
            )

    vtk_vec = data_source.GetArray(vector_array)
    if vtk_vec is None:
        raise KeyError(
            f"Array '{vector_array}' not found. "
            f"Run inspect_vti('{path}') to see available names."
        )
    if vtk_vec.GetNumberOfComponents() != 3:
        raise ValueError(
            f"Array '{vector_array}' has "
            f"{vtk_vec.GetNumberOfComponents()} components, expected 3."
        )

    vec_np = numpy_support.vtk_to_numpy(vtk_vec)          # shape (N, 3)
    # VTK's flat ordering is x-fastest, then y, then z -> reshape (nz,ny,nx,3)
    vec_grid = vec_np.reshape(nz, ny, nx, 3)

    Mx = vec_grid[..., 0].astype(np.float64)
    My = vec_grid[..., 1].astype(np.float64)
    Mz = vec_grid[..., 2].astype(np.float64)
    magnitude = np.sqrt(Mx**2 + My**2 + Mz**2)
    mask = magnitude > 0  # adjust threshold if your "empty" voxels aren't exactly 0

    # --- coordinate grids -------------------------------------------------
    Zg, Yg, Xg = np.meshgrid(z, y, x, indexing="ij")  # each shape (nz,ny,nx)

    sample = {
        "name": sample_name or os.path.splitext(os.path.basename(path))[0],
        "path": path,
        "kind": kind,               # 'image' (.vti) or 'rectilinear' (.vtr)
        "dims": (nx, ny, nz),
        "x_coords": x, "y_coords": y, "z_coords": z,  # 1D physical-unit axes
        "shape": (nz, ny, nx),
        "X": Xg, "Y": Yg, "Z": Zg,
        "Mx": Mx, "My": My, "Mz": Mz,
        "M_vec": vec_grid,
        "magnitude": magnitude,
        "mask": mask,
    }

    # --- optional extra scalar arrays (mask, confidence, density, ...) ---
    if scalar_arrays:
        for name in scalar_arrays:
            vtk_s = data_source.GetArray(name)
            if vtk_s is None:
                print(f"[load_vti_sample] warning: scalar array "
                      f"'{name}' not found, skipping.")
                continue
            s_np = numpy_support.vtk_to_numpy(vtk_s)
            sample[name] = s_np.reshape(nz, ny, nx)

    return sample


# --------------------------------------------------------------------------
# 3. Batch loader — build the per-sample dictionary-of-dictionaries you're
#    already using in this project.
# --------------------------------------------------------------------------
def load_vti_batch(file_map, vector_array=None, scalar_arrays=None,
                    use_cell_data=False):
    """
    file_map : dict {sample_name: path_to_vti}
    Returns  : dict {sample_name: sample_dict}  (see load_vti_sample)
    """
    samples = {}
    for name, path in file_map.items():
        print(f"Loading '{name}' from {path} ...")
        samples[name] = load_vti_sample(
            path,
            vector_array=vector_array,
            scalar_arrays=scalar_arrays,
            use_cell_data=use_cell_data,
            sample_name=name,
        )
    return samples

