# """
# Curvature-based segmentation for isosurfaces (e.g. from skimage.measure.marching_cubes).

# Computes discrete mean curvature (H) and Gaussian curvature (K) per vertex and
# thresholds each into interpretable regions:

#     Mean curvature H      -> 'concave' / 'flat' / 'convex'
#     Gaussian curvature K  -> 'hyperbolic (saddle)' / 'parabolic/flat' / 'elliptic (dome/cup)'

# Requires: numpy, trimesh, matplotlib
# """

# # import numpy as np
# # import trimesh
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# # # def segment_surface_by_curvature(
# # #     verts,
# # #     faces,
# # #     radius=None,
# # #     flat_percentile=33.0,
# # #     plot=True,
# # #     cmap="coolwarm",
# # # ):
# # #     """
# # #     Segment an isosurface into curvature-based regions using both mean and
# # #     Gaussian curvature, so the two can be compared directly.

# # #     Parameters
# # #     ----------
# # #     verts : (N, 3) array
# # #         Vertex positions, e.g. from skimage.measure.marching_cubes.
# # #     faces : (M, 3) int array
# # #         Triangle indices into `verts`.
# # #     radius : float, optional
# # #         Neighborhood radius used to estimate curvature at each vertex
# # #         (trimesh integrates curvature over a ball of this radius around
# # #         each point). If None, defaults to 2x the mean edge length, which
# # #         is a reasonable smoothing scale for marching-cubes meshes (these
# # #         are often noisy at the single-edge scale).
# # #     flat_percentile : float, default 33
# # #         Vertices with |curvature| below this percentile (of the curvature's
# # #         magnitude distribution) are labeled 'flat' / 'parabolic'. Raising
# # #         this makes the flat region larger and the convex/concave (or
# # #         elliptic/hyperbolic) regions smaller, and vice versa.
# # #     plot : bool, default True
# # #         If True, render a 2x2 figure comparing raw and segmented curvature
# # #         for both measures.
# # #     cmap : str, default 'coolwarm'
# # #         Colormap used for the continuous curvature plots.

# # #     Returns
# # #     -------
# # #     dict with keys:
# # #         'mean_curvature'      : (N,) array, estimated mean curvature H per vertex
# # #         'gaussian_curvature'  : (N,) array, estimated Gaussian curvature K per vertex
# # #         'labels_mean'         : (N,) str array, one of 'concave'/'flat'/'convex'
# # #         'labels_gauss'        : (N,) str array, one of
# # #                                  'hyperbolic (saddle)'/'parabolic/flat'/'elliptic (dome/cup)'
# # #         'thresholds'          : dict with the numeric thresholds actually used
# # #         'radius'              : the neighborhood radius actually used
# # #         'mesh'                : the trimesh.Trimesh object built internally
# # #     """
# # #     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

# # #     if radius is None:
# # #         radius = 2.0 * float(np.mean(mesh.edges_unique_length))

# # #     # Discrete curvature *measures*: these are curvature integrated over a
# # #     # ball of `radius` around each vertex. Dividing by the disk area turns
# # #     # them into an approximate pointwise curvature value, which is what
# # #     # makes them comparable across different mesh densities.
# # #     mean_measure = trimesh.curvature.discrete_mean_curvature_measure(mesh, verts, radius)
# # #     gauss_measure = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, verts, radius)

# # #     area_norm = np.pi * radius**2
# # #     H = mean_measure / area_norm
# # #     K = gauss_measure / area_norm

# # #     # Percentile-based thresholds: robust to differing curvature scales
# # #     # without the user having to guess absolute units.
# # #     h_thresh = np.percentile(np.abs(H), flat_percentile)
# # #     k_thresh = np.percentile(np.abs(K), flat_percentile)

# # #     labels_mean = np.where(
# # #         H > h_thresh, "convex", np.where(H < -h_thresh, "concave", "flat")
# # #     )
# # #     labels_gauss = np.where(
# # #         K > k_thresh,
# # #         "elliptic (dome/cup)",
# # #         np.where(K < -k_thresh, "hyperbolic (saddle)", "parabolic/flat"),
# # #     )

# # #     results = {
# # #         "mean_curvature": H,
# # #         "gaussian_curvature": K,
# # #         "labels_mean": labels_mean,
# # #         "labels_gauss": labels_gauss,
# # #         "thresholds": {"mean": h_thresh, "gauss": k_thresh},
# # #         "radius": radius,
# # #         "mesh": mesh,
# # #     }

# # #     if plot:
# # #         _plot_comparison(verts, faces, H, K, labels_mean, labels_gauss, cmap)

# # #     return results


# # # # ----------------------------------------------------------------------
# # # # Plotting helpers
# # # # ----------------------------------------------------------------------

# # # _MEAN_COLORS = {"concave": "#3b6fd6", "flat": "#b0b0b0", "convex": "#d63b3b"}
# # # _GAUSS_COLORS = {
# # #     "hyperbolic (saddle)": "#e0a52c",
# # #     "parabolic/flat": "#b0b0b0",
# # #     "elliptic (dome/cup)": "#3ba05a",
# # # }


# # # def _face_values_from_vertex(values, faces):
# # #     """Average a per-vertex scalar/array over each face's three vertices."""
# # #     return values[faces].mean(axis=1)


# # # def _add_continuous_mesh(ax, verts, faces, vertex_values, cmap, title):
# # #     face_vals = _face_values_from_vertex(vertex_values, faces)
# # #     norm = plt.Normalize(vmin=np.percentile(face_vals, 1), vmax=np.percentile(face_vals, 99))
# # #     colors = plt.get_cmap(cmap)(norm(face_vals))

# # #     poly = Poly3DCollection(verts[faces], facecolors=colors, edgecolor="none")
# # #     ax.add_collection3d(poly)
# # #     _finalize_3d_axes(ax, verts, title)

# # #     mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
# # #     mappable.set_array([])
# # #     plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)


# # # def _add_categorical_mesh(ax, verts, faces, vertex_labels, color_map, title):
# # #     # majority label per face, from its three vertices
# # #     face_labels = vertex_labels[faces]
# # #     face_labels = np.array(
# # #         [np.unique(row, return_counts=True)[0][np.argmax(np.unique(row, return_counts=True)[1])]
# # #          for row in face_labels]
# # #     )
# # #     colors = np.array([color_map[label] for label in face_labels])

# # #     poly = Poly3DCollection(verts[faces], facecolors=colors, edgecolor="none")
# # #     ax.add_collection3d(poly)
# # #     _finalize_3d_axes(ax, verts, title)

# # #     handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=lbl)
# # #                for lbl, c in color_map.items()]
# # #     ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


# # # def _finalize_3d_axes(ax, verts, title):
# # #     mins, maxs = verts.min(axis=0), verts.max(axis=0)
# # #     ax.set_xlim(mins[0], maxs[0])
# # #     ax.set_ylim(mins[1], maxs[1])
# # #     ax.set_zlim(mins[2], maxs[2])
# # #     ax.set_title(title, fontsize=10)
# # #     ax.set_box_aspect(maxs - mins)
# # #     ax.set_axis_off()


# # # def _plot_comparison(verts, faces, H, K, labels_mean, labels_gauss, cmap):
# # #     fig = plt.figure(figsize=(12, 10))

# # #     ax1 = fig.add_subplot(2, 2, 1, projection="3d")
# # #     _add_continuous_mesh(ax1, verts, faces, H, cmap, "Mean curvature (H)")

# # #     ax2 = fig.add_subplot(2, 2, 2, projection="3d")
# # #     _add_categorical_mesh(ax2, verts, faces, labels_mean, _MEAN_COLORS,
# # #                            "Mean curvature segmentation")

# # #     ax3 = fig.add_subplot(2, 2, 3, projection="3d")
# # #     _add_continuous_mesh(ax3, verts, faces, K, cmap, "Gaussian curvature (K)")

# # #     ax4 = fig.add_subplot(2, 2, 4, projection="3d")
# # #     _add_categorical_mesh(ax4, verts, faces, labels_gauss, _GAUSS_COLORS,
# # #                            "Gaussian curvature segmentation")

# # #     fig.tight_layout()
# # #     plt.show()



# # def _mixed_voronoi_areas(mesh):
# #     """Barycentric approximation of the mixed Voronoi area at each vertex:
# #     one third of the area of every incident triangle."""
# #     tri_areas = mesh.area_faces
# #     vertex_area = np.zeros(len(mesh.vertices))
# #     faces = mesh.faces
# #     for k in range(3):
# #         np.add.at(vertex_area, faces[:, k], tri_areas / 3.0)
# #     return vertex_area
 
 
# # def cotangent_mean_curvature(mesh):
# #     """Signed mean curvature H per vertex via the cotangent-weighted
# #     Laplace-Beltrami operator. Positive H = locally convex w.r.t. the
# #     outward vertex normal (matches the H=1/R convention for a sphere)."""
# #     verts, faces = mesh.vertices, mesh.faces
# #     n = len(verts)
# #     Hx, Hy, Hz = np.zeros(n), np.zeros(n), np.zeros(n)
 
# #     for perm in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
# #         i, j, k = faces[:, perm[0]], faces[:, perm[1]], faces[:, perm[2]]
# #         vi, vj, vk = verts[i], verts[j], verts[k]
# #         e1, e2 = vi - vk, vj - vk
# #         cos_ang = np.einsum("ij,ij->i", e1, e2) / (
# #             np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-12
# #         )
# #         cos_ang = np.clip(cos_ang, -1, 1)
# #         sin_ang = np.sqrt(1 - cos_ang**2) + 1e-12
# #         cot = cos_ang / sin_ang
 
# #         diff = vi - vj
# #         contrib = cot[:, None] * diff
# #         np.add.at(Hx, i, contrib[:, 0]); np.add.at(Hy, i, contrib[:, 1]); np.add.at(Hz, i, contrib[:, 2])
# #         np.add.at(Hx, j, -contrib[:, 0]); np.add.at(Hy, j, -contrib[:, 1]); np.add.at(Hz, j, -contrib[:, 2])
 
# #     area = _mixed_voronoi_areas(mesh)
# #     Hvec = np.stack([Hx, Hy, Hz], axis=1) / (2 * area[:, None])
# #     Hmag = np.linalg.norm(Hvec, axis=1) * 0.5
 
# #     sign = np.sign(np.einsum("ij,ij->i", Hvec, mesh.vertex_normals))
# #     return sign * Hmag
 
 
# # def exact_gaussian_curvature(mesh):
# #     """Signed Gaussian curvature K per vertex via the exact discrete
# #     Gauss-Bonnet angle defect, divided by the mixed Voronoi area. No
# #     smoothing radius involved, so saddle regions aren't averaged away."""
# #     defects = trimesh.curvature.vertex_defects(mesh)
# #     area = _mixed_voronoi_areas(mesh)
# #     return defects / area
 
 
# # def _auto_zero_tol(values, frac=0.005):
# #     """Small absolute tolerance around zero, scaled to the data's own
# #     dispersion (5th-95th percentile spread) rather than to its rank."""
# #     lo, hi = np.percentile(values, [5, 95])
# #     return frac * (hi - lo)
 
 
# # # ----------------------------------------------------------------------
# # # Main entry point
# # # ----------------------------------------------------------------------
 
# # def segment_surface_by_curvature(
# #     verts,
# #     faces,
# #     zero_tol_frac=0.05,
# #     plot=True,
# #     cmap="coolwarm",
# # ):
# #     """
# #     Segment an isosurface into curvature-based regions using both mean and
# #     Gaussian curvature, so the two can be compared directly.
 
# #     Parameters
# #     ----------
# #     verts : (N, 3) array
# #         Vertex positions, e.g. from skimage.measure.marching_cubes.
# #     faces : (M, 3) int array
# #         Triangle indices into `verts`.
# #     zero_tol_frac : float, default 0.05
# #         Half-width of the "flat"/"parabolic" band around true zero, as a
# #         fraction of each curvature's own 5th-95th percentile spread.
# #         Lower this to call more of the surface convex/concave or
# #         elliptic/hyperbolic; raise it to call more of it flat. This only
# #         controls the width of the near-zero band — it never causes a
# #         sign to be manufactured on the wrong side of zero.
# #     plot : bool, default True
# #         If True, render a 2x2 figure comparing raw and segmented curvature
# #         for both measures.
# #     cmap : str, default 'coolwarm'
# #         Colormap used for the continuous curvature plots.
 
# #     Returns
# #     -------
# #     dict with keys:
# #         'mean_curvature'      : (N,) array, signed mean curvature H per vertex
# #         'gaussian_curvature'  : (N,) array, signed Gaussian curvature K per vertex
# #         'labels_mean'         : (N,) str array, one of 'concave'/'flat'/'convex'
# #         'labels_gauss'        : (N,) str array, one of
# #                                  'hyperbolic (saddle)'/'parabolic/flat'/'elliptic (dome/cup)'
# #         'thresholds'          : dict with the numeric zero-tolerances actually used
# #         'mesh'                : the trimesh.Trimesh object built internally
# #     """
# #     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
 
# #     H = cotangent_mean_curvature(mesh)
# #     K = exact_gaussian_curvature(mesh)
 
# #     h_tol = _auto_zero_tol(H, zero_tol_frac)
# #     k_tol = _auto_zero_tol(K, zero_tol_frac)
 
# #     labels_mean = np.where(H > h_tol, "convex", np.where(H < -h_tol, "concave", "flat"))
# #     labels_gauss = np.where(
# #         K > k_tol,
# #         "elliptic (dome/cup)",
# #         np.where(K < -k_tol, "hyperbolic (saddle)", "parabolic/flat"),
# #     )
 
# #     n_hyperbolic = int((labels_gauss == "hyperbolic (saddle)").sum())
# #     if n_hyperbolic == 0:
# #         print(
# #             "Note: no vertices classified as hyperbolic (saddle) — K >= 0 "
# #             "everywhere on this surface (within tolerance). That means no "
# #             "true saddle-type regions were found at this resolution, which "
# #             "can be a real geometric property of a globally convex/blobby "
# #             "shape. If you expect saddle features, try lowering "
# #             "zero_tol_frac, or check mesh resolution near those features."
# #         )
 
# #     results = {
# #         "mean_curvature": H,
# #         "gaussian_curvature": K,
# #         "labels_mean": labels_mean,
# #         "labels_gauss": labels_gauss,
# #         "thresholds": {"mean": h_tol, "gauss": k_tol},
# #         "mesh": mesh,
# #     }
 
# #     if plot:
# #         _plot_comparison(verts, faces, H, K, labels_mean, labels_gauss, cmap)
 
# #     return results
 
 
# # # ----------------------------------------------------------------------
# # # Plotting helpers
# # # ----------------------------------------------------------------------
 
# # _MEAN_COLORS = {"concave": "#3b6fd6", "flat": "#b0b0b0", "convex": "#d63b3b"}
# # _GAUSS_COLORS = {
# #     "hyperbolic (saddle)": "#e0a52c",
# #     "parabolic/flat": "#b0b0b0",
# #     "elliptic (dome/cup)": "#3ba05a",
# # }
 
 
# # def _face_values_from_vertex(values, faces):
# #     """Average a per-vertex scalar/array over each face's three vertices."""
# #     return values[faces].mean(axis=1)
 
 
# # def _add_continuous_mesh(ax, verts, faces, vertex_values, cmap, title):
# #     face_vals = _face_values_from_vertex(vertex_values, faces)
# #     norm = plt.Normalize(vmin=np.percentile(face_vals, 1), vmax=np.percentile(face_vals, 99))
# #     colors = plt.get_cmap(cmap)(norm(face_vals))
 
# #     poly = Poly3DCollection(verts[faces], facecolors=colors, edgecolor="none")
# #     ax.add_collection3d(poly)
# #     _finalize_3d_axes(ax, verts, title)
 
# #     mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
# #     mappable.set_array([])
# #     plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
 
 
# # def _add_categorical_mesh(ax, verts, faces, vertex_labels, color_map, title):
# #     # majority label per face, from its three vertices
# #     face_labels = vertex_labels[faces]
# #     face_labels = np.array(
# #         [np.unique(row, return_counts=True)[0][np.argmax(np.unique(row, return_counts=True)[1])]
# #          for row in face_labels]
# #     )
# #     colors = np.array([color_map[label] for label in face_labels])
 
# #     poly = Poly3DCollection(verts[faces], facecolors=colors, edgecolor="none")
# #     ax.add_collection3d(poly)
# #     _finalize_3d_axes(ax, verts, title)
 
# #     handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=lbl)
# #                for lbl, c in color_map.items()]
# #     ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
 
 
# # def _finalize_3d_axes(ax, verts, title):
# #     mins, maxs = verts.min(axis=0), verts.max(axis=0)
# #     ax.set_xlim(mins[0], maxs[0])
# #     ax.set_ylim(mins[1], maxs[1])
# #     ax.set_zlim(mins[2], maxs[2])
# #     ax.set_title(title, fontsize=10)
# #     ax.set_box_aspect(maxs - mins)
# #     ax.set_axis_off()
 
 
# # def _plot_comparison(verts, faces, H, K, labels_mean, labels_gauss, cmap):
# #     fig = plt.figure(figsize=(12, 10))
 
# #     ax1 = fig.add_subplot(2, 2, 1, projection="3d")
# #     _add_continuous_mesh(ax1, verts, faces, H, cmap, "Mean curvature (H)")
 
# #     ax2 = fig.add_subplot(2, 2, 2, projection="3d")
# #     _add_categorical_mesh(ax2, verts, faces, labels_mean, _MEAN_COLORS,
# #                            "Mean curvature segmentation")
 
# #     ax3 = fig.add_subplot(2, 2, 3, projection="3d")
# #     _add_continuous_mesh(ax3, verts, faces, K, cmap, "Gaussian curvature (K)")
 
# #     ax4 = fig.add_subplot(2, 2, 4, projection="3d")
# #     _add_categorical_mesh(ax4, verts, faces, labels_gauss, _GAUSS_COLORS,
# #                            "Gaussian curvature segmentation")
 
# #     fig.tight_layout()
# #     plt.show()
 

# """
# Curvature-based segmentation for isosurfaces (e.g. from skimage.measure.marching_cubes).

# Computes mean curvature (H) and Gaussian curvature (K) per vertex using a
# local osculating-quadric fit: for each vertex, the k nearest neighbors
# *in space* (not mesh connectivity) are used to least-squares fit a small
# paraboloid, and H/K are read off that fit analytically. This is
# deliberately decoupled from mesh triangle connectivity, which matters a
# lot in practice: marching-cubes meshes often have thin, irregular
# triangles, and combinatorial curvature formulas (cotangent Laplacian,
# angle-defect/Gauss-Bonnet) are sensitive to exactly that irregularity.
# On such meshes they can report high-frequency, sign-flipping "curvature"
# that is really just triangulation noise — smoothing vertex positions
# does not fix this, because the noise comes from connectivity, not
# position. The quadric fit sidesteps the problem by never using mesh
# connectivity for the estimate at all.

# Validated against the analytic curvature of a sphere (H = 1/R, K = 1/R^2)
# in the accompanying tests.

# Each is then thresholded into interpretable regions:

#     Mean curvature H      -> 'concave' / 'flat' / 'convex'
#     Gaussian curvature K  -> 'hyperbolic (saddle)' / 'parabolic/flat' / 'elliptic (dome/cup)'

# IMPORTANT — thresholding is zero-referenced, not percentile-based. K > 0
# (elliptic) and K < 0 (hyperbolic) are geometrically distinct shape types
# (same-sign vs. opposite-sign principal curvatures), not just "high" and
# "low" values of the same quantity. Splitting at a percentile of |K| would
# silently relabel part of a uniformly-signed distribution as the opposite
# type. Instead, the "flat/parabolic" band is a small absolute tolerance
# around true zero (by default, 5% of the 5th-95th percentile spread of the
# data), and points outside that band keep whatever sign they actually have.
# If your surface has no true saddle regions, `labels_gauss` will correctly
# contain zero 'hyperbolic' vertices rather than manufacturing some from the
# low end of an all-positive distribution.

# Requires: numpy, trimesh, matplotlib
# """

# import numpy as np
# import trimesh
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from scipy.spatial import cKDTree


# # ----------------------------------------------------------------------
# # Curvature estimator
# # ----------------------------------------------------------------------

# def quadric_curvature(verts, vertex_normals, k=15):
#     """
#     Estimate signed mean (H) and Gaussian (K) curvature per vertex by
#     fitting a local osculating paraboloid through each vertex's k nearest
#     spatial neighbors (via a KD-tree — mesh connectivity is not used at
#     all), then reading H and K off the fitted quadratic analytically.

#     Parameters
#     ----------
#     verts : (N, 3) array
#     vertex_normals : (N, 3) array
#         Per-vertex outward unit normals (e.g. mesh.vertex_normals).
#     k : int, default 15
#         Number of nearest neighbors used in each local fit. Larger k
#         smooths out more noise but also more real small-scale features;
#         smaller k preserves fine features but is noisier. For marching
#         cubes meshes, 10-25 is a reasonable range to try.

#     Returns
#     -------
#     H, K : (N,) arrays. Sign convention: H > 0 is locally convex w.r.t.
#         the outward normal (matches H=1/R for a sphere).
#     """
#     n = len(verts)
#     tree = cKDTree(verts)
#     _, idx = tree.query(verts, k=k + 1)
#     idx = idx[:, 1:]  # drop self

#     normal = vertex_normals
#     arbitrary = np.where(np.abs(normal[:, :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
#     u = np.cross(normal, arbitrary)
#     u /= np.linalg.norm(u, axis=1, keepdims=True)
#     w = np.cross(normal, u)

#     rel = verts[idx] - verts[:, None, :]
#     x = np.einsum("nki,ni->nk", rel, u)
#     y = np.einsum("nki,ni->nk", rel, w)
#     z = np.einsum("nki,ni->nk", rel, normal)

#     # least-squares fit z = A x^2 + B xy + C y^2 + D x + E y, solved via
#     # the normal equations for every vertex at once (batched 5x5 solve)
#     design = np.stack([x**2, x * y, y**2, x, y], axis=-1)  # (n, k, 5)
#     MtM = np.einsum("nki,nkj->nij", design, design)
#     Mtz = np.einsum("nki,nk->ni", design, z)
#     MtM += np.eye(5) * 1e-10  # tiny ridge term for numerical stability
#     coeffs = np.linalg.solve(MtM, Mtz[..., None])[..., 0]
#     A, B, C = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

#     H = -(A + C)          # sign convention: + convex w.r.t. outward normal
#     K = 4 * A * C - B**2
#     return H, K


# def _auto_zero_tol(values, frac=0.05):
#     """Small absolute tolerance around zero, scaled to the data's own
#     dispersion (5th-95th percentile spread) rather than to its rank."""
#     lo, hi = np.percentile(values, [5, 95])
#     return frac * (hi - lo)


# # ----------------------------------------------------------------------
# # Main entry point
# # ----------------------------------------------------------------------

# def segment_surface_by_curvature(
#     verts,
#     faces,
#     k_neighbors=15,
#     zero_tol_frac=0.05,
#     plot=True,
#     cmap="coolwarm",
# ):
#     """
#     Segment an isosurface into curvature-based regions using both mean and
#     Gaussian curvature, so the two can be compared directly.

#     Parameters
#     ----------
#     verts : (N, 3) array
#         Vertex positions, e.g. from skimage.measure.marching_cubes.
#     faces : (M, 3) int array
#         Triangle indices into `verts`.
#     k_neighbors : int, default 15
#         Number of nearest neighbors used in the local quadric fit at each
#         vertex. Raise this if your surface is noisy (e.g. raw
#         marching-cubes output, thin/ribbon-like or tube-like geometry);
#         lower it if you're washing out real small-scale features.
#     zero_tol_frac : float, default 0.05
#         Half-width of the "flat"/"parabolic" band around true zero, as a
#         fraction of each curvature's own 5th-95th percentile spread.
#         Lower this to call more of the surface convex/concave or
#         elliptic/hyperbolic; raise it to call more of it flat. This only
#         controls the width of the near-zero band — it never causes a
#         sign to be manufactured on the wrong side of zero. Surfaces that
#         are close to developable almost everywhere (tubes, ribbons) tend
#         to need a larger value here (try 0.1-0.3) since K hovers near
#         zero over most of the surface and a tight band will flicker
#         label as the smooth field gently crosses zero.
#     plot : bool, default True
#         If True, render a 2x2 figure comparing raw and segmented curvature
#         for both measures.
#     cmap : str, default 'coolwarm'
#         Colormap used for the continuous curvature plots.

#     Returns
#     -------
#     dict with keys:
#         'mean_curvature'      : (N,) array, signed mean curvature H per vertex
#         'gaussian_curvature'  : (N,) array, signed Gaussian curvature K per vertex
#         'labels_mean'         : (N,) str array, one of 'concave'/'flat'/'convex'
#         'labels_gauss'        : (N,) str array, one of
#                                  'hyperbolic (saddle)'/'parabolic/flat'/'elliptic (dome/cup)'
#         'thresholds'          : dict with the numeric zero-tolerances actually used
#         'mesh'                : the trimesh.Trimesh object built internally
#     """
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

#     H, K = quadric_curvature(mesh.vertices, mesh.vertex_normals, k=k_neighbors)

#     h_tol = _auto_zero_tol(H, zero_tol_frac)
#     k_tol = _auto_zero_tol(K, zero_tol_frac)

#     labels_mean = np.where(H > h_tol, "convex", np.where(H < -h_tol, "concave", "flat"))
#     labels_gauss = np.where(
#         K > k_tol,
#         "elliptic (dome/cup)",
#         np.where(K < -k_tol, "hyperbolic (saddle)", "parabolic/flat"),
#     )

#     n_hyperbolic = int((labels_gauss == "hyperbolic (saddle)").sum())
#     if n_hyperbolic == 0:
#         print(
#             "Note: no vertices classified as hyperbolic (saddle) — K >= 0 "
#             "everywhere on this surface (within tolerance). That means no "
#             "true saddle-type regions were found at this resolution, which "
#             "can be a real geometric property of a globally convex/blobby "
#             "shape. If you expect saddle features, try lowering "
#             "zero_tol_frac, or check mesh resolution near those features."
#         )

#     results = {
#         "mean_curvature": H,
#         "gaussian_curvature": K,
#         "labels_mean": labels_mean,
#         "labels_gauss": labels_gauss,
#         "thresholds": {"mean": h_tol, "gauss": k_tol},
#         "mesh": mesh,
#     }

#     if plot:
#         _plot_comparison(verts, faces, H, K, labels_mean, labels_gauss, cmap)

#     return results


# # ----------------------------------------------------------------------
# # Plotting helpers
# # ----------------------------------------------------------------------

# _MEAN_COLORS = {"concave": "#3b6fd6", "flat": "#b0b0b0", "convex": "#d63b3b"}
# _GAUSS_COLORS = {
#     "hyperbolic (saddle)": "#e0a52c",
#     "parabolic/flat": "#b0b0b0",
#     "elliptic (dome/cup)": "#3ba05a",
# }


# def _face_values_from_vertex(values, faces):
#     """Average a per-vertex scalar/array over each face's three vertices."""
#     return values[faces].mean(axis=1)


# def _add_continuous_mesh(ax, verts, faces, vertex_values, cmap, title):
#     face_vals = _face_values_from_vertex(vertex_values, faces)
#     norm = plt.Normalize(vmin=np.percentile(face_vals, 1), vmax=np.percentile(face_vals, 99))
#     colors = plt.get_cmap(cmap)(norm(face_vals))

#     poly = Poly3DCollection(verts[faces], facecolors=colors, edgecolor="none")
#     ax.add_collection3d(poly)
#     _finalize_3d_axes(ax, verts, title)

#     mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
#     mappable.set_array([])
#     plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)


# def _add_categorical_mesh(ax, verts, faces, vertex_labels, color_map, title):
#     # majority label per face, from its three vertices
#     face_labels = vertex_labels[faces]
#     face_labels = np.array(
#         [np.unique(row, return_counts=True)[0][np.argmax(np.unique(row, return_counts=True)[1])]
#          for row in face_labels]
#     )
#     colors = np.array([color_map[label] for label in face_labels])

#     poly = Poly3DCollection(verts[faces], facecolors=colors, edgecolor="none")
#     ax.add_collection3d(poly)
#     _finalize_3d_axes(ax, verts, title)

#     handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=lbl)
#                for lbl, c in color_map.items()]
#     ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


# def _finalize_3d_axes(ax, verts, title):
#     mins, maxs = verts.min(axis=0), verts.max(axis=0)
#     ax.set_xlim(mins[0], maxs[0])
#     ax.set_ylim(mins[1], maxs[1])
#     ax.set_zlim(mins[2], maxs[2])
#     ax.set_title(title, fontsize=10)
#     ax.set_box_aspect(maxs - mins)
#     ax.set_axis_off()


# def _plot_comparison(verts, faces, H, K, labels_mean, labels_gauss, cmap):
#     fig = plt.figure(figsize=(12, 10))

#     ax1 = fig.add_subplot(2, 2, 1, projection="3d")
#     _add_continuous_mesh(ax1, verts, faces, H, cmap, "Mean curvature (H)")

#     ax2 = fig.add_subplot(2, 2, 2, projection="3d")
#     _add_categorical_mesh(ax2, verts, faces, labels_mean, _MEAN_COLORS,
#                            "Mean curvature segmentation")

#     ax3 = fig.add_subplot(2, 2, 3, projection="3d")
#     _add_continuous_mesh(ax3, verts, faces, K, cmap, "Gaussian curvature (K)")

#     ax4 = fig.add_subplot(2, 2, 4, projection="3d")
#     _add_categorical_mesh(ax4, verts, faces, labels_gauss, _GAUSS_COLORS,
#                            "Gaussian curvature segmentation")

#     fig.tight_layout()
#     plt.show()

################################################################# Commented out NOV 26 ################################################################################
# """
# 2D domain-wall curvature for binary/labeled field images (e.g. a
# micromagnetic Mz snapshot showing up/down domains).

# Curvature here uses the basic planar-curve definition you asked for:
# kappa = dT/ds, the rate of change of the tangent angle with arc length.
# Equivalently, for any regular parametrization (x(u), y(u)) of the curve
# (no need to reparametrize by arc length explicitly -- the formula below
# is invariant to that):

#     kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)

# Pipeline
# --------
# 1. The boundary between the two domains (e.g. yellow/blue) is extracted
#    as a set of subpixel contours via marching squares
#    (skimage.measure.find_contours) at `level` (0.0 is the natural choice
#    for a field like Mz that is positive in one domain and negative in
#    the other).
# 2. Each raw contour is pixel-jagged (marching-squares staircasing), which
#    would make a naive finite-difference curvature extremely noisy. So
#    each contour is fit with a smoothing spline (scipy.interpolate.splprep)
#    before differentiating -- this is the 2D analogue of the mesh-noise
#    problem from curvature on 3D marching-cubes surfaces.
# 3. kappa is evaluated analytically from the spline's 1st/2nd derivatives.

# Sign convention: kappa > 0 where the boundary is locally convex bulging
# INTO the level > 0 domain (e.g. an isolated round "yellow" island has
# kappa = +1/R everywhere on its boundary); kappa < 0 marks concave
# indentations (e.g. the "blue" domain intruding into "yellow"). Validated
# against a circle of known radius (kappa should equal 1/R) in the tests.

# Physical units: pass `pixel_size` (physical length per pixel, e.g. 2.0
# for 2 nm/pixel) to get curvature in 1/[that length unit] rather than
# 1/pixel. Curvature has units of 1/length, so converting is a straight
# scaling: kappa_physical = kappa_pixel_units / pixel_size (contour
# coordinates are scaled by pixel_size the same way, for correct plotting).
# This was validated against the analytic circle test: doubling pixel_size
# exactly halves the reported curvature, as it must.

# Requires: numpy, scipy, scikit-image, matplotlib
# """

# import numpy as np
# from skimage.measure import find_contours
# from scipy.interpolate import splprep, splev
# import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from matplotlib.colors import TwoSlopeNorm


# def contour_curvature(field, level=0.0, smoothing=2.0, n_eval=400, min_points=8,
#                        pixel_size=1.0):
#     """
#     Extract the domain boundary/boundaries of `field` at `level` and
#     compute signed curvature along each.

#     Parameters
#     ----------
#     field : (H, W) array
#         Scalar field, e.g. your Mz array. Sign of (field - level) defines
#         the two domains.
#     level : float, default 0.0
#         Threshold defining the boundary. 0.0 is the natural choice for a
#         field like Mz that is symmetric about zero.
#     smoothing : float, default 2.0
#         Spline smoothing strength, passed to scipy as `s = smoothing *
#         n_points_in_contour`. Larger values fit a smoother curve (less
#         sensitive to pixel staircasing, but can round off genuinely sharp
#         features); smaller values track the raw pixel boundary more
#         closely (noisier curvature). Start around 1-3 and adjust by eye
#         against the plot. Applied in pixel-index space, so this doesn't
#         need to change when you change `pixel_size`.
#     n_eval : int, default 400
#         Number of points each contour is resampled to for evaluating
#         curvature (independent of the original pixel contour length).
#     min_points : int, default 8
#         Contours with fewer raw points than this are skipped (too short
#         to reliably spline-fit; usually stray single-pixel noise).
#     pixel_size : float, default 1.0
#         Physical length per pixel (e.g. 2.0 for 2 nm/pixel). Curvature
#         scales as 1/pixel_size and coordinates scale as pixel_size, so
#         with the default of 1.0 everything stays in pixel/voxel units
#         exactly as before. Assumes isotropic (square) pixels.

#     Returns
#     -------
#     list of dicts, one per contour, each with:
#         'xy'    : (n_eval, 2) array of (x, y) coordinates in physical
#                   units (x = column * pixel_size, y = row * pixel_size),
#                   suitable for plotting over `field` with imshow(...,
#                   origin='lower', extent=[0, W*pixel_size, 0, H*pixel_size]).
#         'kappa' : (n_eval,) array of signed curvature, in 1/[pixel_size units].
#     """
#     ## find contour between up and down domains 
#     contours = find_contours(field, level=level)
#     results = []
#     for c in contours:
#         if len(c) < min_points:
#             continue
#         rows, cols = c[:, 0], c[:, 1]
#         closed = np.hypot(rows[0] - rows[-1], cols[0] - cols[-1]) < 1.0
#         per = 1 if closed else 0
#         try:
#             ## define a spline of the curve (for each contour, input x y values and fit with a cubic (3) spline)
#             ##  a spline is a function defined in pieces by polynomials that stitch together neighboring points smoothly
#             tck, _ = splprep([cols, rows], s=smoothing * len(c), per=per, k=3)
#         except Exception:
#             continue

#         uu = np.linspace(0, 1, n_eval) ## segments points on the contour from 0 being the beginning of the line and 1 being the end of the line 
#         x, y = splev(uu, tck, der=0) ## evaluated spline and get x y positions
#         dx, dy = splev(uu, tck, der=1) ## first derivative of the spline
#         ddx, ddy = splev(uu, tck, der=2) ## second derivative of the spline

#         ## plane curve definition of curvature (https://en.wikipedia.org/wiki/Curvature)
#         denom = (dx**2 + dy**2) ** 1.5
#         kappa = (dx * ddy - dy * ddx) / np.where(denom == 0, 1e-12, denom)
#         # convert pixel-index-space quantities to physical units
#         kappa = kappa / pixel_size
#         xy = np.stack([x, y], axis=1) * pixel_size
#         results.append({"xy": xy, "kappa": kappa})
#     return results


# def plot_domain_curvature(
#     field,
#     level=0.0,
#     smoothing=2.0,
#     n_eval=400,
#     pixel_size=1.0,
#     unit_label="voxel",
#     cmap="coolwarm",
#     vmax_percentile=98,
#     linewidth=2.5,
#     background_alpha=0.35,
#     figsize=(7, 6),
# ):
#     """
#     Compute and plot domain-wall curvature for `field`, colored with a
#     diverging colormap centered at zero.

#     Parameters
#     ----------
#     pixel_size : float, default 1.0
#         Physical length per pixel (e.g. 2.0 for 2 nm/pixel). See
#         `contour_curvature` -- curvature is reported in 1/[pixel_size units].
#     unit_label : str, default 'voxel'
#         Unit name used in axis labels and the colorbar (e.g. 'nm'). Purely
#         cosmetic -- set this to match whatever `pixel_size` is expressed in.
#     (all other parameters as in `contour_curvature`, plus plot styling)

#     Returns
#     -------
#     (fig, ax, results) where `results` is the same list `contour_curvature`
#     returns, in case you want the raw per-contour curvature values (e.g.
#     for a histogram, or to compare yellow-vs-blue statistics).
#     """
#     results = contour_curvature(field, level=level, smoothing=smoothing,
#                                  n_eval=n_eval, pixel_size=pixel_size)
#     if not results:
#         raise ValueError("No boundary contours found at this level -- check `level` "
#                           "against your field's actual value range.")

#     all_kappa = np.concatenate([r["kappa"] for r in results])
#     vmax = np.percentile(np.abs(all_kappa), vmax_percentile)
#     norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

#     extent = [0, field.shape[1] * pixel_size, 0, field.shape[0] * pixel_size]
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.imshow(field > level, origin="lower", cmap="viridis", alpha=background_alpha,
#               extent=extent)

#     for r in results:
#         xy = r["xy"]
#         segs = np.stack([xy[:-1], xy[1:]], axis=1)
#         lc = LineCollection(segs, cmap=cmap, norm=norm)
#         lc.set_array(r["kappa"][:-1])
#         lc.set_linewidth(linewidth)
#         ax.add_collection(lc)

#     ax.set_xlim(extent[0], extent[1])
#     ax.set_ylim(extent[2], extent[3])
#     ax.set_aspect("equal")
#     ax.set_xlabel(f"x ({unit_label})")
#     ax.set_ylabel(f"y ({unit_label})")
#     ax.set_title("Center line curvature ($\\kappa = dT/ds$)")

#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     fig.colorbar(sm, ax=ax, label=f"κ (1/{unit_label})")
#     fig.tight_layout()

#     n_pos = int((all_kappa > 0).sum())
#     n_neg = int((all_kappa < 0).sum())
#     print(f"{len(results)} boundary contour(s), {len(all_kappa)} points total.")
#     print(f"Convex-into-'field>level' (kappa>0): {n_pos} points, "
#           f"mean {all_kappa[all_kappa>0].mean():.5f}")
#     print(f"Concave / opposite domain intruding (kappa<0): {n_neg} points, "
#           f"mean {all_kappa[all_kappa<0].mean():.5f}")

#     return fig, ax, results


# def curvature_field_dict(results, field_shape, pixel_size=1.0, level=0.0,
#                           slice_axis=None, slice_index=None):
#     """
#     Package contour_curvature() output into the same
#     {array entries..., 'shape':, ...} dict format your other derived-field
#     dicts use (jacobian, curl, divergence, ...), so it can be passed
#     straight into save_volume_analysis:
 
#         results = contour_curvature(Mz_slice, level=0.0, pixel_size=2.0)
#         curvature_dict = curvature_field_dict(
#             results, Mz_slice.shape, pixel_size=2.0, level=0.0,
#             slice_axis='z', slice_index=z_idx,
#         )
#         save_volume_analysis(save_dir, sample_name, date=date,
#                               curvature=curvature_dict,
#                               jacobian=jacobian, curl=curl, ...)
 
#     No changes to save_volume_analysis are needed -- it already handles
#     any dict passed via **named_field_dicts generically.
 
#     Produces two representations of the same curvature data:
 
#     1. 'kappa_map' : dense (H, W) array, same shape as the slice.
#        NaN everywhere except at boundary pixels, where it holds the
#        curvature value (nearest-pixel rasterization of the contour's
#        continuous, sub-pixel curvature values -- if more than one
#        contour point rounds to the same pixel, they're averaged). This
#        is the array to reach for when you want curvature lined up
#        pixel-for-pixel with your other volumetric fields (Mz, jacobian,
#        etc.) at the same (row, col).
 
#     2. 'contour_xy', 'contour_kappa', 'contour_id' : the exact traced
#        boundary points at full sub-pixel spline resolution, flattened
#        across all contours (since the saving format is array-only and
#        can't hold a ragged list-of-arrays directly). 'contour_id' labels
#        which contour (0, 1, 2, ...) each point belongs to, so you can
#        split them back apart:
 
#            xy_of_contour_2 = contour_xy[contour_id == 2]
 
#        Use this representation instead of 'kappa_map' if you want the
#        unrounded curvature values or need to walk along a single
#        contour in order (rasterization destroys point ordering).
 
#     Parameters
#     ----------
#     results : list of dict
#         Output of contour_curvature(...).
#     field_shape : (H, W) tuple
#         Shape of the 2D slice the contours were extracted from, in
#         pixel/voxel-index units (i.e. field.shape, not scaled by
#         pixel_size).
#     pixel_size, level : float
#         The same values passed to contour_curvature -- recorded here as
#         metadata so the saved curvature can be traced back to how it was
#         computed.
#     slice_axis : str, optional
#         Which axis of the 3D volume this slice came from, e.g. 'z'.
#     slice_index : int, optional
#         Index along that axis, e.g. the z-slice number. Recording this
#         is what lets you relate this 2D curvature map back to its
#         position in the original 3D volume.
 
#     Returns
#     -------
#     dict, ready to pass as a **named_field_dicts entry to
#     save_volume_analysis.
#     """
#     H, W = field_shape
#     kappa_sum = np.zeros((H, W))
#     count_map = np.zeros((H, W), dtype=int)
 
#     all_xy, all_kappa, all_id = [], [], []
#     for i, r in enumerate(results):
#         xy_pix = r["xy"] / pixel_size  # back to pixel-index space for rasterization
#         cols = np.clip(np.round(xy_pix[:, 0]).astype(int), 0, W - 1)
#         rows = np.clip(np.round(xy_pix[:, 1]).astype(int), 0, H - 1)
#         np.add.at(kappa_sum, (rows, cols), r["kappa"])
#         np.add.at(count_map, (rows, cols), 1)
 
#         all_xy.append(r["xy"])
#         all_kappa.append(r["kappa"])
#         all_id.append(np.full(len(r["kappa"]), i, dtype=int))
 
#     with np.errstate(invalid="ignore"):
#         kappa_map = np.where(count_map > 0, kappa_sum / np.maximum(count_map, 1), np.nan)
 
#     field_dict = {
#         # array entries
#         "kappa_map": kappa_map,
#         "contour_xy": np.concatenate(all_xy, axis=0) if all_xy else np.zeros((0, 2)),
#         "contour_kappa": np.concatenate(all_kappa) if all_kappa else np.zeros((0,)),
#         "contour_id": np.concatenate(all_id) if all_id else np.zeros((0,), dtype=int),
#         # metadata entries (non-array -> stored in the JSON sidecar)
#         "shape": field_shape,
#         "pixel_size": pixel_size,
#         "level": level,
#         "n_contours": len(results),
#     }
#     if slice_axis is not None:
#         field_dict["slice_axis"] = slice_axis
#     if slice_index is not None:
#         field_dict["slice_index"] = int(slice_index)
 
#     return field_dict



 
# def curvature_volume_dict(field_volume, level=0.0, pixel_size=1.0, smoothing=2.0,
#                            n_eval=400, slice_axis=0, pad_value=0.0, verbose=True):
#     """
#     Compute domain-wall curvature for EVERY slice along `slice_axis` of a
#     3D field volume (e.g. Mz(z, y, x)), and package the whole stack into
#     a dict ready for save_volume_analysis(..., curvature=this_dict).
 
#     This is the volume-level counterpart to curvature_field_dict (which
#     handles a single 2D slice). Use this one when you want curvature
#     saved for the whole volume at once rather than slice by slice.
 
#         results_dict = curvature_volume_dict(Mz_volume, level=0.0,
#                                               pixel_size=2.0, slice_axis=0)
#         save_volume_analysis(save_dir, sample_name, date=date,
#                               curvature=results_dict,
#                               jacobian=jacobian, curl=curl)
 
#     Parameters
#     ----------
#     field_volume : 3D array
#         The full volume, e.g. Mz(z, y, x). Curvature is computed
#         independently on each 2D slice along `slice_axis` -- this does
#         NOT compute a 3D surface curvature, it's the same 2D domain-wall
#         curvature as before, just looped over every slice and stacked.
#     slice_axis : int, default 0
#         Which axis of `field_volume` to iterate slices over. Default 0
#         assumes a (Z, Y, X) layout, i.e. each field_volume[i] is one 2D
#         (Y, X) slice. Set to 1 or 2 if your volume is laid out
#         differently.
#     pad_value : float, default 0.0
#         Value assigned to voxels that are not on any domain wall in
#         their slice. Previously NaN (in curvature_field_dict); now
#         explicit and defaults to 0, as requested, so the saved array has
#         no NaNs in it.
#     level, pixel_size, smoothing, n_eval : as in contour_curvature,
#         applied identically to every slice.
#     verbose : bool, default True
#         Print a one-line progress update per slice (contour count found),
#         useful for volumes with many slices.
 
#     Returns
#     -------
#     dict with array entries:
#         'kappa_map'            : (Nz, Ny, Nx) array (or whatever shape
#                                   matches field_volume, with slice_axis
#                                   in the same position), pad_value off
#                                   the domain walls, curvature value on
#                                   them.
#         'contour_xy'            : (N_total_points, 2) sub-pixel boundary
#                                    coordinates, concatenated across every
#                                    contour in every slice.
#         'contour_kappa'         : (N_total_points,) curvature at each of
#                                    those points.
#         'contour_id'            : (N_total_points,) which contour (local
#                                    to its slice) each point belongs to.
#         'contour_slice_index'   : (N_total_points,) which slice each
#                                    point came from -- use this together
#                                    with contour_id to pull out one
#                                    contour from one slice:
#                                        mask = (contour_slice_index==s) & (contour_id==i)
#                                        xy_s_i = contour_xy[mask]
#     and metadata entries 'shape', 'pixel_size', 'level', 'slice_axis',
#     'n_slices', 'pad_value', 'n_contours_per_slice'.
#     """
#     vol = np.moveaxis(field_volume, slice_axis, 0)  # (n_slices, H, W)
#     n_slices, H, W = vol.shape
 
#     kappa_map_stack = np.full((n_slices, H, W), pad_value, dtype=float)
#     all_xy, all_kappa, all_contour_id, all_slice_index = [], [], [], []
#     n_contours_per_slice = []
 
#     for s in range(n_slices):
#         results = contour_curvature(vol[s], level=level, smoothing=smoothing,
#                                      n_eval=n_eval, pixel_size=pixel_size)
#         n_contours_per_slice.append(len(results))
#         if verbose:
#             print(f"slice {s+1}/{n_slices}: {len(results)} contour(s)")
 
#         kappa_sum = np.zeros((H, W))
#         count_map = np.zeros((H, W), dtype=int)
#         for i, r in enumerate(results):
#             xy_pix = r["xy"] / pixel_size
#             cols = np.clip(np.round(xy_pix[:, 0]).astype(int), 0, W - 1)
#             rows = np.clip(np.round(xy_pix[:, 1]).astype(int), 0, H - 1)
#             np.add.at(kappa_sum, (rows, cols), r["kappa"])
#             np.add.at(count_map, (rows, cols), 1)
 
#             all_xy.append(r["xy"])
#             all_kappa.append(r["kappa"])
#             all_contour_id.append(np.full(len(r["kappa"]), i, dtype=int))
#             all_slice_index.append(np.full(len(r["kappa"]), s, dtype=int))
 
#         with np.errstate(invalid="ignore"):
#             kappa_map_stack[s] = np.where(count_map > 0, kappa_sum / np.maximum(count_map, 1),
#                                            pad_value)
 
#     kappa_map_stack = np.moveaxis(kappa_map_stack, 0, slice_axis)
 
#     field_dict = {
#         # array entries
#         "kappa_map": kappa_map_stack,
#         "contour_xy": np.concatenate(all_xy, axis=0) if all_xy else np.zeros((0, 2)),
#         "contour_kappa": np.concatenate(all_kappa) if all_kappa else np.zeros((0,)),
#         "contour_id": np.concatenate(all_contour_id) if all_contour_id else np.zeros((0,), dtype=int),
#         "contour_slice_index": np.concatenate(all_slice_index) if all_slice_index else np.zeros((0,), dtype=int),
#         # metadata entries
#         "shape": field_volume.shape,
#         "pixel_size": pixel_size,
#         "level": level,
#         "slice_axis": slice_axis,
#         "n_slices": n_slices,
#         "pad_value": pad_value,
#         "n_contours_per_slice": n_contours_per_slice,
#     }
#     return field_dict



# def curvature_field_dict(results, field_shape, pixel_size=1.0, level=0.0,
#                           slice_axis=None, slice_index=None):
#     """
#     Package contour_curvature() output into the same
#     {array entries..., 'shape':, ...} dict format your other derived-field
#     dicts use (jacobian, curl, divergence, ...), so it can be passed
#     straight into save_volume_analysis:
 
#         results = contour_curvature(Mz_slice, level=0.0, pixel_size=2.0)
#         curvature_dict = curvature_field_dict(
#             results, Mz_slice.shape, pixel_size=2.0, level=0.0,
#             slice_axis='z', slice_index=z_idx,
#         )
#         save_volume_analysis(save_dir, sample_name, date=date,
#                               curvature=curvature_dict,
#                               jacobian=jacobian, curl=curl, ...)
 
#     No changes to save_volume_analysis are needed -- it already handles
#     any dict passed via **named_field_dicts generically.
 
#     Produces two representations of the same curvature data:
 
#     1. 'kappa_map' : dense (H, W) array, same shape as the slice.
#        NaN everywhere except at boundary pixels, where it holds the
#        curvature value (nearest-pixel rasterization of the contour's
#        continuous, sub-pixel curvature values -- if more than one
#        contour point rounds to the same pixel, they're averaged). This
#        is the array to reach for when you want curvature lined up
#        pixel-for-pixel with your other volumetric fields (Mz, jacobian,
#        etc.) at the same (row, col).
 
#     2. 'contour_xy', 'contour_kappa', 'contour_id' : the exact traced
#        boundary points at full sub-pixel spline resolution, flattened
#        across all contours (since the saving format is array-only and
#        can't hold a ragged list-of-arrays directly). 'contour_id' labels
#        which contour (0, 1, 2, ...) each point belongs to, so you can
#        split them back apart:
 
#            xy_of_contour_2 = contour_xy[contour_id == 2]
 
#        Use this representation instead of 'kappa_map' if you want the
#        unrounded curvature values or need to walk along a single
#        contour in order (rasterization destroys point ordering).
 
#     Parameters
#     ----------
#     results : list of dict
#         Output of contour_curvature(...).
#     field_shape : (H, W) tuple
#         Shape of the 2D slice the contours were extracted from, in
#         pixel/voxel-index units (i.e. field.shape, not scaled by
#         pixel_size).
#     pixel_size, level : float
#         The same values passed to contour_curvature -- recorded here as
#         metadata so the saved curvature can be traced back to how it was
#         computed.
#     slice_axis : str, optional
#         Which axis of the 3D volume this slice came from, e.g. 'z'.
#     slice_index : int, optional
#         Index along that axis, e.g. the z-slice number. Recording this
#         is what lets you relate this 2D curvature map back to its
#         position in the original 3D volume.
 
#     Returns
#     -------
#     dict, ready to pass as a **named_field_dicts entry to
#     save_volume_analysis.
#     """
#     H, W = field_shape
#     kappa_sum = np.zeros((H, W))
#     count_map = np.zeros((H, W), dtype=int)
 
#     all_xy, all_kappa, all_id = [], [], []
#     for i, r in enumerate(results):
#         xy_pix = r["xy"] / pixel_size  # back to pixel-index space for rasterization
#         cols = np.clip(np.round(xy_pix[:, 0]).astype(int), 0, W - 1)
#         rows = np.clip(np.round(xy_pix[:, 1]).astype(int), 0, H - 1)
#         np.add.at(kappa_sum, (rows, cols), r["kappa"])
#         np.add.at(count_map, (rows, cols), 1)
 
#         all_xy.append(r["xy"])
#         all_kappa.append(r["kappa"])
#         all_id.append(np.full(len(r["kappa"]), i, dtype=int))
 
#     with np.errstate(invalid="ignore"):
#         kappa_map = np.where(count_map > 0, kappa_sum / np.maximum(count_map, 1), np.nan)
 
#     field_dict = {
#         # array entries
#         "kappa_map": kappa_map,
#         "contour_xy": np.concatenate(all_xy, axis=0) if all_xy else np.zeros((0, 2)),
#         "contour_kappa": np.concatenate(all_kappa) if all_kappa else np.zeros((0,)),
#         "contour_id": np.concatenate(all_id) if all_id else np.zeros((0,), dtype=int),
#         # metadata entries (non-array -> stored in the JSON sidecar)
#         "shape": field_shape,
#         "pixel_size": pixel_size,
#         "level": level,
#         "n_contours": len(results),
#     }
#     if slice_axis is not None:
#         field_dict["slice_axis"] = slice_axis
#     if slice_index is not None:
#         field_dict["slice_index"] = int(slice_index)
 
#     return field_dict
 ###################################################################################################################################################################################################


import numpy as np
from scipy.interpolate import splprep, splev
from skimage.measure import find_contours

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import inspect
from functools import partial

############################ Saving ###################################




def _contour_fn_settings(contour_fn):
    """Name and keyword settings (including defaults) of a contour function.
    Works for plain functions and functools.partial objects. A lambda can't
    be introspected, so you only get its name."""
    kwargs, base = {}, contour_fn
    while isinstance(base, partial):
        kwargs = {**base.keywords, **kwargs}
        base = base.func
    settings = {}
    try:
        for name, p in inspect.signature(base).parameters.items():
            if p.default is not inspect.Parameter.empty:
                settings[name] = p.default
    except (TypeError, ValueError):
        pass
    settings.update(kwargs)
    return getattr(base, "__name__", repr(base)), settings


def _to_jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def curvature_field_dict(contour_fn, field, results=None, pixel_size=None,
                         level=None, slice_axis=None, slice_index=None):
    """
    Package contour curvature output into the dict format used by
    save_volume_analysis (same as jacobian, curl, divergence, ...).

    Parameters
    ----------
    contour_fn : callable
        Function taking a 2D field and returning the list of contour dicts
        (best: a functools.partial of contour_curvature_evenspacing). Its
        settings are recorded in the JSON sidecar as `contour_fn_<name>`
        entries. pixel_size and level are read from it unless passed.
    field : (H, W) array
        The slice the contours come from.
    results : list of dict, optional
        Precomputed contour_fn(field). If None it is computed here. Pass the
        results from your plot call so what you inspected is what you save.
    slice_axis, slice_index : optional
        Which axis/index of the 3D volume this slice came from.

    Saved arrays
    ------------
    kappa_map       : (H, W) curvature rasterized to the nearest pixel (NaN
                      elsewhere; points landing on the same pixel are averaged).
                      Lines up pixel-for-pixel with Mz, jacobian, etc.
    contour_xy      : (N, 2) evenly spaced spline points, physical units
    contour_kappa   : (N,)   signed curvature at those points
    contour_normal  : (N, 2) unit surface normals (nx, ny)
    contour_id      : (N,)   which contour each point belongs to
    contour_length  : (n_contours,) arc length of each contour
    Split a contour back out with e.g. contour_xy[contour_id == 2].
    contour_id is per slice: ids are not matched between z slices.
    """
    H, W = field.shape
    fn_name, settings = _contour_fn_settings(contour_fn)
    pixel_size = settings.get("pixel_size") if pixel_size is None else pixel_size
    level = settings.get("level", 0.0) if level is None else level
    if pixel_size is None:
        raise ValueError("pixel_size not found on contour_fn; pass pixel_size= explicitly.")
    if results is None:
        results = contour_fn(field)

    kappa_sum = np.zeros((H, W))
    count_map = np.zeros((H, W), dtype=int)
    all_xy, all_kappa, all_normal, all_id, lengths = [], [], [], [], []

    for i, r in enumerate(results):
        xy_pix = r["xy"] / pixel_size
        cols = np.clip(np.round(xy_pix[:, 0]).astype(int), 0, W - 1)
        rows = np.clip(np.round(xy_pix[:, 1]).astype(int), 0, H - 1)
        np.add.at(kappa_sum, (rows, cols), r["kappa"])
        np.add.at(count_map, (rows, cols), 1)

        all_xy.append(r["xy"])
        all_kappa.append(r["kappa"])
        all_normal.append(r["normal"])
        all_id.append(np.full(len(r["kappa"]), i, dtype=int))
        lengths.append(r["length"])

    with np.errstate(invalid="ignore"):
        kappa_map = np.where(count_map > 0, kappa_sum / np.maximum(count_map, 1), np.nan)

    field_dict = {
        # array entries
        "kappa_map": kappa_map,
        "contour_xy": np.concatenate(all_xy, axis=0) if all_xy else np.zeros((0, 2)),
        "contour_kappa": np.concatenate(all_kappa) if all_kappa else np.zeros((0,)),
        "contour_normal": np.concatenate(all_normal, axis=0) if all_normal else np.zeros((0, 2)),
        "contour_id": np.concatenate(all_id) if all_id else np.zeros((0,), dtype=int),
        "contour_length": np.asarray(lengths, dtype=float),
        # metadata entries (non-array -> JSON sidecar)
        "shape": tuple(field.shape),
        "pixel_size": pixel_size,
        "level": level,
        "n_contours": len(results),
        "contour_fn": fn_name,
        **{f"contour_fn_{k}": _to_jsonable(v) for k, v in settings.items()},
    }
    if slice_axis is not None:
        field_dict["slice_axis"] = slice_axis
    if slice_index is not None:
        field_dict["slice_index"] = int(slice_index)
    return field_dict



#############################################################################

def contour_curvature_evenspacing(field, level=0.0, smoothing=0.0, n_eval=400,
                      point_spacing=None, min_eval=16, uniform_arclength=True,
                      min_points=8, pixel_size=8.0, spline_order = 3):
    """
    Extract the domain boundary/boundaries of `field` at `level` and
    compute signed curvature along each.

    Parameters
    ----------
    field : (H, W) array
        Scalar field, e.g. your Mz array.
    level : float, default 0.0
        Threshold defining the boundary.
    smoothing : float, default 2.0
        Spline smoothing strength (s = smoothing * n_points_in_contour).
    n_eval : int, default 400
        Fixed number of evaluation points per contour. Only used when
        `point_spacing` is None.
    point_spacing : float or None, default None
        Desired spacing between evaluation points, in physical units
        (same units as `pixel_size`, e.g. nm). If given, the number of
        points per contour is  round(arc_length / point_spacing) + 1,
        so it scales with each contour's length. Overrides `n_eval`.
    min_eval : int, default 16
        Lower bound on the number of points when `point_spacing` is used,
        so very short contours still get a usable curvature profile.
    uniform_arclength : bool, default True
        If True (and `point_spacing` is given), points are equally spaced
        in arc length rather than in the spline parameter.
    min_points : int, default 8
        Contours with fewer raw points than this are skipped.
    pixel_size : float, default 1.0
        Physical length per pixel. Assumes square pixels.

    Returns
    -------
    list of dicts, one per contour, each with:
        'xy'     : (n, 2) coordinates in physical units
        'kappa'  : (n,) signed curvature, in 1/[pixel_size units]
        'normal' : (n, 2) unit normals
        'length' : total arc length of the fitted spline, physical units
    where n = n_eval, or the arc-length-derived count if point_spacing is set.
    """
    contours = find_contours(field, level=level)
    results = []

    for c in contours:
        if len(c) < min_points:
            continue
        rows, cols = c[:, 0], c[:, 1]
        closed = np.hypot(rows[0] - rows[-1], cols[0] - cols[-1]) < 1.0
        per = 1 if closed else 0
        try:

            ## define a spline of the curve (for each contour, input x y values and fit with a cubic (3) spline)
            ##  a spline is a function defined in pieces by polynomials that stitch together neighboring points smoothly

            tck, _ = splprep([cols, rows], s=smoothing * len(c), per=per, k=spline_order)
        except Exception:
            continue

        # --- arc length of the fitted spline (dense evaluation) ---
        u_dense = np.linspace(0, 1, max(2000, 10 * len(c))) ## segments points on the contour from 0 being the beginning of the line and 1 being the end of the line. n_eval determined by voxel size
        xd, yd = splev(u_dense, tck, der=0) ## evaluated spline and get x y positions
        s_cum = np.r_[0, np.cumsum(np.hypot(np.diff(xd), np.diff(yd)))]  # pixels on the arc
        length_px = s_cum[-1]
        length_phys = length_px * pixel_size

        # --- choose evaluation parameters ---
        if point_spacing is None:
            n = n_eval
            uu = np.linspace(0, 1, n)
        else:
            n = max(min_eval, int(round(length_phys / point_spacing)) + 1)
            if uniform_arclength:
                # invert s(u): find u values at equally spaced arc lengths based on voxel size
                uu = np.interp(np.linspace(0, length_px, n), s_cum, u_dense)
            else:
                uu = np.linspace(0, 1, n)

        x, y = splev(uu, tck, der=0)
        dx, dy = splev(uu, tck, der=1) ## first derivative of the spline -----> This is the unit tangent. Rotate this 90 deg. to get the normal 
        ddx, ddy = splev(uu, tck, der=2) ## second derivative of the spline

        # unit tangent and unit normal (x = column, y = row)
        speed = np.hypot(dx, dy)
        speed = np.where(speed == 0, 1e-12, speed)
        tx, ty = dx / speed, dy / speed
        nx, ny = -ty, tx                     # tangent rotated +90 degrees == the surface normal 

        ## plane curve definition of curvature (https://en.wikipedia.org/wiki/Curvature)
        denom = (dx**2 + dy**2) ** 1.5
        # convert pixel-index-space quantities to physical units
        kappa = (dx * ddy - dy * ddx) / np.where(denom == 0, 1e-12, denom)
        kappa = kappa / pixel_size
        xy = np.stack([x, y], axis=1) * pixel_size

        results.append({
            "xy": xy,
            "kappa": kappa,
            "normal": np.stack([nx, ny], axis=1),
            "length": length_phys,
        })
    return results



def plot_domain_curvature_evenspace(image, field=None, contour_fn=None, level=0.0,
                          smoothing=0.0, point_spacing=None, pixel_size=8.0,
                          vmax_percentile=68, unit_label="nm",
                          cmap="coolwarm", image_cmap="viridis", image_clim=None,
                          linewidth=2.5, show_points=False, ax=None, origin = 'lower'):
    """
    Overlay the evenly spaced curvature spline on an image.

    Parameters
    ----------
    image : (H, W) array
        Background image to display (e.g. Mx).
    field : (H, W) array or None
        Field the domain boundary is extracted from (e.g. Mz). If None,
        the contour is extracted from `image` itself.
    contour_fn : callable or None
        Function taking a 2D field and returning the list of dicts from
        `contour_curvature` (e.g. your `contour_fn_3`). If given, `level`,
        `smoothing`, and `point_spacing` are ignored, since the function
        carries its own settings. If None, `contour_curvature` is called
        with those arguments.
    point_spacing : float or None
        Spacing between points in physical units. Defaults to `pixel_size`.
    vmax_percentile : float
        Colour scale is symmetric, +/- this percentile of |kappa| over all
        contours (clips outliers so typical curvature stays visible).
    unit_label : str
        Length unit for axis and colorbar labels.
    show_points : bool
        Draw a dot at every evaluation point to visualise the spacing.
    ax : matplotlib Axes or None
        Draw into an existing axes if given.

    Returns
    -------
    fig, ax, results
        `results` is the list of contour dicts (xy, kappa, normal, length).
    """
    if field is None:
        field = image

    # --- get contours + curvature ---
    if contour_fn is not None:
        results = contour_fn(field)
    else:
        results = contour_curvature_evenspacing(
            field, level=level, smoothing=smoothing,
            point_spacing=point_spacing if point_spacing is not None else pixel_size,
            pixel_size=pixel_size)

    if len(results) == 0:
        raise ValueError("No contours found. Check `level` and `min_points`.")

    # --- symmetric colour scale from all curvature values ---
    all_k = np.concatenate([r["kappa"] for r in results])
    vmax = np.nanpercentile(np.abs(all_k), vmax_percentile)
    norm = Normalize(vmin=-vmax, vmax=vmax)

    # --- figure / background image ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # H, W = image.shape
    # extent = [0, W * pixel_size, 0, H * pixel_size]

    H, W = image.shape
    x0, x1 = -0.5 * pixel_size, (W - 0.5) * pixel_size
    y0, y1 = -0.5 * pixel_size, (H - 0.5) * pixel_size
    extent = [x0, x1, y1, y0] if origin == "upper" else [x0, x1, y0, y1]

    imshow_kw = {}
    if image_clim is not None:
        imshow_kw["vmin"], imshow_kw["vmax"] = image_clim
    #ax.imshow(image, cmap=image_cmap, alpha = 0.5, origin="lower", extent=extent, **imshow_kw)
    ax.imshow(image, cmap=image_cmap, alpha=0.5, origin=origin, extent=extent, **imshow_kw)
    # --- curvature-coloured spline segments ---
    lc = None
    for r in results:
        xy, k = r["xy"], r["kappa"]
        segs = np.stack([xy[:-1], xy[1:]], axis=1)   # (n-1, 2, 2)
        k_seg = 0.5 * (k[:-1] + k[1:])               # colour by segment midpoint
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=linewidth)
        lc.set_array(k_seg)
        ax.add_collection(lc)
        if show_points:
            ax.scatter(xy[:, 0], xy[:, 1], c=k, cmap=cmap, norm=norm,
                       s=8, linewidths=0.3, zorder=3)

    cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04, extend="both")
    cbar.set_label(f"Signed curvature (1/{unit_label})")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({unit_label})")
    ax.set_ylabel(f"y ({unit_label})")
    return fig, ax, results


from scipy.ndimage import gaussian_filter, map_coordinates
def compute_profiles(gradient, xy, normal, pixel_size, half_width, n_samples=81, order=3):
    """
    Sample `img` along the normal at each point of a contour.

    img        : 2D array (e.g. gradient magnitude), indexed [row, col]
    xy         : (N, 2) physical positions (x=col, y=row) * pixel_size
    normal     : (N, 2) unit normals (nx, ny)
    half_width : physical distance to sample on each side of the contour
    Returns
    s          : (n_samples,) signed physical distance along the normal
    profiles   : (N, n_samples) sampled values
    XY         : (N, n_samples, 2) physical sample positions

    """
    s = np.linspace(-half_width, half_width, n_samples)

    Xp = xy[:, 0][:, None] + s[None, :] * normal[:, 0][:, None]   # physical
    Yp = xy[:, 1][:, None] + s[None, :] * normal[:, 1][:, None]

    # map_coordinates wants pixel-index (row, col) = (Y, X)
    profiles = map_coordinates(gradient, [Yp / pixel_size, Xp / pixel_size],
                               order=order, mode="nearest")
    return s, profiles, np.stack([Xp, Yp], axis=-1)


####################### fwhm for dw width  #######################

from scipy.optimize import curve_fit


def _sech(x):
    return 1 / np.cosh(x)

MODELS = {
    # name: (func, extra_p0, extra_lower, extra_upper)
    "gauss":   (lambda s, A, mu, w, off: A*np.exp(-0.5*((s-mu)/w)**2) + off, [], [], []),
    "sech2":   (lambda s, A, mu, w, off: A*_sech((s-mu)/w)**2 + off,         [], [], []),
    "sech":    (lambda s, A, mu, w, off: A*_sech((s-mu)/w) + off,            [], [], []),

}

# FWHM = factor * w   (for the Gaussian, w is sigma)
FWHM_FACTOR = {"gauss": 2*np.sqrt(2*np.log(2)),      # 2.355
               "sech":  2*np.arccosh(2),             # 2.634
               "sech2": 2*np.arccosh(np.sqrt(2))}    # 1.763
LABEL = {"gauss": "Gaussian", "sech": "sech", "sech2": "sech²"}


def fit_peak(s, prof, model="gauss"):
    """Fit A*shape((s-mu)/w) + offset. Returns the same keys for every model."""
    func, k = MODELS[model][0], FWHM_FACTOR[model]
    nan = np.nan
    out = dict(success=False, A=nan, mu=nan, w=nan, offset=nan, fwhm=nan, rms=nan,
               mu_err=nan, w_err=nan, fwhm_err=nan)
    ok = np.isfinite(prof)
    if ok.sum() < 6:
        return out
    s_, p_ = s[ok], prof[ok]
    ds = s_[1] - s_[0]
    j = np.argmax(p_)
    off0 = p_.min()
    A0 = p_[j] - off0
    above = p_ > off0 + A0 / 2
    width0 = s_[above].max() - s_[above].min() if above.sum() > 1 else 4 * ds
    w0 = max(width0 / k, ds)                      # start from the half-max width

    try:
        popt, pcov = curve_fit(func, s_, p_, p0=[A0, s_[j], w0, off0],
                               bounds=([0, s_[0], ds/2, -np.inf],
                                       [np.inf, s_[-1], s_[-1] - s_[0], np.inf]),
                               maxfev=20000)
    except (RuntimeError, ValueError):
        return out
    perr = np.sqrt(np.diag(pcov))
    out.update(success=True, A=popt[0], mu=popt[1], w=popt[2], offset=popt[3],
               fwhm=k * popt[2], mu_err=perr[1], w_err=perr[2], fwhm_err=k * perr[2],
               rms=np.sqrt(np.mean((p_ - func(s_, *popt)) ** 2)), f=func, popt=popt)
    return out


def fwhm_from_curve(s, y, baseline):
    """Half-max width by linear interpolation of the crossings around the peak."""
    j = np.nanargmax(y)
    half = baseline + (y[j] - baseline) / 2
    l = j
    while l > 0 and y[l] > half:
        l -= 1
    r = j
    while r < len(y) - 1 and y[r] > half:
        r += 1
    if y[l] > half or y[r] > half:        # never dropped below half max
        return np.nan, np.nan, np.nan
    sl = s[l] + (half - y[l]) * (s[l+1] - s[l]) / (y[l+1] - y[l])
    sr = s[r-1] + (half - y[r-1]) * (s[r] - s[r-1]) / (y[r] - y[r-1])
    return sr - sl, sl, sr


def fwhm_along_contour(gradient, result, pixel_size, half_width,
                       n_samples=81, order=3, model="gauss"):
    """Fit `model` to the normal profile at every point of one contour."""
    s, profiles, XY = compute_profiles(gradient, result["xy"], result["normal"],
                                       pixel_size, half_width, n_samples, order)
    N = len(profiles)
    ny, nx = gradient.shape
    inside = ((XY[..., 0] >= 0) & (XY[..., 0] <= (nx - 1) * pixel_size) &
              (XY[..., 1] >= 0) & (XY[..., 1] <= (ny - 1) * pixel_size)).all(axis=1)

    keys = ["fwhm", "w", "mu", "A", "offset", "rms", "fwhm_err", "mu_err", "w_err"]
    out = {k: np.full(N, np.nan) for k in keys}
    out["fwhm_direct"] = np.full(N, np.nan)
    out["success"] = np.zeros(N, bool)

    for j, p in enumerate(profiles):
        if not (inside[j] and np.all(np.isfinite(p))):
            continue
        out["fwhm_direct"][j] = fwhm_from_curve(s, p, p.min())[0]   # model-free
        f = fit_peak(s, p, model)
        if f["success"]:
            out["success"][j] = True
            for k in keys:
                out[k][j] = f[k]

    out.update(inside=inside, pt=np.arange(N), half_width=half_width, model=model)
    return out


def fwhm_slice(grad2d, contours, pixel_size, half_width, min_good=20, model = 'gauss'):
    """Fit every contour in one slice and pool the good FWHM values."""
    vals, cmeans, used = [], [], []
    n_pts = 0
    for ci, r in enumerate(contours):
        fw = fwhm_along_contour(grad2d, r, pixel_size, half_width, model = model)
        good = fw["success"] & fw["inside"] & (fw["fwhm"] < fw["half_width"])
        n_pts += len(good)
        if good.sum() < min_good:          # skip tiny / mostly-rejected contours
            continue
        v = fw["fwhm"][good]
        vals.append(v)
        cmeans.append(v.mean())
        used.append(dict(idx=ci, xy=r["xy"], mean=v.mean(), n_good=int(good.sum())))

    v = np.concatenate(vals) if vals else np.array([])
    cm = np.array(cmeans)
    sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else np.nan
    return dict(
        n_good=len(v), n_pts=n_pts, n_contours=len(used),
        mean=v.mean() if len(v) else np.nan,
        median=np.median(v) if len(v) else np.nan,
        std=sd(v),
        stderr=sd(v) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
        p16=np.percentile(v, 16) if len(v) else np.nan,
        p84=np.percentile(v, 84) if len(v) else np.nan,
        vals=v, contour_means=cm, contours=used)



def fwhm_vs_z_slices(Mz3, grad3, contour_fn, pixel_size, half_width,
                     z_indices=None, min_good=20, model = 'gauss'):
    """Whole-slice FWHM for each z index. No tracking: every contour in the slice is used."""
    z_indices = range(len(Mz3)) if z_indices is None else z_indices
    rows = {}
    for k in z_indices:
        contours = contour_fn(Mz3[k])
        rows[k] = fwhm_slice(grad3[k], contours, pixel_size, half_width, min_good, model = model)
        r = rows[k]
        print(f"z={k}: {r['n_contours']} contours, {r['n_good']}/{r['n_pts']} points used, "
              f"mean {r['mean']:.1f}, median {r['median']:.1f}")
    return rows


def plot_normal_cross_section(field, gradient, results, contour_idx, point_idx,
                              pixel_size, half_width, n_samples=81, order=3,
                              cmap="magma", fit=True, model="gauss", unit="nm",
                              detailed_label=False):
    """
    Panel 1: gradient map, full spline, and the selected normal (extent = what's profiled).
    Panel 2: profile of `gradient` along that normal, with a `model` fit and its residual.
    model: any key of MODELS ("gauss", "sech", "sech2").
    """
    r = results[contour_idx]
    xy, normal = r["xy"], r["normal"]
    i = point_idx % len(xy)                      # allows negative indices

    # profile at just the selected point
    s, prof, XY = compute_profiles(gradient, xy[i:i+1], normal[i:i+1],
                                   pixel_size, half_width, n_samples, order)
    prof, XY = prof[0], XY[0]

    f = fit_peak(s, prof, model) if fit else None
    resid = np.full_like(prof, np.nan)           # defined even if the fit fails or is off

    nrows, ncols = field.shape
    extent = (-0.5 * pixel_size, (ncols - 0.5) * pixel_size,
              (nrows - 0.5) * pixel_size, -0.5 * pixel_size)   # origin="upper"

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # panel 1: gradient map + spline + selected normal
    im = ax0.imshow(gradient, extent=extent, origin="upper", cmap=cmap)
    ax0.plot(xy[:, 0], xy[:, 1], "w-", lw=1.5, label="spline")
    ax0.plot(XY[:, 0], XY[:, 1], "r-", lw=2, label="profile")
    ax0.plot(xy[i, 0], xy[i, 1], "ro", ms=6, mec="w")
    ax0.set_xlabel(f"x ({unit})"); ax0.set_ylabel(f"y ({unit})")
    ax0.set_title(f"contour {contour_idx}, point {i}")
    ax0.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # panel 2: profile + fit
    ax1.plot(s, prof, "k.-", label="data")
    ax1.axvline(0, color="r", ls="--", lw=1, label="DW center")

    if f is not None and f["success"]:
        sf = np.linspace(s[0], s[-1], 500)
        label = f"{LABEL[model]} fit\nFWHM = {f['fwhm']:.3g} {unit}"
        if detailed_label:
            wname = "σ" if model == "gauss" else "w"
            label += (f"\nmean = {f['mu']:.3g} ± {f['mu_err']:.2g}"
                      f"\n{wname} = {f['w']:.3g} ± {f['w_err']:.2g}"
                      f"\nA = {f['A']:.3g}, offset = {f['offset']:.3g}")
        ax1.plot(sf, f["f"](sf, *f["popt"]), "b-", lw=2, label=label)

        # FWHM marker at half maximum of the fitted curve (valid for all three shapes)
        half = f["offset"] + f["A"] / 2
        ax1.hlines(half, f["mu"] - f["fwhm"] / 2, f["mu"] + f["fwhm"] / 2,
                   color="tab:green", lw=2)

        # residual at the sampled points
        resid = prof - f["f"](s, *f["popt"])
        ax1.axhline(0, color="gray", lw=1)
        ax1.plot(s, resid, "k.-", lw=0.8, alpha=0.5, label="residual")
    elif fit:
        ax1.text(0.05, 0.9, "fit failed", transform=ax1.transAxes, color="r")

    ax1.set_xlabel(f"distance along normal ({unit})")
    ax1.set_ylabel("gradient")
    ax1.set_title("cross-section along normal")
    ax1.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    return fig, (ax0, ax1), s, prof, f, resid

def plot_fwhm_along_contour(field, result, fw, contour_idx, pixel_size, half_width, model = 'gauss', 
                            mark_pt=None, unit="nm", ):
    # quality mask: fit worked, profile inside image, FWHM not truncated by the window
    good = fw["success"] & fw["inside"] & (fw["fwhm"] < fw["half_width"])
    xy = result["xy"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.5),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # left: contour colored by fitted FWHM
    ny, nx = field.shape
    extent = (-0.5 * pixel_size, (nx - 0.5) * pixel_size,
              (ny - 0.5) * pixel_size, -0.5 * pixel_size)
    ax0.imshow(field, extent=extent, origin="upper", cmap="viridis")
    sc = ax0.scatter(xy[good, 0], xy[good, 1], c=fw["fwhm"][good], s=8, cmap='bwr')
    ax0.plot(xy[~good, 0], xy[~good, 1], ".", c="k", ms=3, label="rejected")
    if mark_pt is not None:
        ax0.plot(*xy[mark_pt], "co", ms=8, mfc="none")
    fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.04, label=f"FWHM ({unit})")
    ax0.set_xlabel(f"x ({unit})"); ax0.set_ylabel(f"y ({unit})")
    ax0.set_title(f"contour {contour_idx}"); ax0.legend(fontsize=8)

    # right: FWHM vs point index
    #ax1.errorbar(fw["pt"][good], fw["fwhm"][good], yerr= fw["fwhm_err"][good], color = "b", linestyle = ".", label=f"{model} fit")
    ax1.errorbar(fw["pt"][good], fw["fwhm"][good], yerr=fw["fwhm_err"][good], color="b", marker="o", fmt = ' ', label=f"{model} fit")
    ax1.axhline(np.mean(fw["fwhm"][good]), color = "b", label = f"avg dw width = {np.mean(fw["fwhm"][good]):.0f} nm")
    # ax1.plot(fw["pt"][good], fw["fwhm_direct"][good], ".", c="orange", ms=4,
    #          label="direct (model-free)")
    # ax1.errorbar(fw["pt"][~good], fw["fwhm"][~good], yerr= fw["fwhm_err"][~good], color ="r", marker = "o", fmt = ' ', ms=4, label="double peak")
    ax1.plot(fw["pt"][~good], fw["fwhm"][~good], "rx", ms=4, label="double peak")
    if mark_pt is not None:
        ax1.axvline(mark_pt, c="c", ls=":")
    ax1.set_xlabel("point on spline"); ax1.set_ylabel(f"FWHM ({unit})")
    ax1.set_title(f"FWHM along contour for profile length = {2* half_width} nm"); 
    ax1.legend(fontsize=10, loc = 'upper right')
    fig.tight_layout()
    return fig, (ax0, ax1)


from scipy.spatial import cKDTree

def overlap_score(xyA, xyB, tol):
    """Fraction of points of each curve lying within tol of the other (1 = same curve)."""
    dAB = cKDTree(xyB).query(xyA)[0]
    dBA = cKDTree(xyA).query(xyB)[0]
    return 0.5 * ((dAB < tol).mean() + (dBA < tol).mean())


def find_match(ref_xy, contours, tol, min_overlap=0.3):
    if not contours:
        return None, 0.0
    scores = [overlap_score(ref_xy, c["xy"], tol) for c in contours]
    j = int(np.argmax(scores))
    return (j, scores[j]) if scores[j] >= min_overlap else (None, scores[j])


def fwhm_vs_z(Mz3, grad3, contour_fn, z_ref, ref_result, pixel_size, half_width,
              tol, min_overlap=0.3, common_tol=None, model = 'gauss'):
    """
    Follow one wall through z and summarize its FWHM in each slice.
    contour_fn  : function(Mz_slice) -> list of contour dicts
    ref_result  : the contour dict at z_ref (e.g. results[6])
    tol         : max wall displacement between adjacent slices (same units as xy)
    common_tol  : if set, only use points within this distance of the z_ref contour,
                  so every slice averages over roughly the same stretch of wall
    """
    nz = len(Mz3)
    rows, tracked = {}, {}

    def summarize(k, r, score):
        fw = fwhm_along_contour(grad3[k], r, pixel_size, half_width, model = model)
        good = fw["success"] & fw["inside"] & (fw["fwhm"] < fw["half_width"])
        if common_tol is not None:
            good &= cKDTree(ref_result["xy"]).query(r["xy"])[0] < common_tol
        v = fw["fwhm"][good]
        rows[k] = dict(
            n_good=int(good.sum()), n_pts=len(good), overlap=score,
            mean=np.mean(v) if len(v) else np.nan,
            median=np.median(v) if len(v) else np.nan,
            std=np.std(v, ddof=1) if len(v) > 1 else np.nan,
            stderr = np.std(v, ddof=1)/np.sqrt(len(v)) if len(v) > 1 else np.nan,
            vals=v.copy(),
            p16=np.percentile(v, 16) if len(v) else np.nan,
            p84=np.percentile(v, 84) if len(v) else np.nan,
            direct=np.nanmedian(fw["fwhm_direct"][good]) if len(v) else np.nan)
        tracked[k] = r

    summarize(z_ref, ref_result, 1.0)

    for direction in (+1, -1):                    # walk up, then down, from z_ref
        ref_xy = ref_result["xy"]
        k = z_ref + direction
        while 0 <= k < nz:
            contours = contour_fn(Mz3[k])
            j, score = find_match(ref_xy, contours, tol, min_overlap)
            if j is None:
                print(f"lost the wall at z={k} (best overlap {score:.2f})")
                break
            summarize(k, contours[j], score)
            ref_xy = contours[j]["xy"]            # chain: compare with the previous slice
            k += direction
    return rows, tracked

def plot_fwhm_vs_z_nobin(Mz3, rows, tracked, z_ref, dz_nm=None, pixel_size=1.0, unit="nm", model = 'gauss'):
    zs = np.array(sorted(rows))
    med = np.array([rows[k]["median"] for k in zs])
    lo = np.array([rows[k]["p16"] for k in zs])
    hi = np.array([rows[k]["p84"] for k in zs])
    mean = np.array([rows[k]["mean"] for k in zs])
    n = np.array([rows[k]["n_good"] for k in zs])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})

    # left: check that the tracking followed the same wall
    ny, nx = Mz3[z_ref].shape
    extent = (-0.5 * pixel_size, (nx - 0.5) * pixel_size, (ny - 0.5) * pixel_size, -0.5 * pixel_size)
    ax0.imshow(Mz3[z_ref], extent=extent, origin="upper", cmap="gray")
    cmap = plt.get_cmap("viridis")
    for k in zs:
        xy = tracked[k]["xy"]
        ax0.plot(xy[:, 0], xy[:, 1], "-", lw=1.2, color=cmap((k - zs[0]) / max(1, zs[-1] - zs[0])))
    sm = plt.cm.ScalarMappable(plt.Normalize(zs[0], zs[-1]), "viridis")
    fig.colorbar(sm, ax=ax0, fraction=0.046, pad=0.04, label="z index")
    ax0.set_title(f"tracked wall (background: Mz at z={z_ref})")
    ax0.set_xlabel(f"x ({unit})"); ax0.set_ylabel(f"y ({unit})")

    # right: FWHM vs z
    ax1.errorbar(zs, med, yerr=[med - lo, hi - med], fmt="bo-", capsize=3, label="median (16-84%)")
    ax1.plot(zs, mean, "s", c="c", ms=4, label="mean")
    ax1.axvline(z_ref, c="gray", ls=":", label="reference slice")
    for k, nn in zip(zs, n):
        ax1.annotate(str(nn), (k, ax1.get_ylim()[0]), ha="center", va="bottom", fontsize=7, color="gray")
    ax1.set_xlabel("z index"); ax1.set_ylabel(f"FWHM of {model} fit ({unit})")
    ax1.set_title("FWHM vs. z (numbers = points used)")
    if dz_nm:
        sec = ax1.secondary_xaxis("top", functions=(lambda z: z * dz_nm, lambda z: z / dz_nm))
        sec.set_xlabel(f"z ({unit})")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    return fig, (ax0, ax1)


def bin_fwhm_by_z(rows, dz_nm, bin_nm=8.0):
    """
    Pool the per-point FWHM values of all slices falling in each z bin.
    Bins start at 0 nm: [0, 8), [8, 16), ...  Slice k is placed at z = k*dz_nm.
    """
    zs = np.array(sorted(rows))
    bins = np.floor(zs * dz_nm / bin_nm + 1e-9).astype(int)
    out = []
    for b in np.unique(bins):
        ks = zs[bins == b]
        v = np.concatenate([rows[k]["vals"] for k in ks])
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        out.append(dict(bin=b, center=(b + 0.5) * bin_nm, slices=ks, n=len(v),
                        mean=v.mean(), median=np.median(v),
                        std=v.std(ddof=1) if len(v) > 1 else np.nan,  stderr=v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else np.nan  ))
    return out


def plot_fwhm_vs_z_bin_onecontour(Mz3, rows, tracked, z_ref, dz_nm, pixel_size=1.0, unit="nm", bin_nm=8.0, model = 'gauss', plot_spline = False):
    zs = np.array(sorted(rows))
    binned = bin_fwhm_by_z(rows, dz_nm, bin_nm)
    xc   = np.array([d["center"] for d in binned])
    mean = np.array([d["mean"] for d in binned])
    med  = np.array([d["median"] for d in binned])
    std  = np.array([d["std"] for d in binned])
    stderr  = np.array([d["stderr"] for d in binned])
    n    = np.array([d["n"] for d in binned])


    if plot_spline == True: 
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.8),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

        # left: check that the tracking followed the same wall
        ny, nx = Mz3[z_ref].shape
        extent = (-0.5 * pixel_size, (nx - 0.5) * pixel_size,
                (ny - 0.5) * pixel_size, -0.5 * pixel_size)
        ax0.imshow(Mz3[z_ref], extent=extent, origin="upper", cmap="gray")
        cmap = plt.get_cmap("viridis")
        for k in zs:
            xy = tracked[k]["xy"]
            ax0.plot(xy[:, 0], xy[:, 1], "-", lw=1.2,
                    color=cmap((k - zs[0]) / max(1, zs[-1] - zs[0])))
        sm = plt.cm.ScalarMappable(plt.Normalize(zs[0] * dz_nm, zs[-1] * dz_nm), "viridis")
        fig.colorbar(sm, ax=ax0, fraction=0.046, pad=0.04, label=f"z ({unit})")
        ax0.set_title(f"tracked wall (background: Mz at z={z_ref})")
        ax0.set_xlabel(f"x ({unit})"); ax0.set_ylabel(f"y ({unit})")

        # right: binned FWHM vs z (mean and median offset slightly so error bars don't overlap)
        off = 0.06 * bin_nm
        ax1.errorbar(xc, mean, yerr=stderr, fmt="bs", capsize=4, ms=6, label="mean ± stderr")
        # ax1.errorbar(xc + off, med, yerr=stderr, fmt="o", c="orange", mfc="none", capsize=4, ms=6,
        #              label="median ± stderr")
        # ax1.axvline(z_ref * dz_nm, c="gray", ls=":", label="reference slice")
        # for x, nn in zip(xc, n):
        #     ax1.annotate(f"n={nn}", (x, 0.02), xycoords=("data", "axes fraction"),
        #                  ha="center", va="bottom", fontsize=8, color="gray")
        ax1.set_xticks(xc)
        ax1.set_xticklabels([f"{x - bin_nm/2:.0f}-{x + bin_nm/2:.0f}" for x in xc])
        ax1.set_xlabel(f"z bin ({unit})")
        ax1.set_ylabel(f"FWHM of {model} fit ({unit})")
        ax1.set_title(f"FWHM vs. z, {bin_nm:g} nm bins")
        ax1.legend(fontsize=8)
        fig.tight_layout()


    else: 

        fig, (ax1) = plt.subplots(1, 1, figsize=(7, 4.8))

        # left: check that the tracking followed the same wall
        ny, nx = Mz3[z_ref].shape
        
        # right: binned FWHM vs z (mean and median offset slightly so error bars don't overlap)
        off = 0.06 * bin_nm
        ax1.errorbar(xc, mean, yerr=stderr, fmt="bs", capsize=4, ms=6, label="mean ± stderr")
        # ax1.errorbar(xc + off, med, yerr=stderr, fmt="o", c="orange", mfc="none", capsize=4, ms=6,
        #              label="median ± stderr")
        # ax1.axvline(z_ref * dz_nm, c="gray", ls=":", label="reference slice")
        # for x, nn in zip(xc, n):
        #     ax1.annotate(f"n={nn}", (x, 0.02), xycoords=("data", "axes fraction"),
        #                  ha="center", va="bottom", fontsize=8, color="gray")
        ax1.set_xticks(xc)
        ax1.set_xticklabels([f"{x - bin_nm/2:.0f}-{x + bin_nm/2:.0f}" for x in xc])
        ax1.set_xlabel(f"z bin ({unit})")
        ax1.set_ylabel(f"FWHM ({unit})")
        ax1.set_title(f"FWHM of {model} fit vs. z, {bin_nm:g} nm bins \n one contour")
        ax1.legend(fontsize=8)
        fig.tight_layout()

    return fig, binned



def bin_slice_fwhm_by_z(rows, dz_nm, bin_nm=8.0):
    zs = np.array(sorted(rows))
    bins = zs if bin_nm is None else np.floor(zs * dz_nm / bin_nm + 1e-9).astype(int)
    sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else np.nan
    out = []
    for b in np.unique(bins):
        ks = zs[bins == b]
        v = np.concatenate([rows[k]["vals"] for k in ks])
        v = v[np.isfinite(v)]
        cm = np.concatenate([rows[k]["contour_means"] for k in ks])
        if len(v) == 0:
            continue
        out.append(dict(
            center=ks[0] * dz_nm if bin_nm is None else (b + 0.5) * bin_nm,
            slices=ks, n=len(v), n_contours=len(cm),
            mean=v.mean(), median=np.median(v), std=sd(v),
            stderr=sd(v) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
            stderr_contour=sd(cm) / np.sqrt(len(cm)) if len(cm) > 1 else np.nan))
    return out


def plot_fwhm_vs_z_slices(Mz3, rows, z_ref, dz_nm, pixel_size=1.0, unit="nm",
                          bin_nm=8.0, err="stderr", show_median=False, cmap="viridis", model = 'gauss'):
    """err: 'std' | 'stderr' (std/sqrt(points)) | 'stderr_contour' (std of contour means/sqrt(contours))"""
    binned = bin_slice_fwhm_by_z(rows, dz_nm, bin_nm)
    xc = np.array([d["center"] for d in binned])
    mean = np.array([d["mean"] for d in binned])
    med = np.array([d["median"] for d in binned])
    e = np.array([d[err] for d in binned])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    # left: every contour in the reference slice, colored by its mean FWHM
    ny, nx = Mz3[z_ref].shape
    extent = (-0.5 * pixel_size, (nx - 0.5) * pixel_size,
              (ny - 0.5) * pixel_size, -0.5 * pixel_size)
    ax0.imshow(Mz3[z_ref], extent=extent, origin="upper", cmap="gray")
    cs = rows[z_ref]["contours"]
    allm = [c["mean"] for c in cs]
    if allm:
        vmin, vmax = min(allm), max(allm)
        for c in cs:
            sc = ax0.scatter(c["xy"][:, 0], c["xy"][:, 1], c=np.full(len(c["xy"]), c["mean"]),
                             vmin=vmin, vmax=vmax, s=3, cmap=cmap)
        fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.04, label=f"contour mean FWHM ({unit})")
    ax0.set_title(f"contours used at z={z_ref} ({rows[z_ref]['n_contours']})")
    ax0.set_xlabel(f"x ({unit})"); ax0.set_ylabel(f"y ({unit})")

    # right: whole-slice FWHM vs z
    ax1.errorbar(xc, mean, yerr=e, fmt="bs-" if bin_nm is None else "bs",
                 capsize=4, ms=6, label=f"mean ± {err}")
    if show_median:
        ax1.errorbar(xc, med, yerr=e, fmt="o", c="orange", mfc="none", capsize=4, ms=6,
                     label=f"median ± {err}")
    if bin_nm is not None:
        ax1.set_xticks(xc)
        ax1.set_xticklabels([f"{x - bin_nm/2:.0f}-{x + bin_nm/2:.0f}" for x in xc])
        ax1.set_xlabel(f"z bin ({unit})")
        ax1.set_title(f"FWHM of {model}  vs. z, all contours, {bin_nm:g} nm bins")
    else:
        ax1.set_xlabel(f"z ({unit})")
        ax1.set_title(f"FWHM of {model} fit vs. z, all contours")
    ax1.set_ylabel(f"FWHM ({unit})")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    return fig, (ax0, ax1), binned



############################################# Calculate polar and azimuthal angle #############################################
from matplotlib.gridspec import GridSpec

def sample_angles(Mx, My, Mz, xy, normal, pixel_size, half_width, n_samples=161, order=3):
    """theta, phi, |m| along every normal; arrays have shape (N, n_samples)."""
    s, mx, _ = compute_profiles(Mx, xy, normal, pixel_size, half_width, n_samples, order)
    _, my, _ = compute_profiles(My, xy, normal, pixel_size, half_width, n_samples, order)
    _, mz, _ = compute_profiles(Mz, xy, normal, pixel_size, half_width, n_samples, order)
    norm = np.sqrt(mx**2 + my**2 + mz**2)
    theta = np.arccos(np.clip(mz / norm, -1, 1))
    phi = np.arctan2(my/ norm, mx/ norm)
    return s, theta, phi, norm


def theta_model(s, mu, D, sgn):
    return 2 * np.arctan(np.exp(np.clip(sgn * (s - mu) / D, -100, 100)))


def fit_theta(s, theta):
    out = dict(success=False, mu=np.nan, D=np.nan, mu_err=np.nan, D_err=np.nan,
               sgn=np.nan, rms=np.nan)
    ds = s[1] - s[0]
    sgn = 1.0 if theta[-1] > theta[0] else -1.0      # theta goes 0->pi or pi->0 along +s
    mu0 = s[np.argmin(np.abs(theta - np.pi / 2))]
    f = lambda s, mu, D: theta_model(s, mu, D, sgn)
    try:
        popt, pcov = curve_fit(f, s, theta, p0=[mu0, 4 * ds],
                               bounds=([s[0], ds / 2], [s[-1], s[-1] - s[0]]))
    except (RuntimeError, ValueError):
        return out
    perr = np.sqrt(np.diag(pcov))
    out.update(success=True, mu=popt[0], D=popt[1], mu_err=perr[0], D_err=perr[1],
               sgn=sgn, rms=np.sqrt(np.mean((theta - f(s, *popt)) ** 2)), f=f)
    return out


def orient_normals(Mz, result, pixel_size, probe=4):
    """Flip the normal so +s points toward Mz>0. Safe to call twice."""
    xy, n = result["xy"], result["normal"]
    d = probe * pixel_size
    f = lambda p: map_coordinates(Mz, [p[:, 1] / pixel_size, p[:, 0] / pixel_size],
                                  order=1, mode="nearest")
    flip = 1.0 if np.nanmedian(f(xy + d * n) - f(xy - d * n)) >= 0 else -1.0
    result["normal"] = flip * n
    result["flip"] = result.get("flip", 1.0) * flip
    result["kappa_n"] = result["flip"] * result["kappa"]   # >0: wall bends toward +s


SQ2 = 1 / np.sqrt(2)
wrap = lambda a: np.angle(np.exp(1j * a))

def wall_profile(Mx, My, Mz, gradient, r, i, pixel_size, half_width,
                 n_samples=161, order=3, sin_min=0.2, model="gauss",
                 window=0.5, window_from="fit"):
    """
    Single source of truth for one normal profile (used by both plots).
    Enforces: +s points toward +Mz, so psi = +-180 is positive Neel (+Mz -> -Mz).
    """
    orient_normals(Mz, r, pixel_size)              # contour-wide orientation
    xy = r["xy"]
    sl = slice(i, i + 1)
    n_i = r["normal"][i:i + 1].copy()

    def sample(n_loc):
        s, th, ph, _ = sample_angles(Mx, My, Mz, xy[sl], n_loc, pixel_size,
                                     half_width, n_samples, order)
        _, g, XY = compute_profiles(gradient, xy[sl], n_loc, pixel_size,
                                    half_width, n_samples, order)
        return s, th[0], ph[0], g[0], XY[0]

    s, th, ph, g, XY = sample(n_i)

    # local enforcement: Mz must be larger on the +s side than on the -s side
    far = np.abs(s) > 0.5 * np.abs(s).max()
    mz = np.cos(th)
    flipped = bool(np.nanmean(mz[far & (s > 0)]) < np.nanmean(mz[far & (s < 0)]))
    if flipped:
        n_i = -n_i
        s, th, ph, g, XY = sample(n_i)

    alpha = np.arctan2(n_i[0, 1], n_i[0, 0])
    psi = wrap(ph - alpha)

    fit = fit_peak(s, g, model)
    if window_from == "fit" and fit["success"]:
        mu, fwhm = fit["mu"], fit["fwhm"]
    else:
        fwhm, cl, cr = fwhm_from_curve(s, g, np.nanmin(g))
        mu = 0.5 * (cl + cr)
    ok = np.isfinite(mu) and np.isfinite(fwhm)
    lo, hi = (mu - window * fwhm, mu + window * fwhm) if ok else (np.nan, np.nan)
    inwin = (s >= lo) & (s <= hi) if ok else np.zeros_like(s, bool)
    good = inwin & (np.sin(th) > sin_min)

    return dict(s=s, psi=psi, phi=ph, theta=th, grad=g, XY=XY, fit=fit, n_i=n_i,
                good=good, inwin=inwin, mu=mu, fwhm=fwhm, lo=lo, hi=hi,
                alpha=alpha, ok=ok, flipped=flipped)

def plot_angle_decomposition(Mx, My, Mz, gradient, contour_slice, contour_idx, point_idx,
                             pixel_size, half_width, n_samples=161, order=3,
                             cmap="magma", sin_min=0.2, unit="nm",
                             jac=None, k=None, model='gauss'):
    r = contour_slice[contour_idx]
    xy = r["xy"]
    i = point_idx % len(xy)
    sl = slice(i, i + 1)

    P = wall_profile(Mx, My, Mz, gradient, r, i, pixel_size, half_width,
                     n_samples=n_samples, order=order, sin_min=sin_min, model=model)
    s, th, ph, g, XY = P["s"], P["theta"], P["phi"], P["grad"], P["XY"]
    psi, alpha, f, n_i = P["psi"], P["alpha"], P["fit"], P["n_i"]
    ok = np.sin(th) > sin_min

    ft = fit_theta(s, th)

    nrows, ncols = Mz.shape
    extent = (-0.5 * pixel_size, (ncols - 0.5) * pixel_size,
              (nrows - 0.5) * pixel_size, -0.5 * pixel_size)

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.1, 1],
                  height_ratios=[1.3, 1.3, 1.6], hspace=0.1)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)

    # map
    im = ax0.imshow(gradient, extent=extent, origin="upper", cmap=cmap)
    ax0.plot(xy[:, 0], xy[:, 1], "w-", lw=1.5, label="spline")
    ax0.plot(XY[:, 0], XY[:, 1], "r-", lw=2, label="profile")
    ax0.plot(xy[i, 0], xy[i, 1], "ro", ms=6, mec="w")
    ax0.set_xlabel(f"x ({unit})"); ax0.set_ylabel(f"y ({unit})")
    ax0.set_title(f"contour {contour_idx}, point {i}")
    ax0.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    sf = np.linspace(s[0], s[-1], 800)

    # theta
    ax1.plot(s, np.degrees(th), "k.", label="data")
    if ft["success"]:
        ax1.plot(sf, np.degrees(ft["f"](sf, ft["mu"], ft["D"])), "m-", lw=2,
                 label=f"Δ = {ft['D']:.3g} ± {ft['D_err']:.2g} {unit}")
    ax1.set_ylabel("Polar Angle θ (deg)"); ax1.legend(fontsize=8)
    ax1.set_title("angle decomposition along normal")

    ax1.axvline(f["mu"] - f["fwhm"] / 2, color="tab:green", lw=2, linestyle='--')
    ax1.axvline(f["mu"] + f["fwhm"] / 2, color="tab:green", lw=2, linestyle='--')
    ax1.axvline(0 - ft['D'] / 2, color="magenta", lw=2, linestyle='--')
    ax1.axvline(0 + ft['D'] / 2, color="magenta", lw=2, linestyle='--')

    # phi (lab frame) and psi (wall frame)
    ax2.plot(s, np.degrees(ph), ".", color="0.65", ms=4, label="φ (lab frame)")
    ax2.plot(s[ok], np.degrees(psi[ok]), ".", color="k", ms=5,
             label=f"ψ = φ − α  (α = {np.degrees(alpha):.0f}°)")
    for y in (-90, 90):
        ax2.axhline(y, color="magenta", lw=1, alpha=0.7,
                    label="Bloch (ψ = ±90°)" if y == -90 else None)
    for y in (-180, 0, 180):
        ax2.axhline(y, color="cyan", lw=1, alpha=0.7,
                    label="Néel (ψ = 0, ±180°)" if y == -180 else None)

    ax2.axvline(f["mu"] - f["fwhm"] / 2, color="tab:green", lw=2, linestyle='--')
    ax2.axvline(f["mu"] + f["fwhm"] / 2, color="tab:green", lw=2, linestyle='--')
    ax2.axvline(0 - ft['D'] / 2, color="magenta", lw=2, linestyle='--')
    ax2.axvline(0 + ft['D'] / 2, color="magenta", lw=2, linestyle='--')

    ax2.set_ylim(-195, 195)
    ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.set_ylabel("Azimuthal Angle ψ (deg)")
    ax2.legend(fontsize=7, ncol=2, loc="upper right")

    # gradient
    ax3.plot(s, g, "k.", label="Gradient profile")

    if jac is not None and k is not None:
        gxy = np.sqrt(sum(jac[f"dM{c}_d{d}"][k] ** 2 for c in "xyz" for d in "xy"))
        gz = np.sqrt(sum(jac[f"dM{c}_dz"][k] ** 2 for c in "xyz"))
        pp = {}
        for name, arr, st, lab in (("xy", gxy, "r-", "in-plane (x,y) part"),
                                   ("z", gz, "g--", "through-thickness (z) part")):
            pp[name] = compute_profiles(arr, xy[sl], n_i, pixel_size,
                                        half_width, n_samples, order)[1][0]
            ax3.plot(s, pp[name], st, lw=1.5, label=lab)
        ax3.plot(s, np.hypot(pp["xy"], pp["z"]), "c:", lw=2,
                 label="√(xy² + z²)  (should match Grad_slice)")

    if f is not None and f["success"]:
        sf = np.linspace(s[0], s[-1], 500)
        label = f"{LABEL[model]} fit\nFWHM = {f['fwhm']:.3g} {unit}"
        wname = "σ" if model == "gauss" else "w"
        label += (f"\nmean = {f['mu']:.3g} ± {f['mu_err']:.2g}")
                  #f"\n{wname} = {f['w']:.0g} ± {f['w_err']:.1g}")
        ax3.plot(sf, f["f"](sf, *f["popt"]), "b-", lw=2, label=label)

        half = f["offset"] + f["A"] / 2
        ax3.hlines(half, f["mu"] - f["fwhm"] / 2, f["mu"] + f["fwhm"] / 2,
                   color="tab:green", lw=2)
        ax3.axvline(f["mu"] - f["fwhm"] / 2, color="tab:green", lw=2, linestyle='--')
        ax3.axvline(f["mu"] + f["fwhm"] / 2, color="tab:green", lw=2, linestyle='--')

        resid = g - f["f"](s, *f["popt"])
        ax3.axhline(0, color="gray", lw=1)
        ax3.plot(s, resid, "k.-", lw=0.8, alpha=0.5, label="residual")

    ax3.axvline(0 - ft['D'] / 2, color="magenta", lw=2, linestyle='--')
    ax3.axvline(0 + ft['D'] / 2, color="magenta", lw=2, linestyle='--')

    ax3.set_ylabel("|∇m|"); ax3.legend(fontsize=8)
    ax3.set_xlabel(f"distance along normal ({unit})")

    for a in (ax1, ax2, ax3):
        a.axvline(0, color="r", ls="--", lw=1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    return fig, (ax0, ax1, ax2, ax3), dict(s=s, theta=th, phi=ph, psi=psi, alpha=alpha,
                                           grad=g, ft=ft, profile=P)

def psi_vs_z(Mx3, My3, Mz3, grad3, contour_fn, z_ref, ref_result, ref_pt,
             pixel_size, half_width, tol, z_indices=None, min_overlap=0.3, **kw):
    """kw is passed to wall_profile (n_samples, order, sin_min, model, window, window_from)."""
    nz = len(Mz3)
    allowed = set(range(nz)) if z_indices is None else set(z_indices)
    out = {}

    def run(k, r, i):
        p = wall_profile(Mx3[k], My3[k], Mz3[k], grad3[k], r, i,
                         pixel_size, half_width, **kw)
        p.update(xy=r["xy"].copy(), pt=i, pos=r["xy"][i].copy())
        out[k] = p

    i0 = ref_pt % len(ref_result["xy"])
    run(z_ref, ref_result, i0)

    for direction in (+1, -1):
        ref_xy, ref_pos = out[z_ref]["xy"], out[z_ref]["pos"]
        k = z_ref + direction
        while 0 <= k < nz and k in allowed:
            contours = contour_fn(Mz3[k])
            j, score = find_match(ref_xy, contours, tol, min_overlap)
            if j is None:
                print(f"lost the wall at z={k} (best overlap {score:.2f})")
                break
            r = contours[j]
            d, i = cKDTree(r["xy"]).query(ref_pos)
            if d > tol:
                print(f"z={k}: nearest point is {d:.1f} away (> tol), stopping")
                break
            run(k, r, int(i))
            ref_xy, ref_pos = out[k]["xy"], out[k]["pos"]
            k += direction

    nf = [k for k in sorted(out) if out[k]["flipped"]]
    if nf:
        print(f"note: normal flipped locally (contour-wide orientation disagreed) at z = {nf}")
    return out



def _wrap_to(y_deg, lo):
    return (np.asarray(y_deg, float) - lo) % 360 + lo

def _break_jumps(x, y_deg, jump=180):
    """Insert NaN where the wrapped angle jumps, so lines don't cross the plot at +-180."""
    x, y = np.asarray(x, float), np.asarray(y_deg, float)
    j = np.where(np.abs(np.diff(y)) > jump)[0] + 1
    return np.insert(x, j, np.nan), np.insert(y, j, np.nan)


def psi_gradient(p, min_run=3):
    """
    dpsi/ds (rad per unit length) on the 'good' points of one profile.
    psi is unwrapped inside each contiguous run of good points before differentiating,
    so a jump across +-180 deg is not read as a huge rotation. NaN elsewhere.
    smooth : odd Savitzky-Golay window (in samples) applied to the unwrapped psi; 0 = none
    """
    s, psi, good = p["s"], p["psi"], p["good"]
    d = np.full_like(s, np.nan)
    idx = np.where(good)[0]
    if len(idx) == 0:
        return d
    for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if len(run) < min_run:
            continue
        u = np.unwrap(psi[run])
        d[run] = np.gradient(u, s[run])
    return d



def plot_psi_vs_z(Mz3, profiles, z_ref, dz_nm, pixel_size, unit="nm", cmap="viridis",
                  center=False, ylim = (-30, 195), sin_min=0.2, window=0.5, show_points=True, smooth=0,
                  wrap_lo=None):
    zs = np.array(sorted(profiles))
    zn = zs * dz_nm
    norm = Normalize(zn.min(), zn.max() if zn.max() > zn.min() else zn.min() + 1)
    cm = plt.get_cmap(cmap)
    P0 = profiles[z_ref]                           # reference slice for the FWHM lines

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.1, 1], hspace=0.08)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)

    ny, nx = Mz3[z_ref].shape
    extent = (-0.5 * pixel_size, (nx - 0.5) * pixel_size,
              (ny - 0.5) * pixel_size, -0.5 * pixel_size)
    ax0.imshow(Mz3[z_ref], extent=extent, origin="upper", cmap="gray")
    for k, z in zip(zs, zn):
        p = profiles[k]
        ax0.plot(p["xy"][:, 0], p["xy"][:, 1], "-", lw=1, color=cm(norm(z)))
        if show_points:
            ax0.plot(*p["pos"], "o", ms=5, color=cm(norm(z)), mec="w", mew=0.5)
    ax0.set_xlabel(f"x ({unit})"); ax0.set_ylabel(f"y ({unit})")

    seen = set()
    for kk in range(-4, 5):
        y = 90 * kk
        if not -195 <= y <= 195:
            continue
        if kk % 2:
            col, lab = "magenta", "Bloch (ψ = ±90°)"
        elif kk % 4 == 0:
            col, lab = "tab:blue", "Néel − (−Mz → +Mz), ψ = 0"
        else:
            col, lab = "cyan", "Néel + (+Mz → −Mz), ψ = ±180°"
        ax1.axhline(y, color=col, lw=1, alpha=0.6, label=lab if lab not in seen else None)
        seen.add(lab)

    for k, z in zip(zs, zn):
        p = profiles[k]
        if not p["good"].any():
            continue
        x = p["s"][p["good"]] - (p["mu"] if center else 0.0)
        y = np.degrees(p["psi"][p["good"]])
        if wrap_lo is not None:
            y = _wrap_to(y, wrap_lo)
        xb, yb = _break_jumps(x, y)
        ax1.plot(xb, yb, "-", color=cm(norm(z)), lw=1.6)
        ax1.plot(x, y, ".", color=cm(norm(z)), ms=4)

    if not center:
        ax1.axvline(0, color="r", ls="--", lw=1)


    ax1.set_ylim(ylim)
    ax1.set_ylabel("Azimuthal Angle ψ (deg)")
    #ax1.legend(fontsize=7, ncol=2, loc="lower left") ############################################# Change here

    c0 = P0["mu"] if center else 0.0               # FWHM edges of the reference slice
    edges = (P0["mu"] - P0["fwhm"] / 2 - c0, P0["mu"] + P0["fwhm"] / 2 - c0)
    for x in edges:
        ax1.axvline(x, color="tab:green", lw=2, linestyle='--')

    ax2.axhline(0, color="gray", lw=1)
    for k, z in zip(zs, zn):
        p = profiles[k]
        d = np.degrees(psi_gradient(p))
        m = p["inwin"] & np.isfinite(p["mu"])
        if not np.isfinite(d[m]).any():
            continue
        off = p["mu"] if center else 0.0
        ax2.plot(p["s"][m] - off, d[m], ".-", color=cm(norm(z)), lw=1.2, ms=3)

    ax2.axvline(0, color="r", ls="--", lw=1,
                label="gradient peak (s − μ = 0)" if center else "contour (s = 0)")
    ax2.set_xlabel(f"position along normal ({unit}) \n  +s → +Mz domain")
    ax2.set_ylabel(f"dψ/ds (deg/{unit})")
    # ax2.legend(fontsize=7, loc="upper right")
    for x in edges:
        ax2.axvline(x, color="tab:green", lw=2, linestyle='--')

    sm = plt.cm.ScalarMappable(norm, cm)
    fig.colorbar(sm, ax=[ax1, ax2], fraction=0.046, pad=0.02, label=f"z ({unit})")
    return fig, (ax0, ax1, ax2)

########################################## unwrap chi #################################


def arc_length(xy):
    """Cumulative distance along a contour, in the same units as xy (nm)."""
    return np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]

def track_wall(Mz3, contour_fn, z_ref, ref_result, tol, min_overlap=0.3):
    """Matching contour at every z the wall can be followed to (chained from z_ref)."""
    tracked = {z_ref: ref_result}
    for direction in (+1, -1):
        ref_xy = ref_result["xy"]
        k = z_ref + direction
        while 0 <= k < len(Mz3):
            cs = contour_fn(Mz3[k])
            j, score = find_match(ref_xy, cs, tol, min_overlap)
            if j is None:
                print(f"lost the wall at z={k} (best overlap {score:.2f})")
                break
            tracked[k] = cs[j]
            ref_xy = cs[j]["xy"]
            k += direction
    return tracked

def _reduce_profile(P, mode="center"):
    """One profile -> (neel, bloch, cos^2 psi). mode: 'center' (at gradient peak mu) or 'window'."""
    g = P["good"]
    if not P["ok"] or g.sum() < 3:
        return np.nan, np.nan, np.nan
    s, psi = P["s"][g], P["psi"][g]
    neel, bloch = -np.cos(psi), np.sin(psi)
    if mode == "center":
        if not (s[0] <= P["mu"] <= s[-1]):
            return np.nan, np.nan, np.nan
        n, b = np.interp(P["mu"], s, neel), np.interp(P["mu"], s, bloch)
        den = n**2 + b**2
        return n, b, (n**2 / den if den > 0 else np.nan)
    w = np.sin(P["theta"][g]) ** 2                      # weight by in-plane fraction
    return (np.average(neel, weights=w), np.average(bloch, weights=w),
            np.average(np.cos(psi) ** 2, weights=w))


def psi_along_wall(Mx3, My3, Mz3, grad3, tracked, z_ref, pixel_size, half_width, tol,
                   dz_nm, step=4, reduce="center", **kw):
    """
    kw -> wall_profile (model, window, window_from, sin_min, n_samples, order).
    Returns arrays of shape (n_z, n_arc): neel, bloch, c2, fwhm, kappa, x, y.
    The arc coordinate is that of the reference slice; other slices are matched by position.
    """
    ref = tracked[z_ref]
    zs = np.array(sorted(tracked))
    row = {k: n for n, k in enumerate(zs)}
    idx0 = np.arange(0, len(ref["xy"]), step)
    M = len(idx0)
    A = {n: np.full((len(zs), M), np.nan)
         for n in ("neel", "bloch", "c2", "fwhm", "kappa", "x", "y")}

    def fill(k, r, idx, which):
        for m in which:
            i = int(idx[m])
            P = wall_profile(Mx3[k], My3[k], Mz3[k], grad3[k], r, i,
                             pixel_size, half_width, **kw)
            A["neel"][row[k], m], A["bloch"][row[k], m], A["c2"][row[k], m] = \
                _reduce_profile(P, reduce)
            # A["fwhm"][row[k], m] = P["fwhm"]
            A["fwhm"][row[k], m] = (P["fwhm"] if P["ok"] and P["fwhm"] < half_width else np.nan)
            A["kappa"][row[k], m] = r.get("kappa_n", r["kappa"])[i]   # set by orient_normals
            A["x"][row[k], m], A["y"][row[k], m] = r["xy"][i]

    fill(z_ref, ref, idx0, range(M))
    for direction in (+1, -1):
        pos, alive = ref["xy"][idx0].copy(), np.ones(M, bool)
        k = z_ref + direction
        while k in tracked and alive.any():
            r = tracked[k]
            d, idx = cKDTree(r["xy"]).query(pos)
            alive &= d <= tol                           # drop points that lost their match
            fill(k, r, idx, np.where(alive)[0])
            pos = np.where(alive[:, None], r["xy"][idx], pos)   # chain from slice to slice
            k += direction

    # r0 = list(out["zs"]).index(out["z_ref"])
    # drift = np.hypot(out["x"] - out["x"][r0], out["y"] - out["y"][r0])   # (n_z, n_arc), same units as xy

    return dict(zs=zs, z_ref=z_ref, dz_nm=dz_nm, pixel_size=pixel_size,
                arc=arc_length(ref["xy"])[idx0], ref_xy=ref["xy"], idx0=idx0, **A)


QUANT = {"c2":    ("cos²ψ  (1 = Néel, 0 = Bloch)", 0, 1, "cividis"),
         "neel":  ("−cosψ  (+1 = Néel +, −1 = Néel −)", -1, 1, "RdBu_r"),
         "bloch": ("sinψ  (signed Bloch component)", -1, 1, "RdBu_r"),
         "fwhm":  ("FWHM (nm)", None, None, "viridis")}

def _z(out, z_center_idx):
    return (out["zs"] - (0 if z_center_idx is None else z_center_idx)) * out["dz_nm"]


def plot_chi_heatmap(out, profiles_by_pt, quantity="c2", z_center_idx=None, unit="nm",
                     dz_nm=None, ylim=(-195, 195), cmap_z="viridis",
                     kcolor="red", line_color="white", center=False, figsize=(30, 12),
                     fs=15, num_fs=22):
    """
    fs     : font size for axis labels, tick labels, and colorbar labels (>= 15)
    num_fs : font size for the panel numbers
    """
    lab, vmin, vmax, cmap = QUANT[quantity]
    z = _z(out, z_center_idx); arc = out["arc"]
    cm = plt.get_cmap(cmap).reversed(); cm.set_bad("0.85")

    pts = list(profiles_by_pt)
    n = len(pts)
    arc_full = arc_length(out["ref_xy"])
    pt_arc = {p: arc_full[p] for p in pts}
    z_ref = out["z_ref"]

    rc = {"font.family": "serif",
          "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
          "mathtext.fontset": "stix",           # Times-like math (κ, ψ, etc.)
          "font.size": fs, "axes.labelsize": fs,
          "xtick.labelsize": fs, "ytick.labelsize": fs}

    with plt.rc_context(rc):
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = GridSpec(2, n, figure=fig, height_ratios=[1.2, 1])
        a1 = fig.add_subplot(gs[0, :])

        m = a1.pcolormesh(arc, z, np.ma.masked_invalid(out[quantity]), cmap=cm,
                          vmin=vmin, vmax=vmax, shading="nearest")
        cb = fig.colorbar(m, ax=a1, pad=0.02)
        cb.set_label(lab, fontsize=fs)
        cb.ax.tick_params(labelsize=fs)
        a1.set_xlabel(f"arc length along reference wall ({unit})", fontsize=fs)
        a1.set_ylabel(f"z ({unit})" + ("" if z_center_idx is None else ", 0 = film center"),
                      fontsize=fs)

        # white lines where each normal was taken, numbered to match the bottom panels
        ztop = np.nanmax(z)
        for i, p in enumerate(pts):
            a1.axvline(pt_arc[p], color=line_color, lw=2, ls="--")
            a1.text(pt_arc[p], ztop, f" {i+1}", color=line_color, va="top", ha="left",
                    fontsize=num_fs, fontweight="bold")

        # kappa overlay on a secondary y-axis, kappa = 0 at center
        ak = a1.twinx()
        kappa = out["kappa"][list(out["zs"]).index(out["z_ref"])]
        ak.plot(arc, kappa, color=kcolor, lw=3)
        ak.axhline(0, color="gray", lw=0.8, ls="--")
        kmax = np.nanmax(np.abs(kappa))
        ak.set_ylim(-1.05 * kmax, 1.05 * kmax)
        ak.set_ylabel(f"κ (1/{unit})", color=kcolor, fontsize=fs)
        ak.tick_params(axis="y", colors=kcolor, labelsize=fs)
        ak.set_xlim(arc.min(), arc.max())

        # bottom panels: psi vs position along the normal, one per point_idx
        zs_all = sorted(set().union(*[set(pr) for pr in profiles_by_pt.values()]))
        dz = dz_nm if dz_nm is not None else 1.0
        zn_all = np.array(zs_all) * dz
        norm = Normalize(zn_all.min(),
                         zn_all.max() if zn_all.max() > zn_all.min() else zn_all.min() + 1)
        cz = plt.get_cmap(cmap_z)

        bottom = []
        for i, p in enumerate(pts):
            ax = fig.add_subplot(gs[1, i], sharey=bottom[0] if bottom else None)
            bottom.append(ax)
            prof = profiles_by_pt[p]
            zs = np.array(sorted(prof))

            for kk in range(-4, 5):
                y = 90 * kk
                if -195 <= y <= 195:
                    col = "magenta" if kk % 2 else ("tab:blue" if kk % 4 == 0 else "cyan")
                    ax.axhline(y, color=col, lw=1, alpha=0.6)

            for k in zs:
                pk = prof[k]
                if not pk["good"].any():
                    continue
                x = pk["s"][pk["good"]] - (pk["mu"] if center else 0.0)
                y = np.degrees(pk["psi"][pk["good"]])
                xb, yb = _break_jumps(x, y)
                c = cz(norm(k * dz))
                ax.plot(xb, yb, "-", color=c, lw=1.6)
                ax.plot(x, y, ".", color=c, ms=4)

            P0 = prof[z_ref] if z_ref in prof else prof[zs[0]]
            c0 = P0["mu"] if center else 0.0
            for e in (P0["mu"] - P0["fwhm"] / 2 - c0, P0["mu"] + P0["fwhm"] / 2 - c0):
                ax.axvline(e, color="k", lw=1, ls="--")
            if not center:
                ax.axvline(0, color="r", ls="--", lw=1)

            ax.set_ylim(ylim)
            # panel number in the top-left corner (replaces the title)
            ax.text(0.03, 0.96, f"{i+1}", transform=ax.transAxes, va="top", ha="left",
                    fontsize=num_fs, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            ax.set_xlabel(f"position along normal ({unit})", fontsize=fs)
            ax.tick_params(labelsize=fs)
            if i == 0:
                ax.set_ylabel("Azimuthal angle ψ (deg)", fontsize=fs)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

        sm = plt.cm.ScalarMappable(norm, cz)
        cb2 = fig.colorbar(sm, ax=bottom, pad=0.02)
        cb2.set_label(f"z ({unit})", fontsize=fs)
        cb2.ax.tick_params(labelsize=fs)

    return fig, (a1, ak, bottom)


def plot_points_on_spline(Mz_ref, result, point_idxs, pixel_size, out=None, unit="nm",
                          figsize=(8, 8), mark_color="white"):
    """
    Mz_ref     : reference Mz slice (e.g. Mz_slice)
    result     : results_3[contour_idx], the same contour passed to psi_vs_z / plot_angle_decomposition
    point_idxs : the 4 indices into the full contour (e.g. [270, 300, 340, 380])
    out        : optional out_arc, used to check consistency with the heatmap
    """
    xy = np.asarray(result["xy"])
    arc_full = arc_length(xy)
    ny, nx = Mz_ref.shape
    extent = (-0.5 * pixel_size, (nx - 0.5) * pixel_size,
              (ny - 0.5) * pixel_size, -0.5 * pixel_size)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.imshow(Mz_ref, extent=extent, origin="upper", cmap="gray")
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=arc_full, cmap="plasma", s=6)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label=f"arc length ({unit})")

    for i, p in enumerate(point_idxs):
        x, y = xy[p]
        ax.plot(x, y, "o", ms=10, mfc="none", mec=mark_color, mew=2)
        ax.annotate(f"{i+1}  (idx {p}, {arc_full[p]:.0f} {unit})", (x, y),
                    xytext=(8, 8), textcoords="offset points",
                    color=mark_color, fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, lw=0))

    # start/end markers so the contour direction is clear
    ax.plot(*xy[0], "s", color="tab:green", ms=7, label="start (idx 0)")
    ax.plot(*xy[-1], "^", color="tab:red", ms=7, label=f"end (idx {len(xy)-1})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel(f"x ({unit})"); ax.set_ylabel(f"y ({unit})")
    ax.set_title("Selected normals on the reference contour")

    # consistency checks against the heatmap's data
    if out is not None:
        same_xy = (len(out["ref_xy"]) == len(xy)) and np.allclose(out["ref_xy"], xy)
        print(f"out['ref_xy'] matches this contour: {same_xy}")
        for i, p in enumerate(point_idxs):
            a_heat = out["arc"][np.where(out["idx0"] == p)[0]]
            msg = f"{a_heat[0]:.1f}" if len(a_heat) else "not on the step grid"
            print(f"  {i+1}: idx {p} -> arc_full {arc_full[p]:.1f} | heatmap arc {msg} "
                  f"| xy = ({xy[p,0]:.1f}, {xy[p,1]:.1f})")
    return fig, ax


import warnings


def _width_array(out):
    """(n_z, n_arc) DW width on the reference arc, rows ordered like out['zs']."""
    if "fwhm" not in out:
        raise KeyError("out has no 'fwhm' array; width overlay needs per-z FWHM "
                       "resampled onto out['arc'] (same shape as out['kappa']).")
    return np.asarray(out["fwhm"], float)

def _width_deviation(W, ref="layer"):
    """Width minus its mean. ref='layer': each z layer's own mean over arc length
    (each spline centered on 0). ref='global': one mean over all layers and arc points."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if ref == "layer":
            base = np.nanmean(W, axis=1, keepdims=True)
        elif ref == "global":
            base = np.nanmean(W)
        else:
            raise ValueError("width_ref must be 'layer' or 'global'")
    return W - base, base


def _style_twin(ax, color, side_offset=None):
    """Colour-code a twin y-axis; optionally push its spine outward so two don't overlap."""
    if side_offset is not None:
        ax.spines["right"].set_position(("outward", side_offset))
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for name, sp in ax.spines.items():
            sp.set_visible(name == "right")
    ax.spines["right"].set_color(color)
    ax.tick_params(axis="y", colors=color)


def plot_chi_heatmap_w_width(out, profiles_by_pt, quantity="c2", z_center_idx=None, unit="nm",
                     dz_nm=None, ylim=(-195, 195), cmap_z="viridis",
                     kcolor="red", line_color="white", center=False, figsize=(30, 12),
                     fs=15, num_fs=22,
                     overlay="kappa", wcolor="green", width_mode="mean", width_ref = 'layer', 
                     stagger=90, cbar_pad=None):
    """
    overlay    : "kappa" | "width" | "both" | "none"   curve(s) drawn over the heatmap
    wcolor     : colour of the DW-width overlay
    width_mode : "mean"  -> mean over z with a 16-84% band
                 "per_z" -> one thin line per z layer
                 "both"  -> faint per-z lines plus the bold mean
    stagger    : points to offset the second right-hand axis when overlay="both"
    cbar_pad   : heatmap colorbar pad; default is 0.02, or 0.09 for overlay="both"
                 (the extra room keeps the staggered axis off the colorbar)
    fs     : font size for axis labels, tick labels, and colorbar labels (>= 15)
    num_fs : font size for the panel numbers

    Returns fig, (a1, ak, bottom) where ak is now a dict {"kappa": ax or None,
    "width": ax or None}.
    """
    if overlay not in ("kappa", "width", "both", "none"):
        raise ValueError("overlay must be 'kappa', 'width', 'both' or 'none'")
    if width_mode not in ("mean", "per_z", "both"):
        raise ValueError("width_mode must be 'mean', 'per_z' or 'both'")
    show_k = overlay in ("kappa", "both")
    show_w = overlay in ("width", "both")
    if cbar_pad is None:
        cbar_pad = 0.09 if overlay == "both" else 0.02

    lab, vmin, vmax, cmap = QUANT[quantity]
    z = _z(out, z_center_idx); arc = out["arc"]
    cm = plt.get_cmap(cmap).reversed(); cm.set_bad("0.85")

    pts = list(profiles_by_pt)
    n = len(pts)
    arc_full = arc_length(out["ref_xy"])
    pt_arc = {p: arc_full[p] for p in pts}
    z_ref = out["z_ref"]

    rc = {"font.family": "serif",
          "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
          "mathtext.fontset": "stix",
          "font.size": fs, "axes.labelsize": fs,
          "xtick.labelsize": fs, "ytick.labelsize": fs}

    with plt.rc_context(rc):
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = GridSpec(2, n, figure=fig, height_ratios=[1.2, 1])
        a1 = fig.add_subplot(gs[0, :])

        m = a1.pcolormesh(arc, z, np.ma.masked_invalid(out[quantity]), cmap=cm,
                          vmin=vmin, vmax=vmax, shading="nearest")
        cb = fig.colorbar(m, ax=a1, pad=cbar_pad)
        cb.set_label(lab, fontsize=fs)
        cb.ax.tick_params(labelsize=fs)
        a1.set_xlabel(f"arc length along reference wall ({unit})", fontsize=fs)
        a1.set_ylabel(f"z ({unit})" + ("" if z_center_idx is None else ", 0 = film center"),
                      fontsize=fs)

        # white lines where each normal was taken, numbered to match the bottom panels
        ztop = np.nanmax(z)
        for i, p in enumerate(pts):
            a1.axvline(pt_arc[p], color=line_color, lw=2, ls="--")
            a1.text(pt_arc[p], ztop, f" {i+1}", color=line_color, va="top", ha="left",
                    fontsize=num_fs, fontweight="bold")

        ak = aw = None

        # --- kappa overlay (reference slice), kappa = 0 at center ---
        if show_k:
            ak = a1.twinx()
            kappa = out["kappa"][list(out["zs"]).index(out["z_ref"])]
            ak.plot(arc, kappa, color=kcolor, lw=3)
            ak.axhline(0, color="gray", lw=0.8, ls="--")
            kmax = np.nanmax(np.abs(kappa))
            ak.set_ylim(-1.05 * kmax, 1.05 * kmax)
            ak.set_ylabel(f"κ (1/{unit})", color=kcolor, fontsize=fs)
            ak.tick_params(axis="y", labelsize=fs)
            _style_twin(ak, kcolor)
            ak.set_xlim(arc.min(), arc.max())

        # # --- DW width overlay (all z layers) ---
        # if show_w:
        #     W = _width_array(out)
        #     aw = a1.twinx()
        #     if width_mode in ("per_z", "both"):
        #         for row in W:
        #             aw.plot(arc, row, color=wcolor, lw=0.8,
        #                     alpha=0.3 if width_mode == "both" else 0.5)
        #     if width_mode in ("mean", "both"):
        #         with warnings.catch_warnings():
        #             warnings.simplefilter("ignore", RuntimeWarning)
        #             mean = np.nanmean(W, axis=0)
        #             lo, hi = np.nanpercentile(W, [16, 84], axis=0)
        #         aw.fill_between(arc, lo, hi, color=wcolor, alpha=0.2, lw=0)
        #         aw.plot(arc, mean, color=wcolor, lw=3)
        #     wtop = np.nanpercentile(W, 99)          # robust to a few outlier fits
        #     aw.set_ylim(0, 1.05 * wtop)
        #     aw.set_ylabel(f"DW width, FWHM ({unit})", color=wcolor, fontsize=fs)
        #     aw.tick_params(axis="y", labelsize=fs)
        #     # second axis is staggered outward only when kappa is also shown
        #     _style_twin(aw, wcolor, side_offset=stagger if show_k else None)
        #     aw.set_xlim(arc.min(), arc.max())


        # --- DW width overlay: change from the mean width (centered on 0) ---
        if show_w:
            W = _width_array(out)
            dW, _ = _width_deviation(W, width_ref)
            aw = a1.twinx()
            if width_mode in ("per_z", "both"):
                for row in dW:
                    aw.plot(arc, row, color=wcolor, lw=0.8,
                            alpha=0.3 if width_mode == "both" else 0.5)
            if width_mode in ("mean", "both"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = np.nanmean(dW, axis=0)
                    lo, hi = np.nanpercentile(dW, [16, 84], axis=0)
                aw.fill_between(arc, lo, hi, color=wcolor, alpha=0.2, lw=0)
                aw.plot(arc, mean, color=wcolor, lw=3)
            aw.axhline(0, color=wcolor, lw=1, ls="--")
            wmax = np.nanpercentile(np.abs(dW), 99)      # robust to a few outlier fits
            wmax = wmax if np.isfinite(wmax) and wmax > 0 else 1.0
            aw.set_ylim(-1.05 * wmax, 1.05 * wmax)
            aw.set_ylabel(f"Δ DW width from mean ({unit})", color=wcolor, fontsize=fs)
            aw.tick_params(axis="y", labelsize=fs)
            _style_twin(aw, wcolor, side_offset=stagger if show_k else None)
            aw.set_xlim(arc.min(), arc.max())


        # bottom panels: psi vs position along the normal, one per point_idx
        zs_all = sorted(set().union(*[set(pr) for pr in profiles_by_pt.values()]))
        dz = dz_nm if dz_nm is not None else 1.0
        zn_all = np.array(zs_all) * dz
        norm = Normalize(zn_all.min(),
                         zn_all.max() if zn_all.max() > zn_all.min() else zn_all.min() + 1)
        cz = plt.get_cmap(cmap_z)

        bottom = []
        for i, p in enumerate(pts):
            ax = fig.add_subplot(gs[1, i], sharey=bottom[0] if bottom else None)
            bottom.append(ax)
            prof = profiles_by_pt[p]
            zs = np.array(sorted(prof))

            for kk in range(-4, 5):
                y = 90 * kk
                if -195 <= y <= 195:
                    col = "magenta" if kk % 2 else ("tab:blue" if kk % 4 == 0 else "cyan")
                    ax.axhline(y, color=col, lw=1, alpha=0.6)

            for k in zs:
                pk = prof[k]
                if not pk["good"].any():
                    continue
                x = pk["s"][pk["good"]] - (pk["mu"] if center else 0.0)
                y = np.degrees(pk["psi"][pk["good"]])
                xb, yb = _break_jumps(x, y)
                c = cz(norm(k * dz))
                ax.plot(xb, yb, "-", color=c, lw=1.6)
                ax.plot(x, y, ".", color=c, ms=4)

            P0 = prof[z_ref] if z_ref in prof else prof[zs[0]]
            c0 = P0["mu"] if center else 0.0
            for e in (P0["mu"] - P0["fwhm"] / 2 - c0, P0["mu"] + P0["fwhm"] / 2 - c0):
                ax.axvline(e, color="k", lw=1, ls="--")
            if not center:
                ax.axvline(0, color="r", ls="--", lw=1)

            ax.set_ylim(ylim)
            ax.text(0.03, 0.96, f"{i+1}", transform=ax.transAxes, va="top", ha="left",
                    fontsize=num_fs, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            ax.set_xlabel(f"position along normal ({unit})", fontsize=fs)
            ax.tick_params(labelsize=fs)
            if i == 0:
                ax.set_ylabel("Azimuthal angle ψ (deg)", fontsize=fs)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

        sm = plt.cm.ScalarMappable(norm, cz)
        cb2 = fig.colorbar(sm, ax=bottom, pad=0.02)
        cb2.set_label(f"z ({unit})", fontsize=fs)
        cb2.ax.tick_params(labelsize=fs)

    return fig, (a1, {"kappa": ak, "width": aw}, bottom)


def _dpsi_ds(s, psi, good):
    """dψ/ds in deg per unit s. ψ (radians) is unwrapped and differentiated separately
    on each contiguous run of good points, so ±180° wraps don't create spikes and
    gaps in `good` don't get bridged. Points outside the runs are NaN (breaks the line)."""
    d = np.full(len(s), np.nan)
    idx = np.flatnonzero(good)
    if idx.size < 2:
        return d
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(run) >= 2:
            d[run] = np.gradient(np.degrees(np.unwrap(psi[run])), s[run])
    return d


def plot_chi_heatmap_w_width_and_chigrad(out, profiles_by_pt, quantity="c2", z_center_idx=None, unit="nm",
                     dz_nm=None, ylim=(-195, 195), cmap_z="viridis",
                     kcolor="red", line_color="white", center_on="central",
                     figsize=(30, 15), fs=15, num_fs=22,
                     overlay="kappa", wcolor="green", width_mode="mean", width_ref="layer",
                     stagger=90, cbar_pad=None):
    """
    Top: heatmap of `quantity` with optional kappa / width overlays.
    Below: for each selected point, ψ vs position along the normal (row 1) and
    dψ/ds (row 2), one line per z layer.

    center_on : where x = 0 sits in the bottom panels
        "central" -> at the Gaussian-fit center (mu) of the central slice; every z layer
                     is shifted by that same amount, so layers stay comparable
        "contour" -> at the contour itself (s = 0); the central-slice mu is drawn as the
                     red line instead
        "layer"   -> each layer is shifted by its own mu (the old center=True)
    The red dashed line (0) and black dashed lines (±FWHM/2) are always taken from
    the central slice's fit. Central slice = z_center_idx if given, else out["z_ref"]
    (the nearest available slice is used if that one has no profile at a point).

    overlay, wcolor, width_mode, width_ref, stagger, cbar_pad : as before.

    Returns fig, (a1, {"kappa": ax or None, "width": ax or None},
                  {"psi": [axes], "dpsi": [axes]})
    """
    if overlay not in ("kappa", "width", "both", "none"):
        raise ValueError("overlay must be 'kappa', 'width', 'both' or 'none'")
    if width_mode not in ("mean", "per_z", "both"):
        raise ValueError("width_mode must be 'mean', 'per_z' or 'both'")
    if center_on not in ("central", "contour", "layer"):
        raise ValueError("center_on must be 'central', 'contour' or 'layer'")
    show_k = overlay in ("kappa", "both")
    show_w = overlay in ("width", "both")
    if cbar_pad is None:
        cbar_pad = 0.09 if overlay == "both" else 0.02

    lab, vmin, vmax, cmap = QUANT[quantity]
    z = _z(out, z_center_idx); arc = out["arc"]
    cm = plt.get_cmap(cmap).reversed(); cm.set_bad("0.85")

    pts = list(profiles_by_pt)
    n = len(pts)
    arc_full = arc_length(out["ref_xy"])
    pt_arc = {p: arc_full[p] for p in pts}
    z_ref = out["z_ref"]
    z_central = z_center_idx if z_center_idx is not None else z_ref

    rc = {"font.family": "serif",
          "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
          "mathtext.fontset": "stix",
          "font.size": fs, "axes.labelsize": fs,
          "xtick.labelsize": fs, "ytick.labelsize": fs}

    with plt.rc_context(rc):
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = GridSpec(3, n, figure=fig, height_ratios=[1.2, 1, 0.8])
        a1 = fig.add_subplot(gs[0, :])

        m = a1.pcolormesh(arc, z, np.ma.masked_invalid(out[quantity]), cmap=cm,
                          vmin=vmin, vmax=vmax, shading="nearest")
        cb = fig.colorbar(m, ax=a1, pad=cbar_pad)
        cb.set_label(lab, fontsize=fs)
        cb.ax.tick_params(labelsize=fs)
        a1.set_xlabel(f"arc length along reference wall ({unit})", fontsize=fs)
        a1.set_ylabel(f"z ({unit})" + ("" if z_center_idx is None else ", 0 = film center"),
                      fontsize=fs)

        # white lines where each normal was taken, numbered to match the bottom panels
        ztop = np.nanmax(z)
        for i, p in enumerate(pts):
            a1.axvline(pt_arc[p], color=line_color, lw=2, ls="--")
            a1.text(pt_arc[p], ztop, f" {i+1}", color=line_color, va="top", ha="left",
                    fontsize=num_fs, fontweight="bold")

        ak = aw = None

        # --- kappa overlay (reference slice), kappa = 0 at center ---
        if show_k:
            ak = a1.twinx()
            kappa = out["kappa"][list(out["zs"]).index(out["z_ref"])]
            ak.plot(arc, kappa, color=kcolor, lw=3)
            ak.axhline(0, color="gray", lw=0.8, ls="--")
            kmax = np.nanmax(np.abs(kappa))
            ak.set_ylim(-1.05 * kmax, 1.05 * kmax)
            ak.set_ylabel(f"κ (1/{unit})", color=kcolor, fontsize=fs)
            ak.tick_params(axis="y", labelsize=fs)
            _style_twin(ak, kcolor)
            ak.set_xlim(arc.min(), arc.max())

        # --- DW width overlay: change from the mean width (centered on 0) ---
        if show_w:
            W = _width_array(out)
            dW, _ = _width_deviation(W, width_ref)
            aw = a1.twinx()
            if width_mode in ("per_z", "both"):
                for row in dW:
                    aw.plot(arc, row, color=wcolor, lw=0.8,
                            alpha=0.3 if width_mode == "both" else 0.5)
            if width_mode in ("mean", "both"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = np.nanmean(dW, axis=0)
                    lo, hi = np.nanpercentile(dW, [16, 84], axis=0)
                aw.fill_between(arc, lo, hi, color=wcolor, alpha=0.2, lw=0)
                aw.plot(arc, mean, color=wcolor, lw=3)
            aw.axhline(0, color=wcolor, lw=1, ls="--")
            wmax = np.nanpercentile(np.abs(dW), 99)
            wmax = wmax if np.isfinite(wmax) and wmax > 0 else 1.0
            aw.set_ylim(-1.05 * wmax, 1.05 * wmax)
            aw.set_ylabel(f"Δ DW width from mean ({unit})", color=wcolor, fontsize=fs)
            aw.tick_params(axis="y", labelsize=fs)
            _style_twin(aw, wcolor, side_offset=stagger if show_k else None)
            aw.set_xlim(arc.min(), arc.max())

        # --- bottom: ψ (row 1) and dψ/ds (row 2) vs position along the normal ---
        zs_all = sorted(set().union(*[set(pr) for pr in profiles_by_pt.values()]))
        dz = dz_nm if dz_nm is not None else 1.0
        zn_all = np.array(zs_all) * dz
        norm = Normalize(zn_all.min(),
                         zn_all.max() if zn_all.max() > zn_all.min() else zn_all.min() + 1)
        cz = plt.get_cmap(cmap_z)

        psi_axes, dpsi_axes, all_d = [], [], []
        for i, p in enumerate(pts):
            ax = fig.add_subplot(gs[1, i], sharey=psi_axes[0] if psi_axes else None)
            ad = fig.add_subplot(gs[2, i], sharex=ax,
                                 sharey=dpsi_axes[0] if dpsi_axes else None)
            psi_axes.append(ax); dpsi_axes.append(ad)
            prof = profiles_by_pt[p]
            zs = np.array(sorted(prof))

            # central-slice Gaussian fit -> the "0" line and the FWHM edges
            kc = zs[np.argmin(np.abs(zs - z_central))]
            mu_c, fwhm_c = prof[kc]["mu"], prof[kc]["fwhm"]
            have_mu = np.isfinite(mu_c)
            have_fw = np.isfinite(fwhm_c)
            c0 = mu_c if (center_on == "contour" and have_mu) else 0.0

            for kk in range(-4, 5):
                y = 90 * kk
                if -195 <= y <= 195:
                    col = "magenta" if kk % 2 else ("tab:blue" if kk % 4 == 0 else "cyan")
                    ax.axhline(y, color=col, lw=1, alpha=0.6)
            ad.axhline(0, color="gray", lw=1, alpha=0.6)

            for k in zs:
                pk = prof[k]
                if not pk["good"].any():
                    continue
                if center_on == "central":
                    shift = mu_c if have_mu else 0.0
                elif center_on == "layer":
                    shift = pk["mu"] if np.isfinite(pk["mu"]) else 0.0
                else:
                    shift = 0.0
                c = cz(norm(k * dz))
                good = pk["good"]

                x = pk["s"][good] - shift
                y = np.degrees(pk["psi"][good])
                xb, yb = _break_jumps(x, y)
                ax.plot(xb, yb, "-", color=c, lw=1.6)
                ax.plot(x, y, ".", color=c, ms=4)

                d = _dpsi_ds(pk["s"], pk["psi"], good)
                xs = pk["s"] - shift
                ad.plot(xs, d, "-", color=c, lw=1.4)
                ad.plot(xs[good], d[good], ".", color=c, ms=4)
                all_d.append(d[np.isfinite(d)])

            for a in (ax, ad):
                if have_mu:
                    a.axvline(c0, color="r", ls="--", lw=1)
                if have_fw:
                    for e in (c0 - fwhm_c / 2, c0 + fwhm_c / 2):
                        a.axvline(e, color="k", lw=1, ls="--")

            ax.set_ylim(ylim)
            ax.text(0.03, 0.96, f"{i+1}", transform=ax.transAxes, va="top", ha="left",
                    fontsize=num_fs, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            ax.tick_params(labelsize=fs, labelbottom=False)
            ad.tick_params(labelsize=fs)
            ad.set_xlabel(f"position along normal ({unit})", fontsize=fs)
            if i == 0:
                ax.set_ylabel("Azimuthal angle ψ (deg)", fontsize=fs)
                ad.set_ylabel(f"dψ/ds (deg/{unit})", fontsize=fs)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(ad.get_yticklabels(), visible=False)

        # shared dψ/ds limits, robust to a few noisy points
        dcat = np.concatenate(all_d) if all_d else np.array([])
        if dcat.size:
            dmax = np.nanpercentile(np.abs(dcat), 99)
            if np.isfinite(dmax) and dmax > 0:
                dpsi_axes[0].set_ylim(-1.05 * dmax, 1.05 * dmax)

        sm = plt.cm.ScalarMappable(norm, cz)
        cb2 = fig.colorbar(sm, ax=psi_axes + dpsi_axes, pad=0.02)
        cb2.set_label(f"z ({unit})", fontsize=fs)
        cb2.ax.tick_params(labelsize=fs)

    return fig, (a1, {"kappa": ak, "width": aw}, {"psi": psi_axes, "dpsi": dpsi_axes})