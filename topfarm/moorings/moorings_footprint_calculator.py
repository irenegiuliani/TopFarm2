# -*- coding: utf-8 -*-
"""
Refactored and hardened moorings footprint calculator.

Goals:
- Remove fragile vector clipping (Shapely/GEOS) in favor of numeric subsetting.
- Make cone/surface clipping and edge extraction robust to empty/invalid results.
- Compute anchors by pure geometry (ray vs. footprint edges), avoiding ray_trace where possible.
- Add clear errors, small helpers, and consistent numpy handling.

Assumptions:
- Bathymetry is an xarray.DataArray with coords named 'x' and 'y' (monotonic), values are depth z.
- PyVista is available; functions work in scripts or notebooks.

"""
from __future__ import annotations

import logging
from typing import Iterable, List, Tuple, Optional
from topfarm.moorings.seabed_features import seabed_features
import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
import xarray as xr  # only for types; keep .rio if present



# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

EPS = 1e-12


# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------
def subset_rect_da(da: xr.DataArray, x: float, y: float, max_d: float) -> xr.DataArray:
    """Subset a rectangular window from an xarray.DataArray without Shapely/GEOS.

    Handles both increasing/decreasing coord orders. Raises ValueError on empty.
    """
    if 'x' not in da.coords or 'y' not in da.coords:
        raise ValueError("DataArray must have 'x' and 'y' coordinates.")

    x0, x1 = x - max_d, x + max_d
    y0, y1 = y - max_d, y + max_d

    xs = slice(min(x0, x1), max(x0, x1)) if float(da.x[0]) < float(da.x[-1]) else slice(max(x0, x1), min(x0, x1))
    ys = slice(min(y0, y1), max(y0, y1)) if float(da.y[0]) < float(da.y[-1]) else slice(max(y0, y1), min(y0, y1))

    sub = da.sel(x=xs, y=ys)
    if sub.x.size == 0 or sub.y.size == 0:
        raise ValueError("Subset is empty: window lies outside raster extent.")
    return sub


def fine_mesh(x: np.ndarray, y: np.ndarray, z: np.ndarray, resolution_factor: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a finer meshgrid by interpolating a given 2D surface.

    Parameters
    ----------
    x, y : 1D arrays
        Grid coordinates. Must be monotonic.
    z : 2D array
        Values on (y, x) grid.
    resolution_factor : int
        Refinement factor (>1 for finer grid).
    """
    if resolution_factor <= 1:
        X, Y = np.meshgrid(x, y)
        return X, Y, z

    nx, ny = int(len(x) * resolution_factor), int(len(y) * resolution_factor)
    interp = RegularGridInterpolator((y, x), z, method='linear', bounds_error=False, fill_value=np.nan)
    x_f = np.linspace(float(np.min(x)), float(np.max(x)), nx)
    y_f = np.linspace(float(np.min(y)), float(np.max(y)), ny)
    Xf, Yf = np.meshgrid(x_f, y_f)
    Zf = interp(np.column_stack([Yf.ravel(), Xf.ravel()])).reshape(Yf.shape)
    return Xf, Yf, Zf


# -----------------------------------------------------------------------------
# Geometry helpers (azimuth and ray/segment intersection)
# -----------------------------------------------------------------------------
def azimuth_dir_from_north(az_deg: float) -> np.ndarray:
    """Azimuth from North (clockwise) -> 2D direction vector (dx, dy).
    0° -> +Y; 90° -> +X; 180° -> -Y; 270° -> -X
    """
    a = np.deg2rad(az_deg)
    return np.array([np.sin(a), np.cos(a)], dtype=float)


def cross2d(u: np.ndarray, v: np.ndarray) -> float:
    return float(u[0]*v[1] - u[1]*v[0])


def ray_segment_intersection_xy(Oxy: np.ndarray, r: np.ndarray, Axy: np.ndarray, Bxy: np.ndarray, eps: float = EPS) -> Tuple[Optional[float], Optional[float]]:
    """Intersect ray (O + t*r, t>=0) with segment [A,B] in XY; return (t,u) or (None,None)."""
    s = Bxy - Axy
    rxs = cross2d(r, s)
    if abs(rxs) < eps:
        return None, None
    qp = Axy - Oxy
    t = cross2d(qp, s) / rxs   # along ray
    u = cross2d(qp, r) / rxs   # along segment [0,1]
    if t > eps and 0.0 <= u <= 1.0:
        return float(t), float(u)
    return None, None


def iter_poly_segments(poly: pv.PolyData):
    """Yield successive (i0,i1) point indices for each polyline/cell in a PolyData edges object."""
    arr = np.asarray(poly.lines)
    if arr.size == 0:
        # Fallback: use points in order and close
        idxs = np.arange(poly.n_points)
        for a, b in zip(idxs, np.roll(idxs, -1)):
            yield int(a), int(b)
        return

    i = 0
    n_tot = arr.size
    while i < n_tot:
        n = int(arr[i]); ids = arr[i+1:i+1+n]
        for k in range(n-1):
            yield int(ids[k]), int(ids[k+1])
        i += n + 1


# -----------------------------------------------------------------------------
# Core: surface clipping and anchor computation (geometry only)
# -----------------------------------------------------------------------------

def surf_clipping(surface: pv.StructuredGrid, beta: float, alpha: float, x_t: float, y_t: float,
                  mooring_type: str, n_moorings: int, plot: bool) -> Tuple[np.ndarray, List[dict], float]:
    """Generate a conical mooring footprint and compute anchor points on a surface.

    Returns
    -------
    edges_pts : (M,3) array of edge coordinates (footprint boundary)
    anchors : list[dict]
    max_radius : float
    """
    if mooring_type != 'taught':
        raise ValueError('Only "taught" mooring_type is implemented')

    beta_rad = np.deg2rad(beta)
    vertex = np.array([float(x_t), float(y_t), 0.0])

    # Robust cone dimensions
    if surface.n_points == 0:
        raise ValueError("Surface has no points.")
    min_z = float(np.nanmin(surface.points[:, 2]))
    # height of cone equals |min depth| + margin to ensure it crosses bathymetry
    h = abs(min_z) + 800.0
    r = h * np.tan(beta_rad)

    # Build cone (upwards), then clip bathymetry by cone volume
    direction = np.array([0.0, 0.0, 1.0])
    center = vertex - direction * (h / 2.0)
    try:
        cone = pv.Cone(center=center, direction=direction, height=h, radius=r, resolution=400, capping=True)
        # Triangulate/clean once
        surf_tri = surface.extract_surface().triangulate().clean()
        cone_tri = cone.triangulate().clean()
        clipped = surf_tri.clip_surface(cone_tri, invert=True)
        if clipped.n_points == 0:
            raise RuntimeError("Empty clipped surface (bathymetry outside cone).")
        edges = clipped.extract_surface().extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                               non_manifold_edges=False, manifold_edges=False)
        if edges.n_points == 0:
            raise RuntimeError("No boundary edges extracted from clipped surface.")

        # Max footprint radius
        diffs = edges.points - vertex
        p_max = edges.points[np.argmax(np.linalg.norm(diffs, axis=1))]
        max_radius = float(np.linalg.norm(p_max - np.array([x_t, y_t, p_max[2]])))
    except Exception as e:
        LOG.warning(f"%s: falling back to fictitious flat disc for footprint.", e)
        # Fallback: flat disc at projected bathymetry
        vertical_end = np.array([x_t, y_t, -1000.0])
        try:
            pt, _ = surf_tri.ray_trace(vertex, vertical_end)
            center_disc = np.asarray(pt).reshape(-1, 3)[0]
        except Exception:
            center_disc = np.array([x_t, y_t, min_z])
        surface_fict = pv.Disc(center=center_disc, inner=0.0, outer=800.0, normal=(0, 0, 1), c_res=720)
        clipped = surface_fict.triangulate().clean().clip_surface(cone_tri, invert=True)
        edges = clipped.extract_surface().extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                               non_manifold_edges=False, manifold_edges=False)
        if edges.n_points == 0:
            raise RuntimeError("Fallback disc produced no edges; cannot compute footprint.")
        diffs = edges.points - vertex
        p_max = edges.points[np.argmax(np.linalg.norm(diffs, axis=1))]
        max_radius = float(np.linalg.norm(p_max - np.array([x_t, y_t, p_max[2]])))

    # ----------------- Anchors by geometry only (no ray_trace) -----------------
    anchors: List[dict] = []
    O = vertex.copy()
    Oxy = O[:2]

    def pick_anchor_for_az(az_deg: float) -> np.ndarray:
        r_dir = azimuth_dir_from_north(az_deg)
        best_t = np.inf
        best_point = None
        for i0, i1 in iter_poly_segments(edges):
            A = edges.points[i0]
            B = edges.points[i1]
            t, u = ray_segment_intersection_xy(Oxy, r_dir, A[:2], B[:2])
            if t is None:
                continue
            if t < best_t:
                best_t = t
                xy = Oxy + t * r_dir
                z = (1.0 - u) * A[2] + u * B[2]
                best_point = np.array([xy[0], xy[1], z], dtype=float)
        if best_point is None:
            # Fallback: pick the edge vertex with closest azimuth
            V = edges.points[:, :2] - Oxy
            ang = np.arctan2(V[:, 0], V[:, 1])  # azimuth from North
            az = np.deg2rad(az_deg)
            diff = np.abs((ang - az + np.pi) % (2*np.pi) - np.pi)
            idx = int(np.argmin(diff))
            best_point = edges.points[idx]
        return best_point

    for k in range(n_moorings):
        az_k = float(alpha + (360.0 / n_moorings) * k)
        p = pick_anchor_for_az(az_k)
        length = float(np.linalg.norm(p - O))
        anchors.append({
            'name': f'Anchor{k}',
            'coords': np.asarray([p], float),
            'mooring_type': mooring_type,
            'length': length,
        })

    if plot:
        try:
            plotter = pv.Plotter()
            plotter.add_mesh(surface, show_edges=True, color='lightblue')
            plotter.add_mesh(clipped, color='white', opacity=0.25)
            plotter.add_mesh(pv.PolyData(edges.points), color='navy')
            plotter.add_mesh(pv.Sphere(radius=10.0, center=vertex), color='red')
            for a in anchors:
                plotter.add_mesh(pv.Sphere(radius=10.0, center=a['coords'][0]), color='orange')
            plotter.show()
        except Exception as e:
            LOG.warning("Plotting failed: %s", e)

    return edges.points.copy(), anchors, max_radius


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def footprint(da: xr.DataArray, shape_crs, wt_x: Iterable[float], wt_y: Iterable[float],
              beta: float, alpha: float, max_d: float, resolution_factor: int,
              mooring_type: str, n_moorings: int, plot: bool):
    """Generate mooring footprints and anchors for given turbine positions.

    Parameters
    ----------
    da : xarray.DataArray
        Bathymetric raster (coords 'x', 'y').
    shape_crs : CRS
        Unused here; kept for API compatibility.
    wt_x, wt_y : arrays
        Turbine XY coordinates.
    beta, alpha : floats
        Cone half-angle (deg) and initial azimuth (deg from North).
    max_d : float
        Half-size of clipping window around each turbine (same units as x/y).
    resolution_factor : int
        >1 to refine grid via interpolation.
    mooring_type : str
        Only 'taught' supported.
    n_moorings : int
        Number of mooring lines per turbine.
    plot : bool
        Plot intermediate results with PyVista.
    """
    foot_print: List[dict] = []

    # Validate coords
    if 'x' not in da.coords or 'y' not in da.coords:
        raise ValueError("DataArray must have 'x' and 'y' coordinates.")

    for x, y in zip(wt_x, wt_y):
        try:
            clipped = subset_rect_da(da, float(x), float(y), float(max_d))

            # Interpolate if requested
            X, Y = np.meshgrid(clipped.x.data, clipped.y.data)
            if int(resolution_factor) != 1:
                X, Y, Z = fine_mesh(clipped.x.data, clipped.y.data, np.asarray(clipped.data), int(resolution_factor))
            else:
                Z = np.asarray(clipped.data)

            surface = pv.StructuredGrid(np.ascontiguousarray(X), np.ascontiguousarray(Y), np.ascontiguousarray(Z))

            edges, anchors, max_radius = surf_clipping(surface, beta, alpha, float(x), float(y), mooring_type, int(n_moorings), plot)
            foot_print.append({'mooring_footprint': edges, 'anchors': anchors, 'max_radius': max_radius})

        except Exception as e:
            LOG.warning("%s occurred while creating footprint; using fictitious surface.", e)
            # Fallback: flat disc at arbitrary depth
            surface = pv.Disc(center=np.array([x, y, -500.0]), inner=0.0, outer=800.0, normal=(0, 0, 1), c_res=720)
            edges, anchors, max_radius = surf_clipping(surface, beta, alpha, float(x), float(y), mooring_type, int(n_moorings), plot)
            foot_print.append({'mooring_footprint': edges, 'anchors': anchors, 'max_radius': max_radius})

    return foot_print


# Example wrapper preserved (adapt to your project environment)
def moorings_footprint(x, y, site, beta, resolution_factor, max_d, mooring_type, n_moorings, plot):
    # finds wd with higher frequency
    alpha = site.ds.Sector_frequency.wd.data[np.argmax(site.ds.Sector_frequency.data)]
    return seabed_features(
        site,
        footprint(site.water_depth, site.bounds_shape.crs, np.array(x), np.array(y), beta, alpha,
                  max_d, resolution_factor, mooring_type, n_moorings, plot),
        label_col='seabed',          # opzionale; autodetect se mancante
        default_value='Nodata',      # etichetta per punti non assegnati
        repair_geometries=True,      # ripara poligoni invalsi
        prefer_within_then_intersects=True
    )
