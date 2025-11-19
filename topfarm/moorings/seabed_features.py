# -*- coding: utf-8 -*-
"""
Refactored, robust, and efficient seabed feature assignment.

This module assigns a seabed type to each anchor of each turbine footprint by
building a single GeoDataFrame of all anchor points and performing a single
spatial join against the seabed polygons.

Key improvements
----------------
- Avoids nested Python loops with per-feature Shapely checks (slow/fragile).
- Uses GeoPandas spatial index via sjoin (fast, vectorized).
- Handles CRS alignment, invalid geometries, and boundary cases robustly.
- Writes results back into the original `foot_print` structure.

Assumptions
-----------
- `site.seabeds` is a GeoDataFrame with a valid geometry column and CRS.
- A label column exists (default 'seabed'); fallback to common alternatives
  ('Feature', 'name', 'NAME').
- Each `anchor` dict contains `coords` shaped (1,3) or (3,), where coords[0] is (x,y,z).

"""
from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def _detect_label_column(gdf: gpd.GeoDataFrame, preferred: str = 'seabed') -> str:
    """Return the name of the label column in `gdf`.

    Prefers `preferred`, else tries common fallbacks, else raises.
    """
    if preferred in gdf.columns:
        return preferred
    for alt in ('Feature', 'feature', 'SEABED', 'Seabed', 'name', 'NAME', 'Name'):
        if alt in gdf.columns:
            return alt
    raise ValueError(
        "No label column found in seabeds GeoDataFrame. Expected one of: "
        f"'{preferred}', 'Feature', 'feature', 'SEABED', 'Seabed', 'name', 'NAME', 'Name'"
    )


def _collect_anchors(foot_print: List[Dict[str, Any]]) -> gpd.GeoDataFrame:
    """Collect all anchors into a GeoDataFrame with (turbine_id, anchor_id, geometry, z)."""
    rows: List[Dict[str, Any]] = []
    for t_id, item in enumerate(foot_print):
        anchors = item.get('anchors', []) or []
        for a_id, anchor in enumerate(anchors):
            coords = anchor.get('coords', None)
            if coords is None:
                continue
            arr = np.asarray(coords, dtype=float).reshape(-1, 3)
            if arr.size < 3:
                continue
            x, y, z = arr[0]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            rows.append({
                'turbine_id': t_id,
                'anchor_id': a_id,
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'geometry': Point(float(x), float(y)),
            })
    if not rows:
        return gpd.GeoDataFrame(columns=['turbine_id', 'anchor_id', 'x', 'y', 'z', 'geometry'], geometry='geometry')
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs=None)
    return gdf


def seabed_features(site, foot_print: List[Dict[str, Any]], *,
                    label_col: str = 'seabed', default_value: str = 'Nodata',
                    repair_geometries: bool = True, prefer_within_then_intersects: bool = True) -> List[Dict[str, Any]]:
    """Assign seabed labels to each anchor in `foot_print`.

    Parameters
    ----------
    site : object
        Must have attribute `seabeds`: GeoDataFrame of polygonal seabed features.
    foot_print : list of dict
        As returned by the footprint generator. Each item has an 'anchors' list.
    label_col : str, default 'seabed'
        Column name in `site.seabeds` carrying the label. If missing, common
        alternatives are tried.
    default_value : str, default 'Nodata'
        Label to assign when an anchor does not fall in any seabed polygon.
    repair_geometries : bool, default True
        If True, fixes invalid seabed geometries with a zero-width buffer.
    prefer_within_then_intersects : bool, default True
        If True, first try a 'within' spatial join, then fill remaining anchors
        with an 'intersects' join (helps anchors lying exactly on boundaries).

    Returns
    -------
    foot_print : original structure with anchor['seabed'] set to the found label
                 (or `default_value` when none).
    """
    seabeds: gpd.GeoDataFrame = getattr(site, 'seabeds', None)
    if seabeds is None or seabeds.empty:
        LOG.warning("site.seabeds is empty or missing; assigning default '%s' to all anchors.", default_value)
        for item in foot_print:
            for anchor in item.get('anchors', []) or []:
                anchor['seabed'] = default_value
        return foot_print

    # Detect label column (allows alternative names)
    try:
        label_col = _detect_label_column(seabeds, preferred=label_col)
    except ValueError as e:
        LOG.warning("%s Using index as label.", e)
        seabeds = seabeds.copy()
        seabeds['__label__'] = seabeds.index.astype(str)
        label_col = '__label__'

    # Geometry/CRS housekeeping
    seabeds = seabeds.copy()
    if repair_geometries:
        try:
            invalid = ~seabeds.is_valid
            if invalid.any():
                LOG.info("Repairing %d invalid seabed geometries via buffer(0).", int(invalid.sum()))
                seabeds.loc[invalid, 'geometry'] = seabeds.loc[invalid, 'geometry'].buffer(0)
        except Exception as e:
            LOG.warning("Geometry repair failed: %s", e)

    anchors_gdf = _collect_anchors(foot_print)
    if anchors_gdf.empty:
        LOG.warning("No anchors found in foot_print; nothing to label.")
        return foot_print

    # Align CRS: assume anchors are in the same CRS as seabeds; if seabeds.crs exists, set anchors' CRS
    if seabeds.crs is not None:
        if anchors_gdf.crs is None:
            anchors_gdf.set_crs(seabeds.crs, inplace=True)
        elif anchors_gdf.crs != seabeds.crs:
            anchors_gdf = anchors_gdf.to_crs(seabeds.crs)

    # Primary spatial join: points within polygons
    try:
        if prefer_within_then_intersects:
            joined = gpd.sjoin(anchors_gdf, seabeds[[label_col, 'geometry']], how='left', predicate='within')
            # For points on boundaries (no match), try intersects
            missing_mask = joined[label_col].isna()
            if missing_mask.any():
                LOG.info("%d anchors on boundaries or outside; retrying with 'intersects' for those.", int(missing_mask.sum()))
                missing = anchors_gdf.loc[missing_mask]
                joined2 = gpd.sjoin(missing, seabeds[[label_col, 'geometry']], how='left', predicate='intersects')
                joined.loc[missing.index, label_col] = joined2[label_col]
        else:
            joined = gpd.sjoin(anchors_gdf, seabeds[[label_col, 'geometry']], how='left', predicate='intersects')
    except TypeError:
        # GeoPandas < 0.10 uses 'op' instead of 'predicate'
        LOG.info("Falling back to legacy sjoin signature (op='within'/'intersects').")
        if prefer_within_then_intersects:
            joined = gpd.sjoin(anchors_gdf, seabeds[[label_col, 'geometry']], how='left', op='within')
            missing_mask = joined[label_col].isna()
            if missing_mask.any():
                missing = anchors_gdf.loc[missing_mask]
                joined2 = gpd.sjoin(missing, seabeds[[label_col, 'geometry']], how='left', op='intersects')
                joined.loc[missing.index, label_col] = joined2[label_col]
        else:
            joined = gpd.sjoin(anchors_gdf, seabeds[[label_col, 'geometry']], how='left', op='intersects')

    # Map results back to foot_print
    joined[label_col] = joined[label_col].astype(object).fillna(default_value)
    
    # Se un'ancora cade in più poligoni, tieni la prima occorrenza
    # (puoi cambiare la regola: per es. ordinare per un campo di priorità prima di droppare)
    df = joined.loc[:, ['turbine_id', 'anchor_id', label_col]].copy()
    df['turbine_id'] = df['turbine_id'].astype(int)
    df['anchor_id']  = df['anchor_id'].astype(int)
    df = df.drop_duplicates(subset=['turbine_id', 'anchor_id'], keep='first')
    
    # Costruisci la lookup senza itertuples (niente problemi di nomi/underscore)
    keys = list(zip(df['turbine_id'].to_numpy(), df['anchor_id'].to_numpy()))
    vals = df[label_col].astype(str).to_numpy()
    lookup = dict(zip(keys, vals))
    for t_id, item in enumerate(foot_print):
        anchors = item.get('anchors', []) or []
        for a_id, anchor in enumerate(anchors):
            anchor['seabed'] = lookup.get((t_id, a_id), default_value)

    return foot_print
