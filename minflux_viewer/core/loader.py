"""
minflux_viewer.core.loader
==========================
Load and parse MINFLUX ``.mat`` files into a :class:`MinfluxDataset`.

Ported from MATLAB:

* ``arrange_MINFLUX_data_structure.m``
* ``data_parser_m2410.m``
* ``validate_raw_data.m``
* ``compute_extra_property.m``  (derived attributes section)

Supported formats
-----------------
``m2205``
    Abberior Imspector export prior to 2024.
    ``itr`` is a nested struct; ``loc`` lives inside it.

``m2410``
    Abberior Imspector export from 2024 onwards.
    ``itr`` is a flat integer array (per-localisation iteration index).

``m2205-unwrapped``
    Already-flattened variants produced by SMAP / Ries-lab export scripts.

Both MATLAB v5/v6 (``scipy.io``) and v7.3 HDF5 (``h5py``) files are
supported. Marked MINFLUX Viewer Zarr v2 directories are also supported; raw,
MBM/search and processing components restore into the canonical model.
"""

from __future__ import annotations

import json as json
import threading as threading
from datetime import datetime as datetime
from pathlib import Path as Path
from typing import Any as Any

from minflux_viewer.core.loader_format_readers import (
    _CANONICAL_CSV_BOOL_FIELDS as _CANONICAL_CSV_BOOL_FIELDS,
)
from minflux_viewer.core.loader_format_readers import (
    _CANONICAL_CSV_I32_FIELDS as _CANONICAL_CSV_I32_FIELDS,
)
from minflux_viewer.core.loader_format_readers import (
    _CANONICAL_CSV_I64_FIELDS as _CANONICAL_CSV_I64_FIELDS,
)
from minflux_viewer.core.loader_format_readers import (
    _CANONICAL_CSV_U8_FIELDS as _CANONICAL_CSV_U8_FIELDS,
)
from minflux_viewer.core.loader_format_readers import (
    _CSV_COL_ALIASES as _CSV_COL_ALIASES,
)
from minflux_viewer.core.loader_format_readers import (
    _FLAT_X as _FLAT_X,
)
from minflux_viewer.core.loader_format_readers import (
    _FLAT_Y as _FLAT_Y,
)
from minflux_viewer.core.loader_format_readers import (
    _FLAT_Z as _FLAT_Z,
)
from minflux_viewer.core.loader_format_readers import (
    _canonical_csv_dtype as _canonical_csv_dtype,
)
from minflux_viewer.core.loader_format_readers import (
    _extract_json_records as _extract_json_records,
)
from minflux_viewer.core.loader_format_readers import (
    _flat_table_from_dict as _flat_table_from_dict,
)
from minflux_viewer.core.loader_format_readers import (
    _h5_to_dict as _h5_to_dict,
)
from minflux_viewer.core.loader_format_readers import (
    _iter_json_array_records as _iter_json_array_records,
)
from minflux_viewer.core.loader_format_readers import (
    _load_json_streaming_canonical as _load_json_streaming_canonical,
)
from minflux_viewer.core.loader_format_readers import (
    _load_mat as _load_mat,
)
from minflux_viewer.core.loader_format_readers import (
    _looks_like_minflux_json_records as _looks_like_minflux_json_records,
)
from minflux_viewer.core.loader_format_readers import (
    load_csv as load_csv,
)
from minflux_viewer.core.loader_format_readers import (
    load_dataset as load_dataset,
)
from minflux_viewer.core.loader_format_readers import (
    load_from_mfx_array as load_from_mfx_array,
)
from minflux_viewer.core.loader_format_readers import (
    load_json as load_json,
)
from minflux_viewer.core.loader_format_readers import (
    load_npy as load_npy,
)
from minflux_viewer.core.loader_format_readers import (
    load_npz as load_npz,
)
from minflux_viewer.core.loader_format_readers import (
    load_tiff as load_tiff,
)
from minflux_viewer.core.loader_format_readers import (
    load_zarr as load_zarr,
)
from minflux_viewer.core.loader_metadata_recipe import (
    _restore_filter_mask as _restore_filter_mask,
)
from minflux_viewer.core.loader_metadata_recipe import (
    _transform_record_from_recipe as _transform_record_from_recipe,
)
from minflux_viewer.core.loader_metadata_recipe import (
    apply_metadata_recipe as apply_metadata_recipe,
)
from minflux_viewer.core.loader_metadata_recipe import (
    apply_metadata_sidecar as apply_metadata_sidecar,
)
from minflux_viewer.core.loader_metadata_recipe import (
    apply_metadata_sidecar_file as apply_metadata_sidecar_file,
)
from minflux_viewer.core.loader_metadata_recipe import (
    read_metadata_file as read_metadata_file,
)
from minflux_viewer.core.loader_metadata_recipe import (
    read_metadata_sidecar as read_metadata_sidecar,
)
from minflux_viewer.core.loader_normalization import (
    _FIELD_COPY_BLOCK_ROWS as _FIELD_COPY_BLOCK_ROWS,
)
from minflux_viewer.core.loader_normalization import (
    SPATIAL_COORD_FIELDS as SPATIAL_COORD_FIELDS,
)
from minflux_viewer.core.loader_normalization import (
    _add_derived_attributes as _add_derived_attributes,
)
from minflux_viewer.core.loader_normalization import (
    _add_mfx_attr as _add_mfx_attr,
)
from minflux_viewer.core.loader_normalization import (
    _build_channel as _build_channel,
)
from minflux_viewer.core.loader_normalization import (
    _build_mfx_raw_from_json_records as _build_mfx_raw_from_json_records,
)
from minflux_viewer.core.loader_normalization import (
    _build_mfx_raw_from_raw as _build_mfx_raw_from_raw,
)
from minflux_viewer.core.loader_normalization import (
    _build_mfx_raw_m2205_unwrapped as _build_mfx_raw_m2205_unwrapped,
)
from minflux_viewer.core.loader_normalization import (
    _build_mfx_raw_m2410 as _build_mfx_raw_m2410,
)
from minflux_viewer.core.loader_normalization import (
    _compute_properties as _compute_properties,
)
from minflux_viewer.core.loader_normalization import (
    _dcr_is_two_channel as _dcr_is_two_channel,
)
from minflux_viewer.core.loader_normalization import (
    _detect_raw_version as _detect_raw_version,
)
from minflux_viewer.core.loader_normalization import (
    _dtype_base_and_shape as _dtype_base_and_shape,
)
from minflux_viewer.core.loader_normalization import (
    _effective_iteration_column as _effective_iteration_column,
)
from minflux_viewer.core.loader_normalization import (
    _effective_iteration_flat as _effective_iteration_flat,
)
from minflux_viewer.core.loader_normalization import (
    _effective_iteration_metadata as _effective_iteration_metadata,
)
from minflux_viewer.core.loader_normalization import (
    _is_legacy_nested_itr_mfx as _is_legacy_nested_itr_mfx,
)
from minflux_viewer.core.loader_normalization import (
    _iteration_count as _iteration_count,
)
from minflux_viewer.core.loader_normalization import (
    _json_coordinate_array_m as _json_coordinate_array_m,
)
from minflux_viewer.core.loader_normalization import (
    _json_scalar_array as _json_scalar_array,
)
from minflux_viewer.core.loader_normalization import (
    _json_values as _json_values,
)
from minflux_viewer.core.loader_normalization import (
    _json_vector_array as _json_vector_array,
)
from minflux_viewer.core.loader_normalization import (
    _mat_struct_to_dict as _mat_struct_to_dict,
)
from minflux_viewer.core.loader_normalization import (
    _mfx_columns as _mfx_columns,
)
from minflux_viewer.core.loader_normalization import (
    _normalize_mfx_attrs as _normalize_mfx_attrs,
)
from minflux_viewer.core.loader_normalization import (
    _parse_m2205 as _parse_m2205,
)
from minflux_viewer.core.loader_normalization import (
    _parse_m2205_unwrapped as _parse_m2205_unwrapped,
)
from minflux_viewer.core.loader_normalization import (
    _parse_m2410 as _parse_m2410,
)
from minflux_viewer.core.loader_normalization import (
    _parse_raw as _parse_raw,
)
from minflux_viewer.core.loader_normalization import (
    _raw_iteration_count as _raw_iteration_count,
)
from minflux_viewer.core.loader_normalization import (
    _raw_localization_count as _raw_localization_count,
)
from minflux_viewer.core.loader_normalization import (
    _source_metadata_from_raw as _source_metadata_from_raw,
)
from minflux_viewer.core.loader_raw_access import (
    _COORD_NM_BASE as _COORD_NM_BASE,
)
from minflux_viewer.core.loader_raw_access import (
    _DERIVED_BROADCAST_ATTRS as _DERIVED_BROADCAST_ATTRS,
)
from minflux_viewer.core.loader_raw_access import (
    _EFFECTIVE_ITER_META as _EFFECTIVE_ITER_META,
)
from minflux_viewer.core.loader_raw_access import (
    _LOC_ID_PUBLISH_LOCK as _LOC_ID_PUBLISH_LOCK,
)
from minflux_viewer.core.loader_raw_access import (
    VALUE_POOL_SELECTORS as VALUE_POOL_SELECTORS,
)
from minflux_viewer.core.loader_raw_access import (
    _attr_row_selection as _attr_row_selection,
)
from minflux_viewer.core.loader_raw_access import (
    _attrs_include_invalid as _attrs_include_invalid,
)
from minflux_viewer.core.loader_raw_access import (
    _attrs_valid_count as _attrs_valid_count,
)
from minflux_viewer.core.loader_raw_access import (
    _broadcast_derived as _broadcast_derived,
)
from minflux_viewer.core.loader_raw_access import (
    _broadcast_iteration_to_rows as _broadcast_iteration_to_rows,
)
from minflux_viewer.core.loader_raw_access import (
    _derived_last_from_raw as _derived_last_from_raw,
)
from minflux_viewer.core.loader_raw_access import (
    _gather_iteration_values as _gather_iteration_values,
)
from minflux_viewer.core.loader_raw_access import (
    _mfx_get_rows as _mfx_get_rows,
)
from minflux_viewer.core.loader_raw_access import (
    _mfx_raw_len as _mfx_raw_len,
)
from minflux_viewer.core.loader_raw_access import (
    _pooled_loc_values as _pooled_loc_values,
)
from minflux_viewer.core.loader_raw_access import (
    _raw_loc_id as _raw_loc_id,
)
from minflux_viewer.core.loader_raw_access import (
    _selection_rows_match_attr as _selection_rows_match_attr,
)
from minflux_viewer.core.loader_raw_access import (
    _spec_filter_values as _spec_filter_values,
)
from minflux_viewer.core.loader_raw_access import (
    _values_on_attr_rows as _values_on_attr_rows,
)
from minflux_viewer.core.loader_raw_access import (
    apply_saved_filters as apply_saved_filters,
)
from minflux_viewer.core.loader_raw_access import (
    attr_matches_last_valid as attr_matches_last_valid,
)
from minflux_viewer.core.loader_raw_access import (
    attr_matches_selection as attr_matches_selection,
)
from minflux_viewer.core.loader_raw_access import (
    attr_materialized_selection as attr_materialized_selection,
)
from minflux_viewer.core.loader_raw_access import (
    attr_values_1d as attr_values_1d,
)
from minflux_viewer.core.loader_raw_access import (
    attr_values_for_selection as attr_values_for_selection,
)
from minflux_viewer.core.loader_raw_access import (
    effective_attr_values_1d as effective_attr_values_1d,
)
from minflux_viewer.core.loader_raw_access import (
    effective_iteration_for_attr as effective_iteration_for_attr,
)
from minflux_viewer.core.loader_raw_access import (
    effective_iterations_for_attr as effective_iterations_for_attr,
)
from minflux_viewer.core.loader_raw_access import (
    evaluate_saved_filters as evaluate_saved_filters,
)
from minflux_viewer.core.loader_raw_access import (
    is_value_pool_selector as is_value_pool_selector,
)
from minflux_viewer.core.loader_raw_access import (
    iteration_attr_values_1d as iteration_attr_values_1d,
)
from minflux_viewer.core.loader_raw_access import (
    last_valid_attr_values as last_valid_attr_values,
)
from minflux_viewer.core.loader_raw_access import (
    mfx_filter_mask as mfx_filter_mask,
)
from minflux_viewer.core.loader_raw_access import (
    mfx_get as mfx_get,
)
from minflux_viewer.core.loader_raw_access import (
    mfx_row_mask as mfx_row_mask,
)
from minflux_viewer.core.loader_raw_access import (
    resolve_spec_iteration as resolve_spec_iteration,
)

from .dataset import AttrStore as AttrStore
from .dataset import Calibration as Calibration
from .dataset import Channel as Channel
from .dataset import DataProp as DataProp
from .dataset import FileInfo as FileInfo
from .dataset import MinfluxDataset as MinfluxDataset
from .dataset import build_image_dataset as build_image_dataset
from .dataset import build_localization_dataset as build_localization_dataset
from .loader_format_readers import np as np
from .loader_format_readers import sio as sio
