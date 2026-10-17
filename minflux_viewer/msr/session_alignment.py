"""Qt-free alignment over one parsed MSR session; array maps are borrowed, never copied."""
from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np

from minflux_viewer.core.bead_alignment import TRANSFORM_RIGID_XY_Z


class AlignmentSession:
    def __init__(self, parsed: dict, msr_state: SimpleNamespace, *,
                 log: Callable[[str], None], excluded: set[int] | None = None,
                 transform_type: str = TRANSFORM_RIGID_XY_Z) -> None:
        self.parsed = parsed
        self._session_state = msr_state
        self.log = log
        self._bead_unchecked_gris = excluded or set()
        self._transform_type = transform_type

    def _msr_state(self):
        return self._session_state

    def _mbm_transform_type(self) -> str:
        return self._transform_type

    def _viewer_selected_dataset_names(self, import_plan: dict[str, set[str] | None]) -> list[str]:
        return [
            ds.get("display_name") or ds.get("did") or "dataset"
            for ds in self.parsed.get("datasets", [])
            if (ds.get("display_name") or ds.get("did") or "dataset") in import_plan
        ]


    def _viewer_zroot_for_dataset_name(self, name: str) -> str | None:
        for ds in self.parsed.get("datasets", []) or []:
            key = ds.get("display_name") or ds.get("did") or "dataset"
            if key == name:
                return ds.get("zroot")
        return None


    def _viewer_mbm_alignment_available(self, selected: list[str]) -> tuple[bool, str]:
        MFSTATE = self._msr_state()
        from minflux_viewer.core.bead_alignment import (
            MIN_BEADS_FOR_ALIGNMENT,
            extract_bead_summaries,
        )

        if len(selected) < 2:
            return False, "fewer than two selected datasets"
        ref_name = selected[0]
        ref_mbm = self._aligned_mbm_points(ref_name)
        if ref_mbm is None:
            return False, f"no bead points for reference dataset {ref_name}"
        # 2 matched beads suffice (translational XYZ); the transform mode is
        # downgraded to what the matched-bead count supports — see
        # alignment.effective_transform_type.
        min_beads = MIN_BEADS_FOR_ALIGNMENT
        try:
            ref_ids = set(extract_bead_summaries(ref_mbm))
        except Exception as exc:
            return False, f"reference bead points are not parseable ({exc})"
        ref_used = self._viewer_used_bead_gri_for_dataset(ref_name)
        if ref_used is not None:
            ref_ids &= ref_used
            if not ref_ids:
                return False, f"reference dataset {ref_name} has no metadata-marked used beads"
        if len(ref_ids) < min_beads:
            return False, f"reference dataset has {len(ref_ids)} usable bead(s); need at least {min_beads}"
        for moving_name in selected[1:]:
            moving_mbm = self._aligned_mbm_points(moving_name)
            if moving_mbm is None:
                return False, f"no bead points for {moving_name}"
            try:
                moving_ids = set(extract_bead_summaries(moving_mbm))
            except Exception as exc:
                return False, f"bead points for {moving_name} are not parseable ({exc})"
            moving_used = self._viewer_used_bead_gri_for_dataset(moving_name)
            if moving_used is not None:
                moving_ids &= moving_used
                if not moving_ids:
                    return False, f"{moving_name} has no metadata-marked used beads"
            matched = len(ref_ids & moving_ids)
            if matched < min_beads:
                return False, f"{moving_name} has {matched} usable bead(s) matching the reference; need at least {min_beads}"
        return True, ""


    def _viewer_used_bead_gri_for_dataset(
        self, name: str, *, log_prefix: str = "[viewer align]"
    ) -> set[int] | None:
        # Prefer translated legacy MBM metadata (gri↔R-ID map + used list).
        MFSTATE = self._msr_state()
        meta = (getattr(MFSTATE, "mbm_meta_map", {}) or {}).get(name)
        if meta:
            pbg = meta.get("points_by_gri") or {}
            used_names = {str(x).strip() for x in (meta.get("used") or []) if str(x).strip()}
            name_to_gri = {}
            for info in pbg.values():
                if isinstance(info, dict) and info.get("name") is not None:
                    try:
                        name_to_gri[str(info["name"])] = int(info["gri"])
                    except (TypeError, ValueError):
                        pass
            if used_names and name_to_gri:
                used_gri = {name_to_gri[n] for n in used_names if n in name_to_gri}
                if used_gri:
                    self.log(f"{log_prefix} {name}: using {len(used_gri)} "
                             f"metadata-marked used bead(s) (legacy MBM)")
                    return used_gri

        zroot = self._viewer_zroot_for_dataset_name(name)
        if not zroot:
            return None
        mbm_attrs = self._read_zarr_attrs_safe(zroot, "mbm") or self._read_zarr_attrs_safe(zroot, "grd/mbm")
        points_attrs = self._read_zarr_attrs_safe(zroot, "grd/mbm/points")

        used_names = self._extract_used_bead_names(mbm_attrs)
        name_to_gri, used_gri_from_points = self._extract_point_metadata_beads(points_attrs)
        if used_names and name_to_gri:
            used_gri = {name_to_gri[name] for name in used_names if name in name_to_gri}
            missing = sorted(name for name in used_names if name not in name_to_gri)
            if missing:
                preview = ", ".join(missing[:5])
                suffix = "..." if len(missing) > 5 else ""
                self.log(f"{log_prefix} {name}: {len(missing)} used bead name(s) not mapped to gri: {preview}{suffix}")
            if used_gri:
                self.log(f"{log_prefix} {name}: using {len(used_gri)} metadata-marked used bead(s)")
                return used_gri
        if used_gri_from_points:
            self.log(f"{log_prefix} {name}: using {len(used_gri_from_points)} bead(s) marked used in points metadata")
            return used_gri_from_points
        return None


    def _read_zarr_attrs_safe(self, zroot: str, path: str) -> dict:
        try:
            from minflux_viewer.msr.io import read_zarr_attrs

            attrs = read_zarr_attrs(zroot, path)
            return attrs if isinstance(attrs, dict) else {}
        except Exception:
            return {}


    def _extract_used_bead_names(self, attrs: dict) -> set[str]:
        if not isinstance(attrs, dict):
            return set()
        used = attrs.get("used")
        if not isinstance(used, (list, tuple, set)):
            return set()
        return {str(name).strip() for name in used if str(name).strip()}


    def _extract_point_metadata_beads(self, attrs: dict) -> tuple[dict[str, int], set[int]]:
        name_to_gri: dict[str, int] = {}
        used_gri: set[int] = set()

        def visit(node) -> None:
            if isinstance(node, dict):
                if "name" in node and "gri" in node:
                    try:
                        gri = int(node.get("gri"))
                    except Exception:
                        gri = None
                    name = str(node.get("name", "")).strip()
                    if gri is not None and name:
                        name_to_gri[name] = gri
                        if bool(node.get("used", False)):
                            used_gri.add(gri)
                for value in node.values():
                    visit(value)
            elif isinstance(node, (list, tuple)):
                for value in node:
                    visit(value)

        visit(attrs)
        return name_to_gri, used_gri


    def _filter_mbm_points_to_used_beads(
        self, name: str, points, *, log_prefix: str = "[viewer align]"
    ) -> np.ndarray:
        arr = np.asarray(points)
        used_gri = self._viewer_used_bead_gri_for_dataset(name, log_prefix=log_prefix)
        if used_gri is None:
            return arr
        names = set(getattr(getattr(arr, "dtype", None), "names", ()) or ())
        if "gri" not in names:
            return arr
        mask = np.isin(np.asarray(arr["gri"], dtype=np.uint32), np.asarray(sorted(used_gri), dtype=np.uint32))
        filtered = arr[mask]
        self.log(f"{log_prefix} {name}: filtered mbm points to {len(filtered)} row(s) from used bead metadata")
        return filtered


    def _viewer_dataset_looks_3d(self, name: str) -> bool:
        MFSTATE = self._msr_state()

        mfx = MFSTATE.mfx_map.get(name)
        try:
            from minflux_viewer.core.loader_normalization import _normalize_mfx_attrs

            attrs, num_raw, _num_itr, _version, _detail = _normalize_mfx_attrs(
                mfx,
                load_all_itr=False,   # always last-valid; browse via mfx_raw
            )
            loc_z = np.asarray(attrs.get("loc_z", np.zeros(num_raw)), dtype=np.float64)
            finite = np.isfinite(loc_z)
            if np.count_nonzero(finite) and np.ptp(loc_z[finite]) > 1e-12:
                return True
        except Exception:
            pass
        mbm = MFSTATE.mbm_map.get(name)
        try:
            names = set(getattr(getattr(mbm, "dtype", None), "names", ()) or ())
            if "xyz" in names:
                xyz = np.asarray(mbm["xyz"], dtype=np.float64)
                if xyz.ndim >= 2 and xyz.shape[-1] >= 3:
                    z = xyz[..., 2]
                    finite = np.isfinite(z)
                    return bool(np.count_nonzero(finite) and np.ptp(z[finite]) > 1e-12)
        except Exception:
            pass
        return False


    def _viewer_channel_alignment_transforms(
        self,
        import_plan: dict[str, set[str] | None],
        mode: str,
        ref_name: str | None = None,
    ) -> dict[str, dict]:
        MFSTATE = self._msr_state()
        from minflux_viewer.core.bead_alignment import align_channels_from_arrays

        selected = self._viewer_selected_dataset_names(import_plan)
        if len(selected) < 2:
            return {}
        # The reference leads the list; both the mbm and fallback paths treat
        # selected[0] as the reference channel.
        if ref_name and ref_name in selected:
            selected = [ref_name] + [name for name in selected if name != ref_name]
        if mode in {"stage origin", "data centroid"}:
            return self._viewer_fallback_alignment_transforms(selected, mode)
        if mode != "mbm info":
            return {}
        ref_name = selected[0]
        ref_mbm = self._aligned_mbm_points(ref_name)
        if ref_mbm is None:
            raise ValueError(f"Could not find bead points (`grd/mbm/points`) for reference channel {ref_name}.")
        ref_mbm_for_alignment = self._filter_mbm_points_to_used_beads(ref_name, ref_mbm)

        transforms: dict[str, dict] = {}
        transforms[ref_name] = self._viewer_transform_record(
            ref_name,
            ref_name,
            np.eye(4, dtype=np.float64),
            method="reference identity",
        )
        gri_to_rid = self._bead_gri_to_rid(selected)
        excluded = sorted(int(g) for g in (self._bead_unchecked_gris or set()))
        self.log(
            f"[viewer align] user-excluded {len(excluded)} bead(s): "
            f"{self._format_bead_ids(excluded, gri_to_rid)}")
        for moving_name in selected[1:]:
            moving_mbm = self._aligned_mbm_points(moving_name)
            if moving_mbm is None:
                raise ValueError(f"Could not find bead points (`grd/mbm/points`) for {moving_name}.")
            moving_mbm_for_alignment = self._filter_mbm_points_to_used_beads(moving_name, moving_mbm)
            result = align_channels_from_arrays(
                ref_name, ref_mbm_for_alignment, moving_name, moving_mbm_for_alignment,
                transform_type=self._mbm_transform_type())
            transforms[moving_name] = {
                "reference_channel": ref_name,
                "moving_channel": moving_name,
                "matrix_3x3": result.matrix_3x3.tolist(),
                "matrix_4x4": result.matrix_4x4.tolist(),
                "translation_nm": result.translation.tolist(),
                "z_translation_nm": float(result.z_translation),
                "rotation_2x2": result.rotation.tolist(),
                "matched_bead_count": int(result.bead_ids.size),
                "rmse_xy_nm": float(result.rmse),
                "method": "mbm xy rigid plus z translation",
            }
            used = [int(b) for b in result.bead_ids]
            self.log(
                f"[viewer align] {moving_name} -> {ref_name}; used {len(used)} bead(s): "
                f"{self._format_bead_ids(used, gri_to_rid)}; "
                f"translation_nm=({result.translation[0]:.3f}, {result.translation[1]:.3f}); "
                f"rmse_xy_nm={result.rmse:.3f}"
            )
        return transforms


    def _viewer_fallback_alignment_transforms(self, selected: list[str], mode: str) -> dict[str, dict]:
        MFSTATE = self._msr_state()

        if not selected:
            return {}
        ref_name = selected[0]
        ref_centroid = self._mfx_centroid_nm(MFSTATE.mfx_map.get(ref_name))
        transforms: dict[str, dict] = {}
        for name in selected:
            matrix = np.eye(4, dtype=np.float64)
            if name != ref_name and mode == "data centroid":
                moving_centroid = self._mfx_centroid_nm(MFSTATE.mfx_map.get(name))
                matrix[:3, 3] = ref_centroid - moving_centroid
            method = "stage origin" if mode == "stage origin" else "median localization centroid"
            transforms[name] = self._viewer_transform_record(ref_name, name, matrix, method=method)
            t = matrix[:3, 3]
            self.log(
                f"[viewer align] fallback {name} -> {ref_name}; method={method}; "
                f"translation_nm=({t[0]:.3f}, {t[1]:.3f})"
            )
        return transforms


    def _viewer_transform_record(self, ref_name: str, moving_name: str, matrix_4x4: np.ndarray, *, method: str) -> dict:
        matrix = np.asarray(matrix_4x4, dtype=np.float64)
        matrix_3x3 = np.eye(3, dtype=np.float64)
        matrix_3x3[:2, :2] = matrix[:2, :2]
        matrix_3x3[:2, 2] = matrix[:2, 3]
        return {
            "reference_channel": ref_name,
            "moving_channel": moving_name,
            "matrix_3x3": matrix_3x3.tolist(),
            "matrix_4x4": matrix.tolist(),
            "translation_nm": matrix[:2, 3].astype(float).tolist(),
            "z_translation_nm": float(matrix[2, 3]),
            "rotation_2x2": matrix[:2, :2].astype(float).tolist(),
            "matched_bead_count": 0,
            "rmse_xy_nm": 0.0,
            "method": method,
        }


    def _mfx_centroid_nm(self, mfx) -> np.ndarray:
        if mfx is None:
            return np.zeros(3, dtype=np.float64)
        try:
            from minflux_viewer.core.loader_normalization import _normalize_mfx_attrs

            attrs, num_raw, _num_itr, _version, _detail = _normalize_mfx_attrs(
                mfx,
                load_all_itr=False,   # always last-valid; browse via mfx_raw
            )
            loc_x = np.asarray(attrs.get("loc_x", np.full(num_raw, np.nan)), dtype=np.float64).ravel()
            loc_y = np.asarray(attrs.get("loc_y", np.full(num_raw, np.nan)), dtype=np.float64).ravel()
            loc_z = np.asarray(attrs.get("loc_z", np.zeros_like(loc_x)), dtype=np.float64).ravel()
            mask = np.isfinite(loc_x) & np.isfinite(loc_y) & np.isfinite(loc_z)
            if "vld" in attrs:
                vld = np.asarray(attrs["vld"], dtype=bool).ravel()
                if vld.size == mask.size:
                    mask &= vld
            if not np.any(mask):
                return np.zeros(3, dtype=np.float64)
            return np.nanmedian(np.column_stack([loc_x[mask], loc_y[mask], loc_z[mask]]) * 1e9, axis=0)
        except Exception:
            return np.zeros(3, dtype=np.float64)


    def _aligned_mbm_points(self, name):
        """Bead points for *name* with beads excluded via "Show Beads Drift"
        removed. An empty selection (the default) returns all beads unchanged, so
        the alignment behaves exactly as before unless the user made a choice."""
        MFSTATE = self._msr_state()

        points = MFSTATE.mbm_map.get(name)
        if not self._bead_unchecked_gris or points is None:
            return points
        names = getattr(getattr(points, "dtype", None), "names", ()) or ()
        if "gri" not in names:
            return points
        gri = np.asarray(points["gri"]).ravel()
        # Never turn a failed restriction into permission to use every bead.
        return points[~np.isin(gri, list(self._bead_unchecked_gris))]


    def _bead_gri_to_rid(self, names) -> dict[int, str]:
        """Map ``gri`` (numeric bead ID) → R-ID (the ``points_by_gri`` "name",
        e.g. ``R113``) across *names*, for human-readable bead reporting."""
        MFSTATE = self._msr_state()
        meta_map = getattr(MFSTATE, "mbm_meta_map", {}) or {}
        out: dict[int, str] = {}
        for name in names:
            pbg = (meta_map.get(name) or {}).get("points_by_gri") or {}
            if not pbg:
                zroot = self._viewer_zroot_for_dataset_name(name)
                if not zroot:
                    continue
                attrs = self._read_zarr_attrs_safe(zroot, "grd/mbm/points")
                pbg = attrs.get("points_by_gri", {}) if isinstance(attrs, dict) else {}
            for key, info in (pbg or {}).items():
                if not isinstance(info, dict):
                    continue
                try:
                    gri = int(info.get("gri", key))
                except (TypeError, ValueError):
                    continue
                rid = info.get("name")
                if rid:
                    out.setdefault(gri, str(rid))
        return out


    @staticmethod
    def _format_bead_ids(gris, gri_to_rid: dict[int, str] | None = None) -> str:
        """``"12 (R113), 27 (R140)"`` — gri plus R-ID when known; ``(none)`` if empty."""
        gri_to_rid = gri_to_rid or {}
        parts = []
        for g in sorted({int(x) for x in (gris or [])}):
            rid = gri_to_rid.get(g)
            parts.append(f"{g} ({rid})" if rid else str(g))
        return ", ".join(parts) if parts else "(none)"


    def _mbm_naming_metadata(self, key: str, zroot) -> tuple[dict, list]:
        """``(points_by_gri, used)`` for one dataset — the gri↔R-ID map and the
        beads the acquisition marked as used.

        Same two sources as :func:`beads_drift.gather_msr_bead_drift`: the
        translated legacy MBM metadata first, then the modern in-store zarr attrs.
        """
        MFSTATE = self._msr_state()
        meta = (getattr(MFSTATE, "mbm_meta_map", None) or {}).get(key) or {}
        pbg = meta.get("points_by_gri") or {}
        used = meta.get("used") or []
        if (not pbg or not used) and zroot is not None:
            try:
                if not pbg:
                    pbg = (self._read_zarr_attrs_safe(zroot, "grd/mbm/points") or {}).get(
                        "points_by_gri", {})
                if not used:
                    used = (self._read_zarr_attrs_safe(zroot, "mbm") or {}).get("used", [])
            except Exception:
                pass
        return dict(pbg or {}), list(used or [])


    def _combined_loc_bounds_nm(self, names) -> tuple[np.ndarray, np.ndarray] | None:
        """Combined raw-localization bounding box (nm) across *names*, for the
        alignment plot's data-range overlay.

        Surveys each dataset's raw ``mfx['loc']`` with no validity check: an
        ``(Nloc, Nitr, 3)`` grid uses the last iteration, a flat m2410
        ``(N, 2/3)`` array is used as-is. Returns ``(min_xyz, max_xyz)`` in nm
        (z padded with 0 for 2-D data), or ``None`` when no coordinates exist.
        """
        MFSTATE = self._msr_state()

        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for name in names:
            mfx = MFSTATE.mfx_map.get(name)
            fields = set(getattr(getattr(mfx, "dtype", None), "names", ()) or ())
            if "loc" not in fields:
                continue
            loc = np.asarray(mfx["loc"], dtype=np.float64)
            if loc.ndim == 3 and loc.shape[-1] >= 2:
                loc = loc[:, -1, :]  # last iteration of an (Nloc, Nitr, 3) grid
            if loc.ndim != 2 or loc.shape[1] < 2:
                continue
            cols = min(3, loc.shape[1])
            xyz = np.zeros((loc.shape[0], 3), dtype=np.float64)
            xyz[:, :cols] = loc[:, :cols]
            xyz *= 1e9
            xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
            if xyz.size == 0:
                continue
            mins.append(xyz.min(axis=0))
            maxs.append(xyz.max(axis=0))
        if not mins:
            return None
        return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)
