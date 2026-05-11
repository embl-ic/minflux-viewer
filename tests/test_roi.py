from pathlib import Path

from minflux_viewer.core.roi import RoiRecord, RoiStore


def _record(name: str, roi_type: str = "rectangle") -> RoiRecord:
    geometry = {"bounds": [0.0, 0.0, 10.0, 12.0]}
    if roi_type == "oval":
        geometry = {"bounds": [1.0, 2.0, 8.0, 9.0]}
    elif roi_type == "polygon":
        geometry = {"points": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]], "closed": True}
    elif roi_type == "line":
        geometry = {"points": [[0.0, 0.0], [5.0, 5.0]], "closed": False}
    elif roi_type == "point":
        geometry = {"point": [3.0, 4.0]}
    return RoiRecord.create(roi_type, geometry, name=name, coordinate_space="pixel")


def test_roi_store_order_selection_rename_delete() -> None:
    store = RoiStore()
    first = _record("first")
    second = _record("second")
    store.add(first)
    store.add(second)

    assert [r.name for r in store.records] == ["first", "second"]
    assert store.selected_ids == [second.id]

    store.select([second.id])
    store.move_selected(-1)
    assert [r.name for r in store.records] == ["second", "first"]

    store.rename(second.id, "renamed")
    assert store.records[0].name == "renamed"

    store.delete_selected()
    assert [r.name for r in store.records] == ["first"]
    assert store.selected_ids == []


def test_roi_show_flags_and_combine() -> None:
    store = RoiStore()
    a = _record("a")
    b = RoiRecord.create("rectangle", {"bounds": [5.0, 5.0, 10.0, 10.0]}, name="b")
    store.add(a)
    store.add(b)
    store.set_show_all(True)
    store.set_show_labels(True)
    store.select([a.id, b.id])
    store.combine_selected()

    assert store.show_all is True
    assert all(r.label_visible for r in store.records)
    assert store.records[-1].name == "Combined"
    assert store.records[-1].type == "polygon"


def test_roi_native_json_roundtrip(tmp_path: Path) -> None:
    store = RoiStore()
    store.add(_record("poly", "polygon"))
    path = tmp_path / "rois.json"
    store.save(path)

    loaded = RoiStore()
    records = loaded.load(path)

    assert len(records) == 1
    assert records[0].name == "poly"
    assert records[0].type == "polygon"


def test_imagej_roi_zip_roundtrip(tmp_path: Path) -> None:
    store = RoiStore()
    for roi_type in ("rectangle", "oval", "polygon", "line", "point"):
        store.add(_record(roi_type, roi_type))

    path = tmp_path / "RoiSet.zip"
    store.save(path)

    loaded = RoiStore()
    records = loaded.load(path)

    assert [r.type for r in records] == ["rectangle", "oval", "polygon", "line", "point"]
