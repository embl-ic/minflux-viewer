import numpy as np
import pytest

from minflux_viewer.analysis.shape_segmentation import (
    SHAPE_MODELS,
    ScalarField,
    ShapeInstance,
    ShapePrior,
    ShapeSegmentationConfig,
    field_from_points,
    get_shape_model,
    instance_mask,
    instance_outline,
    otsu_threshold,
    segment_shapes,
    segment_shapes_in_image,
    segment_shapes_in_points,
)

PIXEL = 20.0
CAPSULE_PRIOR = ShapePrior.capsule(length_nm=(1400.0, 3400.0),
                                   width_nm=(700.0, 1150.0))


def _instance(center, angle, length, width, model="capsule"):
    size = (length, width, 0.0) if model == "arc_capsule" else (length, width)
    return ShapeInstance(model, center, angle, size, 1.0, 1.0, False, 1.0, 1)


def _scene(specs, shape=(320, 340), pixel_nm=PIXEL, density=0.45, seed=17):
    """A sparse, noisy density image holding the given capsules."""
    rng = np.random.default_rng(seed)
    image = np.zeros(shape, dtype=float)
    for spec in specs:
        mask = instance_mask(_instance(*spec), shape, pixel_nm)
        keep = mask & (rng.random(shape) < density)
        image[keep] += rng.uniform(0.7, 1.0, int(keep.sum()))
    image[rng.random(shape) < 0.002] = 0.5      # sparse background specks
    return image


def _nearest(result, center):
    centers = np.asarray([item.center_nm for item in result.instances])
    return float(np.linalg.norm(centers - np.asarray(center), axis=1).min())


# --------------------------------------------------------------------------- #
# Shape models
# --------------------------------------------------------------------------- #
def test_every_registered_model_round_trips_its_own_geometry():
    """Each SDF must be negative inside, positive outside, and match its area."""
    grid = np.linspace(-3000.0, 3000.0, 240)
    xx, yy = np.meshgrid(grid, grid)
    cell = (grid[1] - grid[0]) ** 2
    sizes = {"capsule": (2400.0, 900.0), "arc_capsule": (2400.0, 900.0, 0.0),
             "ellipse": (2400.0, 900.0), "rectangle": (2400.0, 900.0),
             "disk": (1600.0,)}
    for key, model in SHAPE_MODELS.items():
        size = sizes[key]
        assert len(size) == model.n_size
        sdf = model.sdf(xx, yy, 0.0, 0.0, 0.0, size)
        assert sdf[len(grid) // 2, len(grid) // 2] < 0     # centre is inside
        assert sdf[0, 0] > 0                                # corner is outside
        rasterised = float(np.count_nonzero(sdf < 0) * cell)
        assert rasterised == pytest.approx(model.area(size), rel=0.03)
        outline = model.outline(0.0, 0.0, 0.0, size, 64)
        assert outline.ndim == 2 and outline.shape[1] == 2
        assert np.allclose(outline[0], outline[-1])         # closed


def test_shapes_rotate_rigidly():
    """Rotating the model must move it, not deform it."""
    grid = np.linspace(-3000.0, 3000.0, 200)
    xx, yy = np.meshgrid(grid, grid)
    for key, model in SHAPE_MODELS.items():
        if model.isotropic:
            continue
        size = (2400.0, 900.0, 0.0) if model.n_size == 3 else (2400.0, 900.0)
        flat = int(np.count_nonzero(model.sdf(xx, yy, 0.0, 0.0, 0.0, size) < 0))
        turned = int(np.count_nonzero(model.sdf(xx, yy, 0.0, 0.0, 37.0, size) < 0))
        assert turned == pytest.approx(flat, rel=0.02)


def test_bent_rod_with_zero_bend_is_exactly_the_straight_capsule():
    grid = np.linspace(-2500.0, 2500.0, 160)
    xx, yy = np.meshgrid(grid, grid)
    straight = get_shape_model("capsule").sdf(xx, yy, 0.0, 0.0, 25.0, (2400.0, 900.0))
    bent = get_shape_model("arc_capsule").sdf(xx, yy, 0.0, 0.0, 25.0,
                                              (2400.0, 900.0, 0.0))
    assert np.allclose(straight, bent)


def test_bent_rod_conserves_area_and_leaves_the_straight_footprint():
    """A bend must move the rod off its straight axis without changing its area."""
    grid = np.linspace(-3000.0, 3000.0, 300)
    xx, yy = np.meshgrid(grid, grid)
    model = get_shape_model("arc_capsule")
    cell = (grid[1] - grid[0]) ** 2
    straight = model.sdf(xx, yy, 0.0, 0.0, 0.0, (2400.0, 900.0, 0.0)) < 0
    bent = model.sdf(xx, yy, 0.0, 0.0, 0.0, (2400.0, 900.0, 70.0)) < 0
    assert float(bent.sum() * cell) == pytest.approx(
        model.area((2400.0, 900.0, 70.0)), rel=0.05)
    assert int((bent & ~straight).sum()) > 0.15 * int(straight.sum())


# --------------------------------------------------------------------------- #
# Prior
# --------------------------------------------------------------------------- #
def test_prior_rejects_a_size_tuple_that_does_not_match_the_model():
    with pytest.raises(ValueError, match="size parameter"):
        ShapePrior("capsule", (100.0,), (200.0,)).validate()


def test_prior_rejects_inverted_and_non_positive_bounds():
    with pytest.raises(ValueError, match="below lower bound"):
        ShapePrior("capsule", (2000.0, 700.0), (1000.0, 900.0)).validate()
    with pytest.raises(ValueError, match="must be positive"):
        ShapePrior("capsule", (0.0, 700.0), (2000.0, 900.0)).validate()


def test_a_signed_parameter_may_span_zero():
    """bend_deg is an angle, not a length — a negative bound is legitimate."""
    ShapePrior.arc_capsule(length_nm=(1400.0, 3000.0), width_nm=(700.0, 1100.0),
                           bend_deg=(-60.0, 60.0)).validate()


def test_unknown_model_names_the_known_ones():
    with pytest.raises(ValueError, match="capsule"):
        get_shape_model("banana")


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def test_isolated_objects_are_found_with_their_geometry():
    specs = [((2200.0, 1500.0), 20.0, 2600.0, 900.0),
             ((4600.0, 4600.0), -55.0, 2000.0, 800.0)]
    result = segment_shapes_in_image(_scene(specs), PIXEL, prior=CAPSULE_PRIOR)
    assert len(result) == 2
    for center, angle, length, width in specs:
        assert _nearest(result, center) < 250.0
        best = min(result.instances,
                   key=lambda item: np.hypot(item.center_nm[0] - center[0],
                                             item.center_nm[1] - center[1]))
        assert best.size_nm[0] == pytest.approx(length, abs=450.0)
        assert best.size_nm[1] == pytest.approx(width, abs=250.0)
        assert abs(((best.angle_deg - angle + 90.0) % 180.0) - 90.0) < 12.0
        assert best.iou > 0.75


def test_two_objects_touching_side_by_side_are_separated():
    """The case erosion/watershed cannot split: parallel rods with no waist."""
    specs = [((2400.0, 3200.0), 90.0, 2600.0, 900.0),
             ((3250.0, 3200.0), 90.0, 2600.0, 900.0)]
    result = segment_shapes_in_image(_scene(specs), PIXEL, prior=CAPSULE_PRIOR)
    assert len(result) == 2
    assert {item.component_id for item in result.instances} == {1}   # one blob
    for center, *_ in specs:
        assert _nearest(result, center) < 350.0


def test_two_objects_touching_end_to_end_are_separated():
    specs = [((2600.0, 2000.0), 90.0, 2400.0, 900.0),
             ((2600.0, 4300.0), 90.0, 2400.0, 900.0)]
    result = segment_shapes_in_image(_scene(specs), PIXEL, prior=CAPSULE_PRIOR)
    assert len(result) == 2
    for center, *_ in specs:
        assert _nearest(result, center) < 400.0


def test_a_single_object_is_not_split_into_fragments():
    """The counterpart guard: overlap alone would keep adding instances."""
    result = segment_shapes_in_image(
        _scene([((3000.0, 3000.0), 30.0, 2600.0, 900.0)]), PIXEL,
        prior=CAPSULE_PRIOR)
    assert len(result) == 1
    assert result.stats["components"][0]["chosen_k"] == 1


def test_an_object_running_off_the_frame_is_found_and_flagged_clipped():
    image = _scene([((300.0, 2500.0), 75.0, 2600.0, 900.0)])
    result = segment_shapes_in_image(image, PIXEL, prior=CAPSULE_PRIOR)
    assert len(result) == 1
    found = result.instances[0]
    assert found.clipped
    assert found.visible_fraction < 0.95
    assert result.stats["n_clipped"] == 1


def test_mask_label_and_overlap_rasters_are_consistent():
    specs = [((2400.0, 3200.0), 90.0, 2600.0, 900.0),
             ((3150.0, 3200.0), 90.0, 2600.0, 900.0)]
    result = segment_shapes_in_image(_scene(specs), PIXEL, prior=CAPSULE_PRIOR)
    assert len(result) == 2
    assert np.array_equal(result.overlap_count, result.masks.sum(axis=0))
    assert np.array_equal(result.union_mask, result.overlap_count > 0)
    assert np.array_equal(result.union_mask, result.labels > 0)
    assert result.labels.max() == len(result)
    # Every labelled pixel belongs to the mask it is labelled with.
    for index in range(len(result)):
        assert result.masks[index][result.labels == index + 1].all()


def test_overlapping_models_are_representable_as_masks_but_not_as_labels():
    """Why both rasters exist: a 2-D projection cannot assign a true overlap."""
    shape = (320, 340)
    left = instance_mask(_instance((2400.0, 3200.0), 90.0, 2600.0, 900.0),
                         shape, PIXEL)
    right = instance_mask(_instance((2900.0, 3200.0), 90.0, 2600.0, 900.0),
                          shape, PIXEL)
    shared = left & right
    assert shared.any()                       # the models genuinely overlap
    assert np.stack([left, right]).sum(axis=0).max() == 2


def test_empty_and_blank_fields_return_an_empty_result():
    for values in (np.zeros((0, 0)), np.zeros((40, 50))):
        result = segment_shapes(ScalarField(values, PIXEL))
        assert result.instances == []
        assert result.masks.shape == (0, *values.shape)
        assert not result.union_mask.any()
        assert result.stats["n_instances"] == 0


def test_detection_grid_size_is_guarded():
    cfg = ShapeSegmentationConfig(detection_pixel_nm=1.0, max_detection_pixels=1000)
    with pytest.raises(ValueError, match="detection grid"):
        segment_shapes_in_image(_scene([((2000.0, 2000.0), 0.0, 2000.0, 900.0)]),
                                PIXEL, prior=CAPSULE_PRIOR, cfg=cfg)


def test_image_and_point_cloud_entries_agree():
    """The two front doors must describe the same object."""
    rng = np.random.default_rng(3)
    centre, angle, length, width = (3000.0, 3000.0), 35.0, 2600.0, 900.0
    shape = (320, 340)
    mask = instance_mask(_instance(centre, angle, length, width), shape, PIXEL)
    rows, cols = np.nonzero(mask)
    pick = rng.choice(rows.size, 9000, replace=True)
    x = (cols[pick] + rng.random(pick.size)) * PIXEL
    y = (rows[pick] + rng.random(pick.size)) * PIXEL

    from_points = segment_shapes_in_points(x, y, prior=CAPSULE_PRIOR)
    from_image = segment_shapes_in_image(_scene([(centre, angle, length, width)]),
                                         PIXEL, prior=CAPSULE_PRIOR)
    assert len(from_points) == 1 and len(from_image) == 1
    a, b = from_points.instances[0], from_image.instances[0]
    assert np.hypot(a.center_nm[0] - b.center_nm[0],
                    a.center_nm[1] - b.center_nm[1]) < 300.0
    assert a.size_nm[1] == pytest.approx(b.size_nm[1], abs=250.0)


def test_point_labels_assign_localizations_to_objects():
    rng = np.random.default_rng(5)
    specs = [((2400.0, 3200.0), 90.0, 2600.0, 900.0),
             ((4200.0, 3200.0), 90.0, 2600.0, 900.0)]
    xs, ys, truth = [], [], []
    for index, spec in enumerate(specs, 1):
        mask = instance_mask(_instance(*spec), (340, 340), PIXEL)
        rows, cols = np.nonzero(mask)
        pick = rng.choice(rows.size, 6000, replace=True)
        xs.append((cols[pick] + rng.random(pick.size)) * PIXEL)
        ys.append((rows[pick] + rng.random(pick.size)) * PIXEL)
        truth.append(np.full(pick.size, index))
    x, y = np.concatenate(xs), np.concatenate(ys)
    truth = np.concatenate(truth)

    result = segment_shapes_in_points(x, y, prior=CAPSULE_PRIOR)
    assert len(result) == 2
    assert result.point_labels is not None
    assert result.point_labels.shape == x.shape
    assigned = result.point_labels > 0
    assert assigned.mean() > 0.9
    # Labels are arbitrary ids, so score the induced partition, not the numbers.
    agree = max(
        float((result.point_labels[assigned] == truth[assigned]).mean()),
        float((result.point_labels[assigned] == 3 - truth[assigned]).mean()))
    assert agree > 0.9


def test_explicit_field_bounds_make_clipping_detectable():
    """A point cloud spans only itself, so the real frame must be supplied."""
    rng = np.random.default_rng(11)
    shape = (320, 340)
    mask = instance_mask(_instance((1500.0, 3000.0), 90.0, 2800.0, 900.0),
                         shape, PIXEL)
    rows, cols = np.nonzero(mask)
    x = (cols + rng.random(rows.size)) * PIXEL
    y = (rows + rng.random(rows.size)) * PIXEL
    keep = y < 3000.0                       # the acquisition cut the cell in half
    x, y = x[keep], y[keep]

    # Without a frame the raster is spanned by the data, so nothing can leave it.
    loose = segment_shapes_in_points(x, y, prior=CAPSULE_PRIOR)
    assert len(loose) == 1
    assert not any(item.clipped for item in loose.instances)
    assert loose.instances[0].visible_fraction == pytest.approx(1.0)

    framed = segment_shapes_in_points(x, y, prior=CAPSULE_PRIOR,
                                      bounds_nm=(0.0, 1000.0, 3400.0, 3000.0))
    assert len(framed) == 1
    assert framed.instances[0].clipped
    assert framed.instances[0].visible_fraction < 1.0


def test_localizations_outside_the_frame_are_still_rendered():
    """The scanned frame decides clipping; it must not discard stray points."""
    x = np.array([100.0, 500.0, 900.0, 1300.0])
    y = np.array([100.0, 500.0, 900.0, 1300.0])
    field = field_from_points(x, y, PIXEL, bounds_nm=(400.0, 400.0, 1000.0, 1000.0))
    assert field.values.sum() == x.size          # every point kept
    assert field.visible_bounds() == (400.0, 400.0, 1000.0, 1000.0)
    x0, y0, x1, y1 = field.bounds_nm()           # raster covers frame and data
    assert x0 <= 100.0 and y0 <= 100.0 and x1 >= 1300.0 and y1 >= 1300.0


def test_field_bounds_are_validated():
    with pytest.raises(ValueError, match="x1 > x0"):
        field_from_points(np.array([0.0, 1.0]), np.array([0.0, 1.0]), PIXEL,
                          bounds_nm=(10.0, 0.0, 5.0, 100.0))


def test_instance_outline_and_mask_agree():
    item = _instance((3000.0, 2500.0), 40.0, 2600.0, 900.0)
    outline = instance_outline(item)
    model = item.model
    on_edge = model.sdf(outline[:, 0], outline[:, 1], item.center_nm[0],
                        item.center_nm[1], item.angle_deg, item.size_nm)
    assert np.abs(on_edge).max() < 1.0        # outline lies on the zero level set
    assert instance_mask(item, (300, 320), PIXEL).any()


def test_otsu_threshold_separates_two_modes():
    values = np.concatenate([np.zeros(500), np.full(500, 10.0)])
    assert 0.0 < otsu_threshold(values) < 10.0
    assert otsu_threshold(np.array([])) == 0.0
    assert otsu_threshold(np.full(10, 3.0)) == 3.0


def test_finding_nothing_explains_which_stage_dropped_everything():
    """A bare zero is not actionable; the caller needs the reason.

    This is the shape of the real failure: an over-fine pixel/smoothing pair
    (say, restored from a session on a much smaller shape) breaks the field into
    fragments that never reach one object's worth of area.
    """
    image = _scene([((3000.0, 3000.0), 30.0, 2600.0, 900.0)])
    cfg = ShapeSegmentationConfig(detection_pixel_nm=PIXEL,
                                  min_component_area_frac=50.0)
    result = segment_shapes_in_image(image, PIXEL, prior=CAPSULE_PRIOR, cfg=cfg)
    assert len(result) == 0
    reason = result.stats.get("reason")
    assert reason and "smaller than" in reason
    assert "pixel size" in reason or "smoothing" in reason


def test_a_blank_field_says_so_rather_than_blaming_the_object_size():
    cfg = ShapeSegmentationConfig(detection_pixel_nm=PIXEL, threshold=1e9)
    result = segment_shapes_in_image(
        _scene([((3000.0, 3000.0), 0.0, 2600.0, 900.0)]), PIXEL,
        prior=CAPSULE_PRIOR, cfg=cfg)
    assert len(result) == 0
    assert "foreground threshold" in result.stats.get("reason", "")


def test_a_successful_run_carries_no_failure_reason():
    result = segment_shapes_in_image(
        _scene([((3000.0, 3000.0), 30.0, 2600.0, 900.0)]), PIXEL,
        prior=CAPSULE_PRIOR)
    assert len(result) == 1
    assert "reason" not in result.stats
