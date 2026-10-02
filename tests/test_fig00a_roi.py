from demo_Sep2026.fig00a_averages import ROI_DIAMETER, ROI_X, ROI_Y, circular_roi_mask, roi_center


def test_circular_roi_is_20px_and_inside_the_given_box():
    mask = circular_roi_mask((512, 512))
    cx, cy = roi_center()
    ys, xs = mask.nonzero()
    assert xs.min() >= ROI_X[0]
    assert xs.max() < ROI_X[1]
    assert ys.min() >= ROI_Y[0]
    assert ys.max() < ROI_Y[1]
    assert abs((xs.max() - xs.min() + 1) - ROI_DIAMETER) <= 1
    assert abs((ys.max() - ys.min() + 1) - ROI_DIAMETER) <= 1
    assert abs(xs.mean() - cx) < 0.6
    assert abs(ys.mean() - cy) < 0.6
    assert 280 <= int(mask.sum()) <= 330
