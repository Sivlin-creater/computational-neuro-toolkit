import numpy as np
from neuro_toolkit.bounds_pipeline import compute_hoeffding_bound


def test_hoeffding_bound_monotonic_decrease():
    sizes = np.array([10, 50, 100, 200])
    bounds = compute_hoeffding_bound(sizes, epsilon=0.05)
    assert np.all(np.diff(bounds) <= 0)
