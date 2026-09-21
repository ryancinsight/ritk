"""Value-semantic concurrency contract for the free-threaded PyO3 build."""

from concurrent.futures import ThreadPoolExecutor
import sys

import numpy as np
import pytest

import ritk


@pytest.mark.skipif(
    getattr(sys, "_is_gil_enabled", lambda: True)(),
    reason="requires a free-threaded CPython interpreter",
)
def test_image_value_semantics_across_threads():
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    image = ritk.Image(values, spacing=(1.5, 2.0, 2.5), origin=(4.0, 5.0, 6.0))

    def read_image():
        return image.shape, image.spacing, image.origin, image.to_numpy()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: read_image(), range(8)))

    for shape, spacing, origin, result in results:
        assert shape == (2, 3, 4)
        assert spacing == (1.5, 2.0, 2.5)
        assert origin == (4.0, 5.0, 6.0)
        np.testing.assert_array_equal(result, values)
