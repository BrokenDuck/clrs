"""Small framework-independent validation helpers for reference algorithms."""
import numpy as np


def assert_rank(arrays, rank):
    if not isinstance(arrays, (list, tuple)):
        arrays = [arrays]
    for array in arrays:
        if np.ndim(array) != rank:
            raise AssertionError(f"Expected rank {rank}, got {np.ndim(array)}")
