from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from afm_ebsd_fusion.crystal import crystal_normals_from_sample_normals, sample_normals_from_crystal_normals
from afm_ebsd_fusion.ipf import cubic_reduce_to_standard_sector


def test_sample_crystal_roundtrip() -> None:
    rng = np.random.default_rng(3)
    matrices = Rotation.random(25, random_state=rng).as_matrix()
    normals = rng.normal(size=(25, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    crystal = crystal_normals_from_sample_normals(matrices, normals)
    sample = sample_normals_from_crystal_normals(matrices, crystal)
    dots = np.sum(sample * normals, axis=1)
    assert np.allclose(dots, 1.0, atol=1e-6)


def test_cubic_reduction_orders_to_ipf_sector() -> None:
    dirs = np.array([[1.0, -3.0, 2.0], [-0.1, 0.9, 0.2]])
    folded = cubic_reduce_to_standard_sector(dirs)
    assert np.all(folded[:, 2] >= folded[:, 0])
    assert np.all(folded[:, 0] >= folded[:, 1])
    assert np.allclose(np.linalg.norm(folded, axis=1), 1.0, atol=1e-6)

