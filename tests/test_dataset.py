"""Tests for dataset utilities: EventPFCands, spherical_to_cartesian, renumber_clusters, etc."""

import math
import numpy as np
import pytest
import torch

from src.dataset.functions_data import (
    spherical_to_cartesian,
    renumber_clusters,
    TensorCollection,
    to_tensor,
    EventPFCands,
    EventBatch,
)


# ---------------------------------------------------------------------------
# spherical_to_cartesian
# ---------------------------------------------------------------------------

class TestSphericalToCartesian:
    def test_along_z_axis(self):
        phi = torch.tensor([0.0])
        theta = torch.tensor([0.0])  # theta=0 → z-axis
        r = torch.tensor([5.0])
        result = spherical_to_cartesian(phi, theta, r)
        assert result.shape == (1, 3)
        assert result[0, 2] == pytest.approx(5.0, abs=1e-5)
        assert abs(result[0, 0]) < 1e-5
        assert abs(result[0, 1]) < 1e-5

    def test_along_x_axis(self):
        phi = torch.tensor([0.0])
        theta = torch.tensor([math.pi / 2])
        r = torch.tensor([3.0])
        result = spherical_to_cartesian(phi, theta, r)
        assert result[0, 0] == pytest.approx(3.0, abs=1e-5)
        assert abs(result[0, 1]) < 1e-5
        assert abs(result[0, 2]) < 1e-5

    def test_normalized(self):
        phi = torch.tensor([0.5])
        theta = torch.tensor([1.0])
        r = torch.tensor([100.0])
        result = spherical_to_cartesian(phi, theta, r, normalized=True)
        norm = torch.norm(result, dim=1)
        assert norm[0] == pytest.approx(1.0, abs=1e-5)

    def test_batch(self):
        phi = torch.rand(10)
        theta = torch.rand(10)
        r = torch.rand(10)
        result = spherical_to_cartesian(phi, theta, r)
        assert result.shape == (10, 3)


# ---------------------------------------------------------------------------
# renumber_clusters
# ---------------------------------------------------------------------------

class TestRenumberClusters:
    def test_basic(self):
        tensor = torch.tensor([0, 0, 3, 3, 5, 5])
        result = renumber_clusters(tensor)
        unique_vals = result.unique().tolist()
        assert unique_vals == [0, 1, 2]

    def test_already_sequential(self):
        tensor = torch.tensor([0, 0, 1, 1, 2, 2])
        result = renumber_clusters(tensor)
        torch.testing.assert_close(result.long(), tensor)

    def test_single_cluster(self):
        tensor = torch.tensor([5, 5, 5])
        result = renumber_clusters(tensor)
        assert (result == 0).all()


# ---------------------------------------------------------------------------
# TensorCollection
# ---------------------------------------------------------------------------

class TestTensorCollection:
    def test_init(self):
        tc = TensorCollection(a=torch.tensor([1, 2]), b=torch.tensor([3, 4]))
        assert hasattr(tc, "a")
        assert hasattr(tc, "b")

    def test_to_device(self):
        tc = TensorCollection(x=torch.tensor([1.0, 2.0]))
        tc.to("cpu")
        assert tc.x.device.type == "cpu"

    def test_dict_rep(self):
        tc = TensorCollection(a=torch.tensor([1]), b="not_a_tensor")
        d = tc.dict_rep()
        assert "a" in d
        assert "b" not in d


# ---------------------------------------------------------------------------
# to_tensor
# ---------------------------------------------------------------------------

class TestToTensor:
    def test_from_list(self):
        result = to_tensor([1.0, 2.0, 3.0])
        assert torch.is_tensor(result)
        assert result.dtype == torch.float32

    def test_from_float32_tensor(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        result = to_tensor(t)
        assert result.dtype == torch.float32

    def test_from_float64_to_float32(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = to_tensor(t)
        assert result.dtype == torch.float32

    def test_from_int_tensor(self):
        t = torch.tensor([1, 2, 3])
        result = to_tensor(t)
        assert result.dtype == torch.int64

    def test_from_numpy(self):
        result = to_tensor(np.array([1.0, 2.0]))
        assert torch.is_tensor(result)


# ---------------------------------------------------------------------------
# EventPFCands
# ---------------------------------------------------------------------------

class TestEventPFCands:
    def _make_simple_pfcands(self):
        return EventPFCands(
            pt=[10.0, 20.0, 30.0],
            eta=[0.0, 1.0, -1.0],
            phi=[0.0, 0.5, -0.5],
            mass=[0.14, 0.14, 0.14],
            charge=[1, -1, 1],
            pid=torch.zeros(3),
            pf_cand_jet_idx=[-1, -1, -1],
        )

    def test_length(self):
        pfc = self._make_simple_pfcands()
        assert len(pfc) == 3

    def test_momentum_components(self):
        pfc = self._make_simple_pfcands()
        assert pfc.pxyz.shape == (3, 3)
        assert pfc.E.shape == (3,)
        assert (pfc.E > 0).all()

    def test_pxyz_norm_close_to_p(self):
        pfc = self._make_simple_pfcands()
        p_from_xyz = torch.norm(pfc.pxyz, dim=1)
        torch.testing.assert_close(p_from_xyz, pfc.p, atol=0.05, rtol=0.01)

    def test_theta_computation(self):
        pfc = self._make_simple_pfcands()
        assert pfc.theta.shape == (3,)
        assert (pfc.theta > 0).all()
        assert (pfc.theta < math.pi).all()


# ---------------------------------------------------------------------------
# EventBatch FP16 conversion
# ---------------------------------------------------------------------------

class TestEventBatchHalfPrecision:
    def _make_batch(self):
        return EventBatch(
            input_vectors=torch.randn(10, 4),
            input_scalars=torch.randn(10, 3),
            batch_idx=torch.zeros(10, dtype=torch.long),
            pt=torch.rand(10),
        )

    def test_to_cpu_stays_float32(self):
        batch = self._make_batch()
        batch.to(torch.device("cpu"), half_precision=True)
        assert batch.input_vectors.dtype == torch.float32
        assert batch.input_scalars.dtype == torch.float32

    def test_to_cpu_no_half(self):
        batch = self._make_batch()
        batch.to(torch.device("cpu"), half_precision=False)
        assert batch.input_vectors.dtype == torch.float32

    def test_batch_idx_stays_long(self):
        batch = self._make_batch()
        batch.to(torch.device("cpu"))
        assert batch.batch_idx.dtype == torch.int64
