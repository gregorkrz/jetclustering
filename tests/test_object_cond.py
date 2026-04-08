"""Tests for object condensation layer utilities: calc_eta_phi, huber, clustering, isin, reincrementalize."""

import math
import numpy as np
import pytest
import torch

from src.layers.object_cond import (
    calc_eta_phi,
    huber,
    safe_index,
    assert_no_nans,
    isin,
    reincrementalize,
    get_clustering_np,
    get_clustering,
    scatter_counts_to_indices,
)


# ---------------------------------------------------------------------------
# calc_eta_phi
# ---------------------------------------------------------------------------

class TestCalcEtaPhi:
    def test_output_shape_stacked(self):
        coords = torch.randn(10, 3)
        result = calc_eta_phi(coords, return_stacked=True)
        assert result.shape == (10, 2)

    def test_output_shape_unstacked(self):
        coords = torch.randn(10, 3)
        eta, phi = calc_eta_phi(coords, return_stacked=False)
        assert eta.shape == (10,)
        assert phi.shape == (10,)

    def test_phi_range(self):
        coords = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        result = calc_eta_phi(coords)
        phi = result[:, 1]
        assert phi[0] == pytest.approx(0.0, abs=1e-6)
        assert phi[1] == pytest.approx(math.pi / 2, abs=1e-6)
        assert abs(phi[2]) == pytest.approx(math.pi, abs=1e-6)
        assert phi[3] == pytest.approx(-math.pi / 2, abs=1e-6)

    def test_eta_zero_at_midplane(self):
        """Points in the xy-plane (z=0) should have eta ≈ 0."""
        coords = torch.tensor([[1.0, 1.0, 0.0], [3.0, 4.0, 0.0]])
        result = calc_eta_phi(coords)
        eta = result[:, 0]
        assert eta[0] == pytest.approx(0.0, abs=1e-6)
        assert eta[1] == pytest.approx(0.0, abs=1e-6)

    def test_eta_sign(self):
        """Positive z → positive eta, negative z → negative eta."""
        coords = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
        eta, _ = calc_eta_phi(coords, return_stacked=False)
        assert eta[0] > 0
        assert eta[1] < 0

    def test_consistency_stacked_vs_unstacked(self):
        coords = torch.randn(5, 3)
        stacked = calc_eta_phi(coords, return_stacked=True)
        eta, phi = calc_eta_phi(coords, return_stacked=False)
        torch.testing.assert_close(stacked[:, 0], eta)
        torch.testing.assert_close(stacked[:, 1], phi)


# ---------------------------------------------------------------------------
# huber
# ---------------------------------------------------------------------------

class TestHuber:
    def test_quadratic_region(self):
        d = torch.tensor([0.5])
        delta = 1.0
        result = huber(d, delta)
        assert result.item() == pytest.approx(0.25)

    def test_linear_region(self):
        d = torch.tensor([3.0])
        delta = 1.0
        result = huber(d, delta)
        expected = 2.0 * delta * (abs(3.0) - delta)
        assert result.item() == pytest.approx(expected)

    def test_at_boundary(self):
        d = torch.tensor([1.0])
        delta = 1.0
        result = huber(d, delta)
        assert result.item() == pytest.approx(1.0)

    def test_negative_values(self):
        d = torch.tensor([-2.0])
        delta = 1.0
        result = huber(d, delta)
        expected = 2.0 * delta * (2.0 - delta)
        assert result.item() == pytest.approx(expected)

    def test_zero(self):
        d = torch.tensor([0.0])
        delta = 1.0
        assert huber(d, delta).item() == pytest.approx(0.0)

    def test_batch(self):
        d = torch.tensor([0.0, 0.5, 1.0, 2.0])
        delta = 1.0
        result = huber(d, delta)
        assert result.shape == (4,)


# ---------------------------------------------------------------------------
# safe_index
# ---------------------------------------------------------------------------

class TestSafeIndex:
    def test_present(self):
        assert safe_index([10, 20, 30], 20) == 2  # 1-indexed

    def test_absent(self):
        assert safe_index([10, 20, 30], 99) == 0

    def test_first_element(self):
        assert safe_index([5, 6, 7], 5) == 1


# ---------------------------------------------------------------------------
# assert_no_nans
# ---------------------------------------------------------------------------

class TestAssertNoNans:
    def test_clean_tensor(self):
        assert_no_nans(torch.tensor([1.0, 2.0, 3.0]))

    def test_nan_tensor(self):
        with pytest.raises(AssertionError):
            assert_no_nans(torch.tensor([1.0, float("nan"), 3.0]))


# ---------------------------------------------------------------------------
# isin
# ---------------------------------------------------------------------------

class TestIsin:
    def test_basic(self):
        ar1 = torch.tensor([1, 2, 3, 4, 5])
        ar2 = torch.tensor([2, 4])
        result = isin(ar1, ar2)
        expected = torch.tensor([False, True, False, True, False])
        torch.testing.assert_close(result, expected)

    def test_empty_ar2(self):
        ar1 = torch.tensor([1, 2, 3])
        ar2 = torch.tensor([])
        result = isin(ar1, ar2)
        assert not result.any()

    def test_all_present(self):
        ar1 = torch.tensor([1, 2, 3])
        ar2 = torch.tensor([1, 2, 3])
        result = isin(ar1, ar2)
        assert result.all()


# ---------------------------------------------------------------------------
# scatter_counts_to_indices
# ---------------------------------------------------------------------------

class TestScatterCountsToIndices:
    def test_basic(self):
        counts = torch.tensor([3, 2, 2])
        result = scatter_counts_to_indices(counts)
        expected = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        torch.testing.assert_close(result, expected)

    def test_single_group(self):
        counts = torch.tensor([5])
        result = scatter_counts_to_indices(counts)
        expected = torch.tensor([0, 0, 0, 0, 0])
        torch.testing.assert_close(result, expected)

    def test_empty(self):
        counts = torch.tensor([0, 3])
        result = scatter_counts_to_indices(counts)
        expected = torch.tensor([1, 1, 1])
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# get_clustering_np
# ---------------------------------------------------------------------------

class TestGetClusteringNp:
    def test_two_clusters(self):
        betas = np.array([0.9, 0.01, 0.01, 0.8, 0.01, 0.01])
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ])
        clustering = get_clustering_np(betas, X, tbeta=0.5, td=1.0)
        assert clustering[0] == 0
        assert clustering[1] == 0
        assert clustering[2] == 0
        assert clustering[3] == 3
        assert clustering[4] == 3
        assert clustering[5] == 3

    def test_no_condpoints(self):
        betas = np.array([0.01, 0.02, 0.03])
        X = np.array([[0, 0], [1, 1], [2, 2]])
        clustering = get_clustering_np(betas, X, tbeta=0.5, td=1.0)
        assert (clustering == -1).all()

    def test_all_background(self):
        betas = np.array([0.9, 0.01])
        X = np.array([[0, 0], [100, 100]])
        clustering = get_clustering_np(betas, X, tbeta=0.5, td=0.5)
        assert clustering[0] == 0
        assert clustering[1] == -1


# ---------------------------------------------------------------------------
# get_clustering (torch version)
# ---------------------------------------------------------------------------

class TestGetClustering:
    def test_two_clusters(self):
        betas = torch.tensor([0.9, 0.01, 0.01, 0.8, 0.01, 0.01])
        X = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ])
        clustering = get_clustering(betas, X, tbeta=0.5, td=1.0)
        assert clustering[0].item() == 0
        assert clustering[1].item() == 0
        assert clustering[3].item() == 3
        assert clustering[4].item() == 3

    def test_no_condpoints(self):
        betas = torch.tensor([0.01, 0.02])
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        clustering = get_clustering(betas, X, tbeta=0.5, td=1.0)
        assert (clustering == -1).all()


# ---------------------------------------------------------------------------
# reincrementalize
# ---------------------------------------------------------------------------

class TestReincrementalize:
    def test_docstring_example(self):
        y = torch.LongTensor([0, 0, 0, 1, 1, 3, 3, 0, 0, 0, 0, 0, 2, 2, 3, 3, 0, 0, 1, 1])
        batch = torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        result = reincrementalize(y, batch)
        expected = torch.tensor([0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1])
        torch.testing.assert_close(result, expected)

    def test_no_holes(self):
        y = torch.LongTensor([0, 0, 1, 1])
        batch = torch.LongTensor([0, 0, 0, 0])
        result = reincrementalize(y, batch)
        expected = torch.tensor([0, 0, 1, 1])
        torch.testing.assert_close(result, expected)
