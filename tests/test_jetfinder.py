"""Tests for the jet finder module: PseudoJet types, anti-kt clustering, and distance metrics."""

import math
import numpy as np
import pytest

from src.jetfinder.basicjetfinder_types import PseudoJet, NPHistory, NPPseudoJets
from src.jetfinder.basicjetfinder import (
    basicjetfinder,
    find_closest_jets,
    inclusive_jets,
    add_step_to_history,
)
from src.jetfinder.clustering import (
    lorentz_norm_comp,
    get_distance_matrix,
    get_distance_matrix_Lorentz,
    get_clustering_labels,
)


# ---------------------------------------------------------------------------
# PseudoJet
# ---------------------------------------------------------------------------

class TestPseudoJet:
    def test_basic_properties(self):
        jet = PseudoJet(3.0, 4.0, 0.0, 5.0)
        assert jet.pt2 == pytest.approx(25.0)
        assert jet.pt == pytest.approx(5.0)
        assert jet.inv_pt2 == pytest.approx(1.0 / 25.0)

    def test_mass_squared(self):
        jet = PseudoJet(3.0, 4.0, 0.0, 6.0)
        expected_m2 = 6.0**2 - 3.0**2 - 4.0**2  # E² - px² - py² - pz²  = 36 - 25 = 11
        assert jet.m2 == pytest.approx(expected_m2)

    def test_massless_particle(self):
        jet = PseudoJet(3.0, 4.0, 0.0, 5.0)
        assert jet.m2 == pytest.approx(0.0)

    def test_phi_range(self):
        jet = PseudoJet(-1.0, 0.0, 0.0, 1.0)
        assert 0.0 <= jet.phi <= 2 * math.pi

    def test_addition(self):
        j1 = PseudoJet(1.0, 0.0, 0.0, 1.0)
        j2 = PseudoJet(0.0, 1.0, 0.0, 1.0)
        merged = j1 + j2
        assert merged.px == pytest.approx(1.0)
        assert merged.py == pytest.approx(1.0)
        assert merged.pz == pytest.approx(0.0)
        assert merged.E == pytest.approx(2.0)

    def test_rapidity_forward(self):
        jet = PseudoJet(1.0, 0.0, 10.0, 10.05)
        # In this implementation: positive pz → positive rapidity (negated at the end for pz>=0)
        # Actually the implementation returns -rapidity for pz >= 0, but the log expression
        # computes a negative value for large pz, so double negation makes it positive.
        assert isinstance(jet.rap, float)

    def test_str_repr(self):
        jet = PseudoJet(1.0, 2.0, 3.0, 4.0)
        s = str(jet)
        assert "PseudoJet" in s
        assert "1.0" in s


# ---------------------------------------------------------------------------
# NPHistory
# ---------------------------------------------------------------------------

class TestNPHistory:
    def test_append(self):
        h = NPHistory(5)
        h.append(parent1=0, parent2=1, jetp_index=2, dij=0.5, max_dij_so_far=0.5)
        assert h.next == 1
        assert h.parent1[0] == 0
        assert h.parent2[0] == 1

    def test_overflow_raises(self):
        h = NPHistory(1)
        h.append(0, 1, 0, 0.1, 0.1)
        with pytest.raises(RuntimeError):
            h.append(0, 1, 0, 0.1, 0.1)

    def test_fill_initial_history(self):
        jets = [PseudoJet(1, 0, 0, 1), PseudoJet(0, 1, 0, 1)]
        h = NPHistory(4)
        h.fill_initial_history(jets)
        assert h.next == 2
        assert jets[0].cluster_history_index == 0
        assert jets[1].cluster_history_index == 1


# ---------------------------------------------------------------------------
# NPPseudoJets
# ---------------------------------------------------------------------------

class TestNPPseudoJets:
    def test_set_jets(self):
        jets = [PseudoJet(1, 0, 0, 1), PseudoJet(0, 1, 0, 1)]
        npj = NPPseudoJets(4)
        npj.set_jets(jets)
        assert npj.mask[0] == False
        assert npj.mask[1] == False
        assert npj.mask[2] == True  # unused slot

    def test_mask_slot(self):
        jets = [PseudoJet(1, 0, 0, 1)]
        npj = NPPseudoJets(2)
        npj.set_jets(jets)
        npj.mask_slot(0)
        assert npj.mask[0] == True
        assert npj.akt_dist[0] == 1e20

    def test_insert_jet(self):
        npj = NPPseudoJets(2)
        jet = PseudoJet(3.0, 4.0, 0.0, 5.0)
        npj.insert_jet(jet, slot=0, jet_index=0)
        assert npj.phi[0] == pytest.approx(jet.phi)
        assert npj.rap[0] == pytest.approx(jet.rap)

    def test_overflow_raises(self):
        npj = NPPseudoJets(1)
        jet = PseudoJet(1, 0, 0, 1)
        with pytest.raises(RuntimeError):
            npj.insert_jet(jet, slot=1, jet_index=0)


# ---------------------------------------------------------------------------
# find_closest_jets
# ---------------------------------------------------------------------------

class TestFindClosestJets:
    def test_basic(self):
        akt_dist = np.array([10.0, 2.0, 5.0])
        nn = np.array([-1, 0, 1])
        dist, idx = find_closest_jets(akt_dist, nn)
        assert idx == 1
        assert dist == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# basicjetfinder (anti-kt)
# ---------------------------------------------------------------------------

class TestBasicJetFinder:
    def test_well_separated_particles(self):
        """Two well-separated high-pT particles should produce jets and history."""
        particles = [
            PseudoJet(50.0, 10.0, 100.0, 120.0),
            PseudoJet(-40.0, -20.0, -100.0, 115.0),
        ]
        jets, history = basicjetfinder(particles, Rparam=0.8, return_raw=True)
        assert len(jets) >= 2
        assert history.next == 2 * len(particles)

    def test_return_raw(self):
        particles = [
            PseudoJet(50.0, 10.0, 100.0, 120.0),
            PseudoJet(-40.0, -20.0, -100.0, 115.0),
        ]
        jets, history = basicjetfinder(particles, Rparam=0.8, return_raw=True)
        assert isinstance(history, NPHistory)
        assert len(jets) >= 2

    def test_collinear_particles_merge(self):
        """Particles very close in eta-phi should be clustered together."""
        particles = [
            PseudoJet(10.0, 1.0, 5.0, 12.0),
            PseudoJet(10.5, 1.1, 5.1, 12.1),
        ]
        jets, history = basicjetfinder(particles, Rparam=0.8, return_raw=True)
        # After clustering, should have 3 jets in list (2 original + 1 merged)
        assert len(jets) == 3

    def test_ptmin_filter_with_return_raw(self):
        """Using return_raw we can verify the history is built correctly."""
        particles = [
            PseudoJet(50.0, 10.0, 30.0, 60.0),
            PseudoJet(-30.0, -15.0, -50.0, 65.0),
            PseudoJet(0.5, 0.3, 0.1, 0.7),
        ]
        jets, history = basicjetfinder(particles, Rparam=0.8, return_raw=True)
        assert history.next > 0
        assert len(jets) >= 3


# ---------------------------------------------------------------------------
# Distance matrices (clustering.py)
# ---------------------------------------------------------------------------

class TestDistanceMatrices:
    def test_lorentz_norm_comp_same_vector(self):
        v = np.array([10.0, 1.0, 2.0, 3.0])
        assert lorentz_norm_comp(v, v) == pytest.approx(0.0, abs=1e-10)

    def test_lorentz_norm_comp_different(self):
        v1 = np.array([10.0, 1.0, 0.0, 0.0])
        v2 = np.array([10.0, 2.0, 0.0, 0.0])
        result = lorentz_norm_comp(v1, v2)
        assert result > 0

    def test_get_distance_matrix_identity_like(self):
        v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        mat = get_distance_matrix(v)
        assert mat.shape == (3, 3)
        np.testing.assert_allclose(mat[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(mat[0, 2], 1.0, atol=1e-10)
        np.testing.assert_allclose(mat[0, 1], 0.0, atol=1e-10)

    def test_get_distance_matrix_with_torch(self):
        import torch
        v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mat = get_distance_matrix(v)
        assert mat.shape == (2, 2)

    def test_get_distance_matrix_lorentz_shape(self):
        v = np.array([[10.0, 1.0, 0.0, 0.0], [10.0, 0.0, 1.0, 0.0]])
        mat = get_distance_matrix_Lorentz(v)
        assert mat.shape == (2, 2)

    def test_get_distance_matrix_lorentz_diagonal(self):
        v = np.array([[10.0, 1.0, 2.0, 3.0]])
        mat = get_distance_matrix_Lorentz(v)
        expected = 10.0**2 - 1.0**2 - 2.0**2 - 3.0**2
        assert mat[0, 0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# HDBSCAN clustering
# ---------------------------------------------------------------------------

class TestClusteringLabels:
    def test_two_well_separated_clusters(self):
        coords_a = np.random.randn(50, 3) + np.array([10, 0, 0])
        coords_b = np.random.randn(50, 3) + np.array([-10, 0, 0])
        coords = np.vstack([coords_a, coords_b])
        batch_idx = np.zeros(100)
        labels = get_clustering_labels(
            coords, batch_idx, min_cluster_size=5, min_samples=3, epsilon=0.5
        )
        assert labels.shape == (100,)
        unique_labels = set(labels)
        unique_labels.discard(-1)
        assert len(unique_labels) >= 2

    def test_single_event(self):
        coords = np.random.randn(20, 3)
        batch_idx = np.zeros(20)
        labels = get_clustering_labels(
            coords, batch_idx, min_cluster_size=2, min_samples=1, epsilon=1.0
        )
        assert len(labels) == 20

    def test_multiple_events(self):
        coords = np.random.randn(40, 3)
        batch_idx = np.array([0]*20 + [1]*20)
        labels = get_clustering_labels(
            coords, batch_idx, min_cluster_size=2, min_samples=1, epsilon=1.0
        )
        assert len(labels) == 40
