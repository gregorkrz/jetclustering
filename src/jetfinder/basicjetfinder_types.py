import numpy as np

from math import atan2, pi, log, sqrt


# A few saftey factor constants
_MaxRap = 1e5

# _invalid_phi = -100.0
# _invalid_rap = -1.0e200


class PseudoJet:
    def __init__(self, px, py, pz, E):
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E

        self.pt2 = px * px + py * py
        self.inv_pt2 = 1.0 / self.pt2

        self.rap = self._set_rap()
        self.phi = self._set_phi()

        self.cluster_history_index = -1

    def _set_rap(self):
        if (self.E == abs(self.pz)) and (self.pt2 == 0.0):
            # Point has infinite rapidity -- convert that into a very large
            #    number, but in such a way that different 0-pt momenta will have
            #    different rapidities (so as to lift the degeneracy between
            #                         them) [this can be relevant at parton-level]
            MaxRapHere = _MaxRap + abs(self.pz)
            return MaxRapHere if self.pz >= 0.0 else -MaxRapHere
        effective_m2 = max(0.0, self.m2)  # force non tachyonic mass
        E_plus_pz = self.E + abs(self.pz)  # the safer of p+, p-
        rapidity = 0.5 * log((self.pt2 + effective_m2) / (E_plus_pz * E_plus_pz))
        return rapidity if self.pz < 0 else -rapidity

    def _set_phi(self):
        if self.pt2 == 0.0:
            phi = 0.0
        else:
            phi = atan2(self.py, self.px)
        if phi < 0.0:
            phi += 2.0 * pi
        elif phi > 2.0 * pi:
            phi -= 2.0 * pi
        return phi

    def __str__(self):
        return (
            f"PseudoJet (px: {self.px}, py: {self.py}, pz: {self.pz}, E: {self.E})"
        )
    def __repr__(self):
        return self.__str__()

    @property
    def pt(self):
        """transverse momentum"""
        return sqrt(self.pt2)

    @property
    def m2(self):
        """squared invariant mass"""
        return (self.E + self.pz) * (self.E - self.pz) - self.pt2

    # Need to define the + operator on two jets
    def __add__(self, jetB):
        px = self.px + jetB.px
        py = self.py + jetB.py
        pz = self.pz + jetB.pz
        E = self.E + jetB.E
        return PseudoJet(px, py, pz, E)
'''Jet merging history as numpy arrays'''


class NPHistory:
    def __init__(self, size: int):
        '''Initialise a history struture of arrays'''
        self.size = size

        # Counter for the next slot to fill, which is equivalent to the active 'size'
        # of the history
        self.next = 0

        # Index in history where first parent of this jet was created (-1 if this jet is an
        # original particle)
        self.parent1 = np.empty(size, dtype=int)
        self.parent1.fill(-1)

        # Index in history where second parent of this jet was created (-1 if this jet is an
        # original particle); BeamJet if this history entry just labels the fact that the jet has recombined
        # with the beam)
        self.parent2 = np.empty(size, dtype=int)
        self.parent2.fill(-1)

        # Index in history where the current jet is recombined with another jet to form its child. It
        # is -1 if this jet does not further recombine
        self.child = np.empty(size, dtype=int)
        self.child.fill(-1)

        # Index in the _jets vector where we will find the Jet object corresponding to this jet
        # (i.e. the jet created at this entry of the history). NB: if this element of the history
        # corresponds to a beam recombination, then jetp_index=Invalid
        self.jetp_index = np.empty(size, dtype=int)
        self.jetp_index.fill(-1)

        # The distance corresponding to the recombination at this stage of the clustering.
        self.dij = np.zeros(size, dtype=float)

        # The largest recombination distance seen so far in the clustering history.
        self.max_dij_so_far = np.zeros(size, dtype=float)

    def append(self, parent1: int, parent2: int, jetp_index: int, dij: float, max_dij_so_far: float):
        '''Append a new item to the history'''
        if self.next == self.size:
            raise RuntimeError("History structure is now full, cannot append")

        self.parent1[self.next] = parent1
        self.parent2[self.next] = parent2
        self.jetp_index[self.next] = jetp_index
        self.dij[self.next] = dij
        self.max_dij_so_far[self.next] = max_dij_so_far

        self.next += 1

    def fill_initial_history(self, jets: list[PseudoJet]) -> float:
        '''Fill the initial history with source jets'''
        Qtot = 0.0
        for ijet, jet in enumerate(jets):
            self.jetp_index[ijet] = ijet
            jet.cluster_history_index = ijet
            Qtot = jet.E

        self.next = len(jets)

        return Qtot
from math import atan2, pi, log, sqrt


# A few saftey factor constants
_MaxRap = 1e5

# _invalid_phi = -100.0
# _invalid_rap = -1.0e200

'''Structure of arrays container for holding numpy arrays that correspond to pseudojets'''
import numpy as np
from numba import njit

class NPPseudoJets:
    def __init__(self, size: int):
        '''Setup blank arrays that will be filled later'''
        self.size = size
        self.phi = np.zeros(size, dtype=float)  # phi
        self.rap = np.zeros(size, dtype=float)  # rapidity
        self.inv_pt2 = np.zeros(size, dtype=float)  # 1/pt^2
        self.dist = np.zeros(size, dtype=float)  # nearest neighbour geometric distance
        self.akt_dist = np.zeros(size, dtype=float)  # nearest neighbour antikt metric
        self.nn = np.zeros(size, dtype=int)  # index of my nearest neighbour
        self.mask = np.ones(size, dtype=bool)  # if True this is not an active jet anymore
        self.jets_index = np.zeros(size, dtype=int)  # index reference to the PseudoJet list

    def set_jets(self, jets: list[PseudoJet]):
        if len(jets) > self.phi.size:
            raise RuntimeError(f"Attempted to fill NP PseudoJets, but containers are too small ({self.size})")
        for ijet, jet in enumerate(jets):
            self.phi[ijet] = jet.phi
            self.rap[ijet] = jet.rap
            self.inv_pt2[ijet] = jet.inv_pt2
            self.nn[ijet] = -1
            self.dist[ijet] = self.akt_dist[ijet] = 1e20
            self.mask[ijet] = False
            self.jets_index[ijet] = ijet
        self.next_slot = len(jets)
        self.dist[len(jets):] = self.akt_dist[len(jets):] = 1e20

    def __str__(self) -> str:
        _string = ""
        for ijet in range(self.phi.size):
            _string += (f"{ijet} - {self.phi[ijet]} {self.rap[ijet]} {self.inv_pt2[ijet]} {self.dist[ijet]} "
                        f"{self.akt_dist[ijet]} {self.nn[ijet]} {self.jets_index[ijet]} "
                        f"(mask: {self.mask[ijet]})\n")
        return _string

    def mask_slot(self, ijet: int):
        self.mask[ijet] = True
        self.dist[ijet] = self.akt_dist[ijet] = 1e20

    def insert_jet(self, jet: PseudoJet, slot: int, jet_index: int):
        '''Add a new pseudojet into the numpy structures'''
        if slot >= self.size:
            raise RuntimeError(
                f"Attempted to fill a jet into a slot that doesn't exist (slot {slot} >= size {self.size})")
        self.phi[slot] = jet.phi
        self.rap[slot] = jet.rap
        self.inv_pt2[slot] = jet.inv_pt2
        self.nn[slot] = -1
        self.dist[slot] = self.akt_dist[slot] = 1e20  # Necessary?
        self.jets_index[slot] = jet_index
        self.mask[slot] = False

    def print_jet(self, ijet: int) -> str:
        return (f"{ijet} - {self.phi[ijet]} {self.rap[ijet]} {self.inv_pt2[ijet]} "
                f"{self.dist[ijet]} {self.akt_dist[ijet]} {self.nn[ijet]} {self.jets_index[ijet]} "
                f"(mask: {self.mask[ijet]} -> {self.mask[self.nn[ijet]] if self.nn[ijet] >= 0 else None})")
