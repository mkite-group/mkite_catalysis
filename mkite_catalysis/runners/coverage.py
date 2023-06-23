import itertools
import random
from typing import List

import numpy as np
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Molecule
from pymatgen.core import Structure


class CoverageGenerator:
    """Creates surfaces covered by a given adsorbate"""

    def __init__(
        self,
        surface: Structure,
        adsorbate: Molecule,
        surface_height: float = 0.9,
        adsorption_height: float = 2.0,
        distance_threshold: float = 2.0,
    ):
        self.surface = surface
        self.adsorbate = adsorbate
        self.surf_height = surface_height
        self.ads_height = adsorption_height
        self.dist_thresh = distance_threshold

    def get_finder(self, surface=None):
        if surface is None:
            return AdsorbateSiteFinder(
                self.surface.copy(),
                selective_dynamics=True,
                height=self.surf_height,
            )

        return AdsorbateSiteFinder(
            surface,
            selective_dynamics=True,
            height=self.surf_height,
        )

    def get_sites(self) -> np.ndarray:
        finder = self.get_finder()
        sites = finder.find_adsorption_sites(
            distance=self.ads_height,
            symm_reduce=-1,
        )
        return np.stack(sites["all"])

    def get_distances(self, sites: np.ndarray) -> np.ndarray:
        lattice = self.surface.lattice
        frac_coords = np.stack([lattice.get_fractional_coords(site) for site in sites])
        return lattice.get_all_distances(frac_coords, frac_coords)

    def get_combinations(
        self, num_adsorbates: int, num_configs: int, dists: np.ndarray
    ):
        indices = list(range(len(dists)))

        if num_adsorbates == 1:
            return [[i] for i in indices]

        # as we have an early stopping, make the combinations random
        # instead of relying on sorted indices
        random.shuffle(indices)

        combinations = []
        for comb in itertools.combinations(indices, r=num_adsorbates):
            comb = list(comb)
            # select distance submatrix
            d = dists[:, comb][comb]
            i = np.triu_indices_from(d, k=1)

            if min(d[i]) > self.dist_thresh:
                combinations.append(comb)

            if len(combinations) >= num_configs:
                return combinations

        return combinations

    def generate_random_configs(
        self, num_adsorbates: int, num_configs: int = 50
    ) -> List[Structure]:
        sites = self.get_sites()
        dists = self.get_distances(sites)

        combinations = self.get_combinations(num_adsorbates, num_configs, dists)

        structures = []
        for comb in combinations:
            adsorbed = self.surface.copy()
            for i in comb:
                coords = sites[i]
                finder = self.get_finder(adsorbed)
                adsorbed = finder.add_adsorbate(self.adsorbate, coords)

            adsorbed = self.add_adsorbate_tags(adsorbed)
            structures.append(adsorbed)

        return structures

    def add_adsorbate_tags(self, struct: Structure):
        adsorbed_idx = [
            i
            for i, p in enumerate(struct.site_properties["surface_properties"])
            if p == "adsorbate"
        ]

        for i in adsorbed_idx:
            struct[i].properties["location"] = "adsorbate"

        return struct
