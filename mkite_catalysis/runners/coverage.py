import itertools
import math
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
        max_enumeration: int = int(1e6),
    ):
        self.surface = surface
        self.adsorbate = adsorbate
        self.surf_height = surface_height
        self.ads_height = adsorption_height
        self.dist_thresh = distance_threshold
        self.max_enum = max_enumeration

    def get_finder(self, surface=None):
        if surface is None:
            return AdsorbateSiteFinder(
                self.surface.copy(),
                selective_dynamics=False,
                height=self.surf_height,
            )

        return AdsorbateSiteFinder(
            surface,
            selective_dynamics=False,
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
        num_enums = math.comb(len(dists), num_adsorbates)

        if num_enums < num_configs or num_enums < self.max_enum:
            return self.get_small_combinations(num_adsorbates, num_configs, dists)

        return self.get_large_combinations(num_adsorbates, num_configs, dists)

    def get_small_combinations(
        self, num_adsorbates: int, num_configs: int, dists: np.ndarray
    ):
        indices = list(range(len(dists)))
        if num_adsorbates == 1:
            return [[i] for i in indices]

        combinations = []
        for comb in itertools.combinations(indices, r=num_adsorbates):
            combarr = np.array(comb)
            d = dists[combarr.reshape(-1, 1), combarr.reshape(1, -1)]
            i = np.triu_indices_from(d, k=1)

            if min(d[i]) > self.dist_thresh:
                combinations.append(comb)

        if len(combinations) <= num_configs:
            return combinations

        return random.sample(combinations, num_configs)

    def get_large_combinations(
        self, num_adsorbates: int, num_configs: int, dists: np.ndarray
    ):
        indices = list(range(len(dists)))

        attempts = 0
        combinations = []
        while len(combinations) < num_configs and attempts < self.max_enum:
            # when there are too many combinations, it is
            # better to simply sample the indices
            comb = random.sample(indices, num_adsorbates)

            # select distance submatrix
            d = dists[:, comb][comb]
            i = np.triu_indices_from(d, k=1)

            if min(d[i]) > self.dist_thresh:
                combinations.append(comb)

            attempts += 1

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
            struct[i].properties["selective_dynamics"] = [True, True, True]

        return struct
