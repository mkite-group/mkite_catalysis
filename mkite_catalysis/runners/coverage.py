import itertools
import math
import random
from typing import List
from typing import Tuple

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
        sites = np.stack(sites["all"])

        return self.filter_sites(sites)

    def filter_sites(self, sites: np.ndarray) -> np.ndarray:
        """Filter sites by their distances to existing adsorbates"""
        if "surface_properties" not in self.surface.site_properties:
            return sites

        adsorbates = [
            idx
            for idx, site in enumerate(self.surface)
            if site.properties["surface_properties"] == "adsorbate"
        ]

        if len(adsorbates) == 0:
            return sites

        # computes fractional coordinates for sites and adsorbates
        lattice = self.surface.lattice
        ads_coords = self.surface.frac_coords[adsorbates]
        site_coords = lattice.get_fractional_coords(sites)

        # valid sites do not overlap with adsorbates
        dm = lattice.get_all_distances(ads_coords, site_coords)
        min_dist = dm.min(0)
        valid = min_dist > self.dist_thresh

        return sites[valid]

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
            comb = list(comb)
            if self.indices_are_valid(comb, dists):
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
            if self.indices_are_valid(comb, dists):
                combinations.append(comb)

            attempts += 1

        return combinations

    def swap_indices(
        self,
        indices: List[int],
        num_configs: int,
        dists: np.ndarray,
        num_swap: int = 1,
    ):
        num_adsorbates = len(indices)
        assert num_swap <= num_adsorbates, (
            f"cannot swap {num_swap} sites"
            + f"when there are {num_adsorbates} adsorbates!"
        )

        others = [i for i in range(len(dists)) if i not in indices]

        attempts = 0
        combinations = []
        while len(combinations) < num_configs and attempts < self.max_enum:
            to_keep = random.sample(indices, num_adsorbates - num_swap)
            to_add = random.sample(others, num_swap)
            comb = list(to_keep) + list(to_add)

            if self.indices_are_valid(comb, dists):
                combinations.append(comb)

            attempts += 1

        return combinations

    def indices_are_valid(self, indices: List[int], dists: np.ndarray):
        # select distance submatrix
        d = dists[:, indices][indices]
        i = np.triu_indices_from(d, k=1)

        return min(d[i]) > self.dist_thresh

    def generate_random_configs(
        self, num_adsorbates: int, num_configs: int = 50
    ) -> List[Structure]:
        sites = self.get_sites()
        dists = self.get_distances(sites)

        combinations = self.get_combinations(num_adsorbates, num_configs, dists)

        structures = []
        for comb in combinations:
            adsorb_locs = [sites[i] for i in comb]
            adsorbed = self.adsorb_on_sites(adsorb_locs)
            structures.append(adsorbed)

        return structures

    def adsorb_on_sites(self, sites: List[Tuple[float, float, float]]):
        adsorbed = self.surface.copy()
        for coords in sites:
            finder = self.get_finder(adsorbed)
            adsorbed = finder.add_adsorbate(self.adsorbate, coords)

        adsorbed = self.add_adsorbate_tags(adsorbed)
        return adsorbed

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
