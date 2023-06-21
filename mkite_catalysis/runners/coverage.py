import random
import itertools
import numpy as np
from typing import List
from pymatgen.core import Structure, Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


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

    def get_finder(self):
        return AdsorbateSiteFinder(
            self.surface.copy(),
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
        frac_coords = np.stack(
            [lattice.get_fractional_coordinates(site) for site in sites]
        )
        return lattice.get_all_distances(frac_coords, frac_coords)

    def get_combinations(self, num_adsorbates: int, dists: np.ndarray):
        indices = list(range(dists))
        combinations = []
        for comb in itertools.combinations(indices, r=num_adsorbates):
            pairs = list(itertools.combinations(comb, 2))
            pair_dists = dists[pairs]

            if min(pair_dists) > self.dist_thresh:
                combinations.append(comb)

        return combinations

    def generate_random_configs(
        self, num_adsorbates: int, num_configs: int = 50
    ) -> List[Structure]:
        sites = self.get_sites()
        dists = self.get_distances(sites)

        site_combinations = self.get_combinations(num_adsorbates, dists)
        if len(site_combinations) > num_configs:
            site_combinations = random.sample(site_combinations, num_configs)

        structures = []
        for comb in site_combinations:
            finder = self.get_finder()
            for i in comb:
                coords = sites[i]
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
