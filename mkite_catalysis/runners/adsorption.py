from typing import Dict, List
from pymatgen.core import Structure, Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


class AdsorptionGenerator:
    def __init__(
        self,
        surface: Structure,
        adsorbate: Molecule,
        adsorption_height: float = 2.0,
        distance_threshold: float = 4.0,
    ):
        self.surface = surface
        self.adsorbate = adsorbate
        self.ads_height = adsorption_height
        self.dist_thresh = distance_threshold

    def adsorb(self) -> Structure:
        finder = AdsorbateSiteFinder(
            self.surface,
            selective_dynamics=True,
        )

        sites = finder.find_adsorption_sites(
            distance=self.ads_height,
        )

        # use the location information
        if "all" in sites:
            sites.pop("all")

        # filter only by the relevant sites
        sites = self.filter_sites(sites)

        structures = {}
        for name, all_coords in sites.items():
            site_structs = []
            for coords in all_coords:
                adsorbed = finder.add_adsorbate(self.adsorbate, coords)
                adsorbed = self.add_adsorbate_tags(adsorbed)
                site_structs.append(adsorbed)

            structures[name] = site_structs

        return structures

    def filter_sites(self, sites: Dict[str, List[list]]) -> dict:
        return sites

    def add_adsorbate_tags(self, struct: Structure):
        adsorbed_idx = [
            i
            for i, p in enumerate(struct.site_properties["surface_properties"])
            if p == "adsorbate"
        ]

        for i in adsorbed_idx:
            struct[i].properties["location"] = "adsorbate"

        return struct
