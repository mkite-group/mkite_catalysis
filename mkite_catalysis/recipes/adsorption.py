import os
import time
import math
import numpy as np
from typing import List, Dict
from pydantic import Field
from pymatgen.core import Structure, Molecule

from mkite_core.recipes import BaseOptions, RecipeError
from mkite_core.models import (
    CrystalInfo,
    ConformerInfo,
    NodeResults,
    JobResults,
    RunStatsInfo,
)

from mkite_catalysis.recipes import CatalysisRecipe
from mkite_catalysis.runners.supercell import SupercellCalculator
from mkite_catalysis.runners.adsorption import AdsorptionGenerator


class AdsorptionOptions(BaseOptions):
    min_lateral_size: float = Field(
        6.0, description="Minimum size of the lateral unit cell (in Angstrom)"
    )
    max_supercell_vector: int = Field(
        3, description="Maximum vector when looking for supercells"
    )
    min_supercell_angle: float = Field(
        math.pi / 6,
        description="Minimum angle between supercell lattice vectors (symmetrical w.r.t. pi)",
    )
    adsorption_height: float = Field(
        2.0, description="Height at which the adsorbate will be placed"
    )
    distance_threshold: float = Field(
        3.0, description="Maximum distance between the relevant adsorption sites and the adsorption site"
    )


class AdsorptionRecipe(CatalysisRecipe):
    OPTIONS_CLS = AdsorptionOptions

    def run(self):
        start_time = time.process_time()

        surface, adsorbate = self.get_inputs()

        surface, scale = self.make_lateral_supercell(surface)
        structures = self.adsorb(surface, adsorbate)

        end_time = time.process_time()
        duration = round(end_time - start_time, 6)

        return self.postprocess(structures, duration=duration, scale=scale)

    def get_inputs(self):
        surface, adsorbate = None, None

        for inp in self.info.inputs:
            if "@class" not in inp:
                raise RecipeError(
                    "Cannot detect input type. Please specify \
                    a `@class` tag to all serialized inputs."
                )

            if inp["@class"] == "Conformer":
                adsorbate = ConformerInfo.from_dict(inp)
                adsorbate = adsorbate.as_pymatgen()

            elif inp["@class"] == "Crystal":
                surface = CrystalInfo.from_dict(inp)
                surface = self.get_pymatgen_surface(surface)

        assert surface is not None, "No Crystal provided as input"
        assert adsorbate is not None, "No Conformer provided as input"

        return surface, adsorbate

    def get_pymatgen_surface(self, info: CrystalInfo):
        surface = info.as_pymatgen()
        if "dopant" not in info.attributes:
            return surface

        dopant = info.attributes["dopant"]
        dopant_idx = surface.indices_from_symbol(dopant)

        for i, site in enumerate(surface):
            site_type = "substitute" if i in dopant_idx else "host"
            site.properties["site_type"] = site_type

        return surface

    def get_input_attributes(self, cls_name: str) -> dict:
        for inp in self.info.inputs:
            if cls_name == inp.get("@class", None):
                return inp.get("attributes", {})

        return {}

    def make_lateral_supercell(self, struct: Structure):
        struct = struct.copy()
        opts = self.get_options()
        scell = SupercellCalculator.from_pymatgen(
            struct.lattice,
            dist_threshold=opts["min_lateral_size"],
            max_vector=opts["max_supercell_vector"],
            min_angle=opts["min_supercell_angle"],
        )

        matrix = scell.get_best_transformation()

        struct.make_supercell(matrix)

        scale = float(np.linalg.det(matrix))

        return struct, scale

    def adsorb(
        self, surface: Structure, adsorbate: Molecule
    ) -> Dict[str, List[Structure]]:
        opts = self.get_options()
        generator = AdsorptionGenerator(
            surface,
            adsorbate,
            adsorption_height=opts["adsorption_height"],
        )

        return generator.adsorb()

    def postprocess(
        self, structures: Dict[str, List[Structure]], duration: float, **kwargs
    ) -> JobResults:
        nodes = []
        for ads_site, struct_list in structures.items():
            for struct in struct_list:
                info = CrystalInfo.from_pymatgen(struct)
                info.attributes = {
                    **self.get_input_attributes("Crystal"),
                    "adsorption_site": ads_site,
                    **kwargs,
                }

                nr = NodeResults(chemnode=info.as_dict(), calcnodes=[])
                nodes.append(nr)

        runstats = self.get_run_stats(duration)

        jobres = JobResults(
            job=self.get_done_job(),
            runstats=runstats,
            nodes=nodes,
        )

        jobres.to_json(os.path.join(".", JobResults.file_name()))
        return jobres
