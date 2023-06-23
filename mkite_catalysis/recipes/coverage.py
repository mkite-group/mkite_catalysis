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
from mkite_catalysis.runners.coverage import CoverageGenerator
from mkite_catalysis.runners.deduplicate import Deduplicator


class CoverageOptions(BaseOptions):
    num_configs: int = Field(
        100,
        description="Maximum number of configurations to generate",
    )
    num_adsorbates: int = Field(
        2,
        description="Number of adsorbates to add to the surface",
    )
    surface_height: float = Field(
        0.9,
        description="Height at which atoms are considered to be part of the surface",
    )
    adsorption_height: float = Field(
        2.0, description="Height at which the adsorbate will be placed"
    )
    distance_threshold: float = Field(
        3.0,
        description="Maximum distance between the relevant adsorption sites and the adsorption site",
    )
    deduplicate_k: int = Field(
        30,
        description="Number of neighbors to consider when deduplicating structures",
    )
    deduplicate_tol: float = Field(
        1e-3,
        description="Maximum distance to consider two structures as identical",
    )


class CoverageRecipe(CatalysisRecipe):
    OPTIONS_CLS = AdsorptionOptions

    def run(self):
        start_time = time.process_time()

        surface, adsorbate = self.get_inputs()

        surface, scale = self.make_lateral_supercell(surface)
        structures = self.generate(surface, adsorbate)
        structures = self.deduplicate(structures)

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
                surface = surface.as_pymatgen(surface)

        assert surface is not None, "No Crystal provided as input"
        assert adsorbate is not None, "No Conformer provided as input"

        return surface, adsorbate

    def get_input_attributes(self, cls_name: str) -> dict:
        for inp in self.info.inputs:
            if cls_name == inp.get("@class", None):
                return inp.get("attributes", {})

        return {}

    def generate(
        self, surface: Structure, adsorbate: Molecule
    ) -> Dict[str, List[Structure]]:
        opts = self.get_options()
        generator = CoverageGenerator(
            surface,
            adsorbate,
            surface_height=opts["surface_height"],
            adsorption_height=opts["adsorption_height"],
            distance_threshold=opts["distance_threshold"],
        )

        return generator.generate_random_configs()

    def deduplicate(self, structures: List[Structure]) -> List[Structure]:
        opts = self.get_options()
        dedup = Deduplicator(
            k=opts["deduplicate_k"],
            tol=opts["deduplicate_tol"],
        )
        return dedup.deduplicate(structures)

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
