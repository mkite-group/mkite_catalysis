import os
import time
from typing import Dict
from typing import List

from mkite_catalysis.recipes import CatalysisRecipe
from mkite_catalysis.runners.coverage import CoverageGenerator
from mkite_catalysis.runners.deduplicate import Deduplicator
from mkite_core.models import ConformerInfo
from mkite_core.models import CrystalInfo
from mkite_core.models import JobResults
from mkite_core.models import NodeResults
from mkite_core.recipes import BaseOptions
from mkite_core.recipes import RecipeError
from pydantic import Field
from pymatgen.core import Molecule
from pymatgen.core import Structure


class CoverageOptions(BaseOptions):
    num_adsorbates: List[int] = Field(
        [2],
        description="Number of adsorbates to add to the surface",
    )
    num_configs: int = Field(
        100,
        description="Maximum number of configurations to generate",
    )
    surface_height: float = Field(
        0.9,
        description="Height at which atoms are considered to be \
            part of the surface",
    )
    adsorption_height: float = Field(
        2.0, description="Height at which the adsorbate will be placed"
    )
    distance_threshold: float = Field(
        2.0,
        description="Maximum distance between the relevant adsorption sites \
            and the adsorption site",
    )
    deduplicate_k: int = Field(
        30,
        description="Number of neighbors to consider when \
            deduplicating structures",
    )
    deduplicate_tol: float = Field(
        1e-3,
        description="Maximum distance to consider two structures as identical",
    )
    max_enumeration: int = Field(
        int(1e6),
        description="Maximum number of attempts to enumerate when \
            generating the configurations",
    )


class CoverageRecipe(CatalysisRecipe):
    OPTIONS_CLS = CoverageOptions

    def run(self):
        start_time = time.process_time()

        surface, adsorbate = self.get_inputs()

        structures = self.generate(surface, adsorbate)

        end_time = time.process_time()
        duration = round(end_time - start_time, 6)

        return self.postprocess(structures, duration=duration)

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
                surface = surface.as_pymatgen()

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
    ) -> Dict[int, List[Structure]]:
        opts = self.get_options()
        generator = CoverageGenerator(
            surface,
            adsorbate,
            surface_height=opts["surface_height"],
            adsorption_height=opts["adsorption_height"],
            distance_threshold=opts["distance_threshold"],
            max_enumeration=opts["max_enumeration"],
        )

        results = {}
        for num in opts["num_adsorbates"]:
            structures = generator.generate_random_configs(
                num_adsorbates=num,
                num_configs=opts["num_configs"],
            )
            structures = self.deduplicate(structures)
            results[num] = structures

        return results

    def deduplicate(self, structures: List[Structure]) -> List[Structure]:
        opts = self.get_options()
        dedup = Deduplicator(
            k=opts["deduplicate_k"],
            tol=opts["deduplicate_tol"],
        )
        return dedup.deduplicate(structures)

    def postprocess(
        self, structures: Dict[int, List[Structure]], duration: float, **kwargs
    ) -> JobResults:
        nodes = []
        for num_adsorbates, struct_list in structures.items():
            for struct in struct_list:
                info = CrystalInfo.from_pymatgen(struct)
                info.attributes = {
                    **self.get_input_attributes("Crystal"),
                    **kwargs,
                    "num_adsorbates": num_adsorbates,
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
