import os
import time
import math
from typing import Tuple, List, Union
from pydantic import Field
from pymatgen.core.surface import (
    Slab,
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
)

from pymatgen.analysis.structure_matcher import StructureMatcher
from mkite_core.recipes import BaseOptions
from mkite_core.models import CrystalInfo, NodeResults, JobResults, RunStatsInfo
from mkite_catalysis.recipes import CatalysisRecipe


class SurfaceGenerationOptions(BaseOptions):
    max_miller_index: int = Field(
        2,
        description="Maximum Miller index when generating a surface",
    )
    miller_indices: Union[Tuple[int], List[Tuple[int]]] = Field(
        None,
        description="Miller indices used to generate a surface",
    )
    min_slab_size: float = Field(
        8.0,
        description="Slab thickness (in Angstrom)",
    )
    min_vacuum_size: float = Field(
        15.0,
        description="Size of the vacuum (in Angstrom)",
    )


class SurfaceGenerationRecipe(CatalysisRecipe):
    OPTIONS_CLS = SurfaceGenerationOptions

    def run(self):
        start_time = time.process_time()

        struct = self.get_inputs()
        opts = self.get_options()
        max_index = opts.get("max_miller_index", None)
        miller_indices = opts.get("miller_indices", None)

        if miller_indices is not None:
            if not isinstance(miller_indices, list):
                pass

            elif len(miller_indices) == 0:
                pass

            elif isinstance(miller_indices[0], int):
                slabs = [sl for sl in self.gen_slabs(miller_indices=miller_indices)]

            elif isinstance(miller_indices[0], list):
                slabs = [
                    sl
                    for mi in miller_indices
                    for sl in self.gen_slabs(miller_indices=mi)
                ]

        elif max_index is not None:
            slabs = [
                sl
                for hjk in get_symmetrically_distinct_miller_indices(struct, max_index)
                for sl in self.gen_slabs(miller_indices=hjk)
            ]

        else:
            raise ValueError("No information was passed to generate slab")

        end_time = time.process_time()
        duration = round(end_time - start_time, 6)

        return self.postprocess(slabs, duration)

    def gen_slabs(self, miller_indices: Tuple, tol=0.1, ftol=0.1):
        generator = self.get_slab_generator(miller_indices)

        slabs = generator.get_slabs()
        slabs = [self.format_slab(sl) for sl in slabs]

        return slabs

    def get_slab_generator(self, miller_indices: Tuple) -> SlabGenerator:
        structure = self.get_inputs()
        opts = self.get_options()

        generator = SlabGenerator(
            structure,
            miller_indices,
            min_slab_size=opts["min_slab_size"],
            min_vacuum_size=opts["min_vacuum_size"],
            in_unit_planes=False,
            max_normal_search=max(miller_indices),
        )

        return generator

    def format_slab(self, slab: Slab):
        self.set_site_location(slab)
        self.set_selective_dynamics(slab)
        return slab.get_orthogonal_c_slab()

    def set_site_properties(self, slab: Slab, property_name: str, values):
        for s, v in zip(slab.sites, values):
            s.properties[property_name] = v

    def set_site_location(self, slab: Slab) -> List[str]:
        # {site_index: "location"} mapping
        site_location = {i: "bulk" for i in range(len(slab))}

        for loc, sites in slab.get_surface_sites().items():
            for site, i in sites:
                site_location[i] = loc

        locations = [site_location[i] for i in range(len(site_location))]

        self.set_site_properties(slab, "location", locations)

        return locations

    def set_selective_dynamics(self, slab: Slab) -> List[Tuple[bool]]:
        locations = slab.site_properties["location"]

        seldyn = [
            [True, True, True] if loc != "bottom" else [False, False, False]
            for loc in locations
        ]

        self.set_site_properties(slab, "selective_dynamics", seldyn)

        return seldyn

    def postprocess(self, slabs: List[Slab], duration: float) -> JobResults:
        nodes = []
        for sl in slabs:
            info = CrystalInfo.from_pymatgen(sl)
            info.attributes = {
                "miller_index": sl.miller_index,
                "reconstruction": sl.reconstruction,
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
