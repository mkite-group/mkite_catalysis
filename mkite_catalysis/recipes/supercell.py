import os
import time
import math
import numpy as np
from typing import List, Dict
from pydantic import Field
from pymatgen.core import Structure

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


class SupercellGenerationOptions(BaseOptions):
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


class SupercellGenerationRecipe(CatalysisRecipe):
    OPTIONS_CLS = SupercellGenerationOptions

    def run(self):
        start_time = time.process_time()

        surface = self.get_inputs()

        surface, scale = self.make_lateral_supercell(surface)

        end_time = time.process_time()
        duration = round(end_time - start_time, 6)

        return self.postprocess(surface, duration=duration, scale=scale)

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

    def postprocess(self, surface: Structure, duration: float, scale: float) -> JobResults:
        info = CrystalInfo.from_pymatgen(surface)
        info.attributes = {
            "scale": scale,
        }

        nodes = [NodeResults(chemnode=info.as_dict(), calcnodes=[])]

        runstats = self.get_run_stats(duration)
        jobres = JobResults(
            job=self.get_done_job(),
            runstats=runstats,
            nodes=nodes,
        )

        jobres.to_json(os.path.join(".", JobResults.file_name()))
        return jobres
