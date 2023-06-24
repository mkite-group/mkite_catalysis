import os
import amd
import numpy as np
from ase import Atoms
from ase.io import write
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from typing import List, Union
from tempfile import TemporaryDirectory
from scipy.spatial.distance import squareform


class Deduplicator:
    """Deduplicates the structures using the Pointwise Distance Distribution"""

    def __init__(self, k: int = 100, tol: float = 1e-3):
        self.k = k
        self.tol = tol

    def write_cifs(self, crystals: List[Union[Atoms, Structure]], rootdir: os.PathLike) -> os.PathLike:
        path = os.path.join(rootdir, "atoms.cif")
        atoms = []
        for c in crystals:
            if isinstance(c, Structure):
                atoms.append(AseAtomsAdaptor.get_atoms(c))
            elif isinstance(c, Atoms):
                atoms.append(c)
            else:
                raise ValueError(f"Crystal of type {type(c)} not recognized!")

        write(path, atoms)
        return path

    def get_pdds(self, crystals: List[Union[Atoms, Structure]]):
        with TemporaryDirectory() as tmp:
            path = self.write_cifs(crystals, tmp)
            structs = list(amd.CifReader(path))

        pdds = [amd.PDD(c, k=self.k) for c in structs]
        return pdds

    def get_pdd_distance_matrix(self, pdds: List[amd.PDD]):
        pdd_cdm = amd.PDD_pdist(pdds)
        return squareform(pdd_cdm)

    def find_unique_idx(self, dm: np.ndarray):
        rows, cols = np.where(dm < self.tol)
        remove = [j for i, j in zip(rows, cols) if i < j]
        unique = [i for i in range(len(dm)) if i not in remove]
        return unique

    def deduplicate(self, crystals: List[Union[Atoms, Structure]]):
        if not crystals:
            return crystals

        pdds = self.get_pdds(crystals)
        dm = self.get_pdd_distance_matrix(pdds)
        unique = self.find_unique_idx(dm)

        return [crystals[i] for i in unique]
