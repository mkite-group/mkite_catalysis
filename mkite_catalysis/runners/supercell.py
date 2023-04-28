import numpy as np
from pymatgen.core import Lattice


def enumerate_matrices(max_vector: int) -> np.ndarray:
    """Enumerate all 2x2 transformation matrices up to a cutoff `max_vector`.
    Returns an (N, 2, 2) tensor for simplicity.
    """
    amax = np.abs(max_vector)
    x = np.arange(-amax + 1, amax + 1)

    matrices = np.stack(np.meshgrid(x, x, x, x)).T.reshape(-1, 2, 2)
    dets = np.linalg.det(matrices)

    selection = dets > 0
    matrices = matrices[selection]

    return matrices


class SupercellCalculator:
    def __init__(
        self,
        lattice: np.ndarray,
        dist_threshold: float = 6,
        max_vector: int = 2,
        min_angle: float = np.pi / 6,
    ):
        self.lattice = lattice
        self.threshold = dist_threshold
        self.max_vector = max_vector
        self.min_angle = min_angle

    @classmethod
    def from_pymatgen(cls, lattice: Lattice, **kwargs):
        surf_lattice = lattice.matrix[:2, :2]
        return cls(surf_lattice, **kwargs)

    def get_scaled_lattices(self, matrices: np.ndarray):
        scaled = matrices @ self.lattice
        return scaled

    def get_areas(self, lattices: np.ndarray):
        init_area = np.linalg.det(self.lattice)
        areas = np.linalg.det(lattices) / init_area

        return areas

    def get_criteria(self, lattices: np.ndarray):
        a, b = lattices[:, 0, :], lattices[:, 1, :]

        da = np.linalg.norm(a, axis=-1)
        db = np.linalg.norm(b, axis=-1)
        d1 = np.linalg.norm(a + b, axis=-1)
        d2 = np.linalg.norm(a - b, axis=-1)

        dot = (a * b).sum(-1)
        angles = np.arccos(dot / (da * db))
        ha = np.abs(da * np.sin(angles))
        hb = np.abs(db * np.sin(angles))

        criteria = np.bitwise_and.reduce(
            [
                da >= self.threshold,
                db >= self.threshold,
                d1 >= self.threshold,
                d2 >= self.threshold,
                angles >= self.min_angle,
                angles <= (np.pi - self.min_angle),
            ]
        )

        return criteria, angles

    def sort_by(self, *args, key=None):
        if key is None:
            raise ValueError("Please specify a key for sorting all arrays")

        indices = list(range(len(key)))
        indices = sorted(indices, key=lambda i: key[i])

        sorted_args = []
        for arg in args:
            sorted_args.append(arg[indices])

        return sorted_args

    def select_true(self, *args, key=None):
        if key is None:
            raise ValueError("Please specify a key for sorting all arrays")

        sorted_args = []
        for arg in args:
            sorted_args.append(arg[key])

        return sorted_args

    def find_supercells(self):
        matrices = enumerate_matrices(self.max_vector)
        scaled = self.get_scaled_lattices(matrices)
        areas = self.get_areas(scaled)

        obey_distances, angles = self.get_criteria(scaled)

        scaled, areas, matrices, angles = self.select_true(
            scaled, areas, matrices, angles, key=obey_distances
        )

        min_area = areas == np.min(areas)
        scaled, areas, matrices, angles = self.select_true(
            scaled, areas, matrices, angles, key=min_area
        )

        orthogonality = np.abs(angles - np.pi / 2)
        scaled, areas, matrices, angles = self.sort_by(
            scaled, areas, matrices, angles, key=orthogonality
        )

        return scaled, areas, matrices, angles

    def get_best_transformation(self, three_d: bool = True):
        _, _, matrices, _ = self.find_supercells()

        matrix = matrices[0]

        if three_d:
            eye = np.eye(3, dtype=int)
            eye[:2, :2] = matrix
            matrix = eye

        return matrix
