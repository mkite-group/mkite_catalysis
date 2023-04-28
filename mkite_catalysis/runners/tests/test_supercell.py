import numpy as np
import unittest as ut
from unittest.mock import patch

from mkite_catalysis.runners.supercell import enumerate_matrices, SupercellCalculator


class TestEnumerateMatrices(ut.TestCase):
    def test_enumeration(self):
        matrices = enumerate_matrices(2)
        self.assertEqual(matrices.shape, (95, 2, 2))

        matrices = enumerate_matrices(3)
        self.assertEqual(matrices.shape, (555, 2, 2))


class TestSupercellCalculator(ut.TestCase):
    def setUp(self):
        self.lattice = np.array([[3, 0], [1, 2]])
        self.example_matrices = np.array(
            [
                [[1, 0], [1, 1]],
                [[1, -1], [1, 1]],
                [[1, -2], [1, 1]],
            ]
        )

        self.example_scaled = np.array(
            [[[3, 0], [4, 2]], [[2, -2], [4, 2]], [[1, -4], [4, 2]]]
        )

        self.example_areas = np.array([1, 2, 3])

        self.threshold = 4
        self.calc = SupercellCalculator(
            self.lattice,
            dist_threshold=self.threshold,
            max_vector=2,
        )

    @patch("mkite_catalysis.runners.supercell.enumerate_matrices")
    def test_get_scaled_lattices(self, enumerate_fn_mock):
        enumerate_fn_mock.return_value = self.example_matrices
        scaled = self.calc.get_scaled_lattices(self.example_matrices)

        self.assertTrue(np.allclose(scaled, self.example_scaled))

    def test_get_areas(self):
        areas = self.calc.get_areas(self.example_scaled)

        self.assertTrue(np.allclose(areas, self.example_areas))

    def test_get_criteria(self):
        criteria, _ = self.calc.get_criteria(self.example_scaled)

        self.assertEqual(criteria.tolist(), [False, False, True])

    @patch("mkite_catalysis.runners.supercell.enumerate_matrices")
    def test_find_supercells(self, enumerate_fn_mock):
        enumerate_fn_mock.return_value = self.example_matrices

        scaled, areas, matrices, angles = self.calc.find_supercells()

        self.assertEqual(len(scaled), 1)
        self.assertEqual(len(areas), 1)
        self.assertEqual(len(matrices), 1)
