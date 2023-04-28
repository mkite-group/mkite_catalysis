import json
import unittest as ut
from pkg_resources import resource_filename

from pymatgen.core.surface import Slab, SlabGenerator
from mkite_core.models import JobInfo
from mkite_catalysis.recipes.surfgen import SurfaceGenerationRecipe


INFO = resource_filename("mkite_catalysis.tests.files", "jobinfo.json")
SLAB_EXAMPLE = resource_filename("mkite_catalysis.tests.files", "slab.json")


INFO_OPTIONS = {
    "max_miller_index": 2,
    "miller_indices": (1, 0, 0),
    "min_slab_size": 4,
    "min_vacuum_size": 15,
    "min_lateral_size": 7,
}


class TestSurfaceGeneration(ut.TestCase):
    def setUp(self):
        self.info = JobInfo.from_json(INFO)
        self.info.options = INFO_OPTIONS
        self.recipe = SurfaceGenerationRecipe(self.info)

    def get_slab(self):
        with open(SLAB_EXAMPLE, "r") as f:
            data = json.load(f)

        return Slab.from_dict(data)

    def test_done_job(self):
        data = self.recipe.get_done_job()
        expected = {
            "id": 1,
            "uuid": "7615c560-962d-4053-8c73-36f313d6fa36",
            "status": "D",
        }

        for k, v in expected.items():
            self.assertEqual(data[k], v)

        self.assertTrue("options" in data)

    def test_get_slab_generator(self):
        gen = self.recipe.get_slab_generator([1, 0, 0])

        self.assertIsInstance(gen, SlabGenerator)

    def test_get_slab(self):
        gen = self.recipe.get_slab_generator([1, 0, 0])
        slab = gen.get_slab()

        self.assertEqual(len(slab), 4)
        self.assertAlmostEqual(slab.lattice.a, 3.8608030252785492, places=4)
        self.assertAlmostEqual(slab.lattice.c, 27.025621176949844, places=4)

    def test_set_site_location(self):
        slab = self.get_slab()
        self.recipe.set_site_location(slab)
        expected = ["bottom", "bottom", "bulk", "bulk", "bulk", "bulk", "top", "top"]

        self.assertTrue("location" in slab.site_properties)
        self.assertEqual(slab.site_properties["location"], expected)

    def test_set_selective_dynamics(self):
        slab = self.get_slab()
        self.recipe.set_site_location(slab)
        self.recipe.set_selective_dynamics(slab)

        expected = [
            [False, False, False],
            [False, False, False],
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ]

        self.assertTrue("selective_dynamics" in slab.site_properties)
        self.assertEqual(slab.site_properties["selective_dynamics"], expected)

    def test_run(self):
        def set_options(key, value):
            self.recipe.info.options[key] = value

        set_options("miller_indices", (1, 0, 0))
        set_options("max_miller_index", None)

        slabs = self.recipe.run()
        self.assertEqual(len(slabs.nodes), 1)

        set_options("miller_indices", None)
        set_options("max_miller_index", 2)

        slabs = self.recipe.run()
        self.assertEqual(len(slabs.nodes), 9)

        set_options("miller_indices", None)
        set_options("max_miller_index", None)

        with self.assertRaises(ValueError):
            self.recipe.run()
