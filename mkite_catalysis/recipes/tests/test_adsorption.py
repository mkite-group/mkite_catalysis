import json
import math
import unittest as ut
from pkg_resources import resource_filename

from pymatgen.core import Structure, Molecule
from mkite_core.models import JobInfo, JobResults
from mkite_core.tests.tempdirs import run_in_tempdir
from mkite_catalysis.recipes.adsorption import AdsorptionRecipe


INFO = resource_filename("mkite_catalysis.tests.files", "jobinfo_surf.json")


INFO_OPTIONS = {
    "min_lateral_size": 6,
    "max_supercell_vector": 3,
    "min_supercell_angle": math.pi / 6,
    "adsorption_height": 2.0,
}


class TestAdsorptionRecipe(ut.TestCase):
    def setUp(self):
        self.info = JobInfo.from_json(INFO)
        self.info.options = INFO_OPTIONS
        self.recipe = AdsorptionRecipe(self.info)

    def test_inputs(self):
        surface, adsorbate = self.recipe.get_inputs()

        self.assertIsInstance(surface, Structure)
        self.assertIsInstance(adsorbate, Molecule)

    def test_input_attributes(self):
        surface_attrs = self.recipe.get_input_attributes("Crystal")
        self.assertTrue("miller_index" in surface_attrs)

    def test_make_lateral_supercell(self):
        surface, _ = self.recipe.get_inputs()

        scell, scale = self.recipe.make_lateral_supercell(surface)

        self.assertNotEqual(surface, scell)
        self.assertTrue(surface is not scell)

        self.assertEqual(len(scell), 36)
        self.assertTrue(scell.lattice.a > INFO_OPTIONS["min_lateral_size"])

    def test_adsorb(self):
        surface, adsorbate = self.recipe.get_inputs()

        adsorbed = self.recipe.adsorb(surface, adsorbate)

        self.assertIsInstance(adsorbed, dict)
        self.assertTrue("ontop" in adsorbed)
        self.assertIsInstance(adsorbed["ontop"], list)
        self.assertIsInstance(adsorbed["ontop"][0], Structure)

    @run_in_tempdir
    def test_postprocess(self):
        surface, _ = self.recipe.get_inputs()
        mock_structures = {"ontop": [surface]}

        results = self.recipe.postprocess(mock_structures, duration=2)

        self.assertIsInstance(results, JobResults)
        self.assertTrue("adsorption_site" in results.nodes[0].chemnode["attributes"])
        self.assertEqual(results.nodes[0].calcnodes, [])
