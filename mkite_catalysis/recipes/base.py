import os

from mkite_core.recipes import PythonRecipe, EnvSettings, BaseOptions
from mkite_core.models import CrystalInfo


class CatalysisRecipe(PythonRecipe):
    _PACKAGE_NAME = "mkite_catalysis"
    _METHOD = "GEN"

    SETTINGS_CLS = EnvSettings
    OPTIONS_CLS = BaseOptions

    def get_inputs(self):
        inputs = super().get_inputs()
        info = CrystalInfo.from_dict(inputs[0])
        return info.as_pymatgen()

    def run(self):
        pass
