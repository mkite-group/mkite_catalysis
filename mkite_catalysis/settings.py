from pydantic import Field
from mkite_core.external import load_config
from pkg_resources import resource_filename

from mkite_core.recipes import EnvSettings, BaseOptions


class CatalysisOptions(BaseOptions):
    @classmethod
    def get_defaults(cls):
        return cls(**DEFAULT_OPTIONS)


class CatalysisSettings(EnvSettings):
    DEFAULT_OPTIONS: dict = Field(
        default_factory=VaspOptions.get_defaults,
        description="File where to load the default calculation details\
            for VASP",
    )
