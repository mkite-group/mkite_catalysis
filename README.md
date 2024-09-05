# mkite_catalysis

`mkite_catalysis` is a plugin to run catalysis-related recipes with mkite. 
As of now, the plugin has the following recipes implemented: surface cutting, supercell generation, and combinatorial adsorption, implemented with the help of [pymatgen](https://github.com/materialsproject/pymatgen).

## Documentation

General tutorial for `mkite` and its plugins are available in the [main documentation](https://mkite.org).
Complete API documentation is pending.

## Installation

To install `mkite_catalysis`, use pip:

```bash
pip install mkite_catalysis
```

Alternatively, for a development version, clone this repo and install it in editable form:

```bash
pip install -U git+https://github.com/mkite-group/mkite_catalysis
```

## Contributions

Contributions to the entire mkite suite are welcomed.
You can send a pull request or open an issue for this plugin or either of the packages in mkite.
When doing so, please adhere to the [Code of Conduct](CODE_OF_CONDUCT.md) in the mkite suite.

The mkite package was created by Daniel Schwalbe-Koda <dskoda@ucla.edu>.

### Citing mkite

If you use mkite in a publication, please cite the following paper:

```bibtex
@article{mkite2023,
    title = {mkite: A distributed computing platform for high-throughput materials simulations},
    author = {Schwalbe-Koda, Daniel},
    year = {2023},
    journal = {arXiv:2301.08841},
    doi = {10.48550/arXiv.2301.08841},
    url = {https://doi.org/10.48550/arXiv.2301.08841},
    arxiv={2301.08841},
}
```

## License

The mkite suite is distributed under the following license: Apache 2.0 WITH LLVM exception.

All new contributions must be made under this license.

SPDX: Apache-2.0, LLVM-exception

LLNL-CODE-848161
