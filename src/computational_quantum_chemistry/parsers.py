import pathlib

import yaml

from computational_quantum_chemistry.model_validator import Molecule


def load_molecule_from_yaml(filepath: pathlib.Path | str) -> Molecule:
    with open(filepath, "r") as f:
        molecule_dictionary = yaml.safe_load(f)
    return Molecule.from_dict(molecule_dictionary)
