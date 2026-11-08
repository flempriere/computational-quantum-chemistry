from collections.abc import Sequence

import numpy as np

from computational_quantum_chemistry.model_validator import Molecule


class MoleculeRuntime:
    def __init__(self, symbols, coordinates, bonds):
        self.symbols: Sequence[str] = symbols
        self.coordinates = coordinates
        self.bonds: Sequence[tuple[int, int]] = bonds


def compile_molecule_to_runtime(molecule: Molecule) -> MoleculeRuntime:
    return MoleculeRuntime(
        [atom.symbol for atom in molecule.atoms],
        coordinates=np.array([atom.coordinates for atom in molecule.atoms]),
        bonds=([(bond.first, bond.second) for bond in molecule.bonds]),
    )
