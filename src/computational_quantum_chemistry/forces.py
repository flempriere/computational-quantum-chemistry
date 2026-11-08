from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from computational_quantum_chemistry.molecule import MoleculeRuntime


@dataclass(frozen=True)
class BondParameters:
    r0: float
    k: float


force_parameters = {
    ("H", "O"): BondParameters(0.9572, 500.0),
    ("C", "H"): BondParameters(1.0900, 340.0),
    ("C", "C"): BondParameters(1.5300, 310.0),
}


def compile_energy_functional(
    molecule: MoleculeRuntime,
) -> Callable[[np.ndarray], float]:

    r_0 = []
    k = []
    for atom_1, atom_2 in molecule.bonds:
        try:
            params = force_parameters[
                (molecule.symbols[atom_1], molecule.symbols[atom_2])
            ]
            r_0.append(params.r0)
            k.append(params.k)
        except KeyError:
            raise KeyError(
                f"Force parameters missing for bond ({molecule.symbols[atom_1]}, {molecule.symbols[atom_2]})"
            )

    bond_idx = np.array(molecule.bonds)
    r_0 = np.array(r_0)
    k = np.array(k)

    def energy_functional(coordinates: np.ndarray) -> float:
        coordinates = coordinates.reshape(molecule.coordinates.shape)

        r = np.linalg.norm(
            coordinates[bond_idx[:, 0]] - coordinates[bond_idx[:, 1]], axis=1
        )
        energies = 0.5 * k * (r - r_0) ** 2

        return float(np.sum(energies))

    return energy_functional
