from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy as sp

from computational_quantum_chemistry.forces import compile_energy_functional
from computational_quantum_chemistry.molecule import MoleculeRuntime


@dataclass(frozen=True)
class OptimiserResults:
    coordinates: np.ndarray
    energy: float


def geometry_optimise(
    coordinates: np.ndarray, energy_functional: Callable[[np.ndarray], float]
) -> np.ndarray:
    return sp.optimize.minimize(energy_functional, coordinates.flatten()).x.reshape(
        coordinates.shape
    )


def optimise_molecule_geometry(molecule: MoleculeRuntime) -> OptimiserResults:
    energy = compile_energy_functional(molecule)
    optimised_coordinates = geometry_optimise(molecule.coordinates, energy)
    return OptimiserResults(optimised_coordinates, energy(optimised_coordinates))
