from computational_quantum_chemistry.molecule import compile_molecule_to_runtime
from computational_quantum_chemistry.optimise import (
    optimise_molecule_geometry,
)
from computational_quantum_chemistry.parsers import load_molecule_from_yaml

files = ["water.yaml", "methane.yaml", "water.yaml"]

for file in files:
    print(f"Testing {file.removesuffix('.yaml')}")

    validated_model = load_molecule_from_yaml(file)
    molecule = compile_molecule_to_runtime(validated_model)

    results = optimise_molecule_geometry(molecule)
    print("Optimised geometry...")
    print(f"Coordinates:\n{results.coordinates}\nEnergy:\n{results.energy}")
