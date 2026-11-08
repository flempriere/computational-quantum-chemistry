from collections.abc import Sequence
from typing import Literal, Self

import pydantic


class Metadata(pydantic.BaseModel):
    name: str
    units: dict[str, str] = {"distance": "angstroms"}


class Atom(pydantic.BaseModel):
    id: pydantic.NonNegativeInt
    symbol: str
    coordinates: tuple[float, float, float]


class Bond(pydantic.BaseModel):
    first: pydantic.NonNegativeInt
    second: pydantic.NonNegativeInt


class Molecule(pydantic.BaseModel):
    schema_version: Literal["0.1.0"]
    metadata: Metadata
    atoms: Sequence[Atom]
    bonds: Sequence[Bond]

    @pydantic.field_validator("atoms", mode="before")
    @classmethod
    def validate_atoms(cls, atoms_dict: Sequence[dict]) -> Sequence[Atom]:
        atoms = [Atom(**atom) for atom in atoms_dict]
        return sorted(atoms, key=lambda a: a.id)

    @pydantic.field_validator("bonds", mode="before")
    @classmethod
    def validate_bonds(cls, bonds_dict: Sequence[tuple[int, int]]) -> Sequence[Bond]:
        bonds = [Bond(first=bond[0], second=bond[1]) for bond in bonds_dict]
        return bonds

    @pydantic.model_validator(mode="after")
    def validate_topology(self) -> Self:
        expected_ids = list(range(len(self.atoms)))
        actual_ids = [atom.id for atom in self.atoms]

        if actual_ids != expected_ids:
            raise ValueError(
                "Atom id's must be sequential starting from 0.\n"
                f"Expected {expected_ids}]\nFound {actual_ids}."
            )

        valid_ids = set(actual_ids)
        for bond in self.bonds:
            if bond.first not in valid_ids or bond.second not in valid_ids:
                raise ValueError(
                    f"Bond ({bond.first}, {bond.second}) references non-existent atom ID(s).\n Valid atom IDs are {sorted(valid_ids)}."
                )

            if bond.first == bond.second:
                raise ValueError(
                    f"Bond ({bond.first}, {bond.second}) contains invalid-self bond."
                )

            if self.atoms[bond.first].symbol > self.atoms[bond.second].symbol:
                tmp = bond.first
                bond.first = bond.second
                bond.second = tmp

        return self

    @classmethod
    def from_dict(cls, kwargs: dict) -> Molecule:
        return cls(**kwargs)
