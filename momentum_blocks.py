#!/usr/bin/env python3
"""Translation orbits and momentum projectors for the J1-J2 XXZ ring.

The bit and translation conventions are those of the homework:

    T |s> = |rot(s, +1)>,    k_m = 2*pi*m/N.

For an orbit representative ``r`` with period ``L``, the normalized state

    |r;k> = L^(-1/2) sum_{n=0}^{L-1} exp(-i*k*n) T^n |r>

has ``T`` eigenvalue ``exp(+i*k)``.  It exists only when ``m*L`` is a
multiple of ``N``.  Orbits have disjoint supports, so these columns are
orthonormal without a Gram-Schmidt step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class MomentumSector:
    """One momentum sector and its isometric embedding into the U(1) basis."""

    momentum_index: int
    momentum: float
    periods: tuple[int, ...]
    projector: csr_matrix

    @property
    def dimension(self) -> int:
        return self.projector.shape[1]


def rotate_state(state: int, n_sites: int, shift: int = 1) -> int:
    """Cyclically move the bit at site j to site j+shift."""
    if n_sites <= 0:
        raise ValueError("N 必须为正整数。")
    shift %= n_sites
    if shift == 0:
        return state
    mask = (1 << n_sites) - 1
    return ((state << shift) & mask) | (state >> (n_sites - shift))


def build_translation_matrix(
    n_sites: int, basis: np.ndarray, index: dict[int, int]
) -> csr_matrix:
    """Return T in the supplied fixed-magnetization basis."""
    columns = np.arange(len(basis), dtype=np.int64)
    rows = np.asarray(
        [index[rotate_state(int(state), n_sites)] for state in basis],
        dtype=np.int64,
    )
    values = np.ones(len(basis), dtype=np.float64)
    return coo_matrix(
        (values, (rows, columns)), shape=(len(basis), len(basis))
    ).tocsr()


def translation_orbits(n_sites: int, basis: np.ndarray) -> list[tuple[int, ...]]:
    """Partition a fixed-magnetization basis into translation orbits."""
    remaining = {int(state) for state in basis}
    orbits: list[tuple[int, ...]] = []

    for raw_representative in basis:
        representative = int(raw_representative)
        if representative not in remaining:
            continue

        orbit: list[int] = []
        state = representative
        while True:
            if state in orbit:
                if state != representative:
                    raise AssertionError("平移轨道在回到代表元之前发生重复。")
                break
            orbit.append(state)
            state = rotate_state(state, n_sites)

        for state in orbit:
            remaining.remove(state)
        orbits.append(tuple(orbit))

    if remaining or sum(map(len, orbits)) != len(basis):
        raise AssertionError("平移轨道没有恰好分割固定磁化基。")
    return orbits


def build_momentum_sectors(
    n_sites: int, basis: np.ndarray, index: dict[int, int]
) -> list[MomentumSector]:
    """Build the N momentum-sector projectors in the U(1) basis."""
    orbits = translation_orbits(n_sites, basis)
    sectors: list[MomentumSector] = []

    for momentum_index in range(n_sites):
        momentum = 2.0 * np.pi * momentum_index / n_sites
        compatible = [
            orbit
            for orbit in orbits
            if (momentum_index * len(orbit)) % n_sites == 0
        ]

        rows: list[int] = []
        columns: list[int] = []
        values: list[complex] = []
        for column, orbit in enumerate(compatible):
            normalization = np.sqrt(len(orbit))
            for translation_power, state in enumerate(orbit):
                rows.append(index[state])
                columns.append(column)
                values.append(
                    np.exp(-1j * momentum * translation_power) / normalization
                )

        projector = coo_matrix(
            (values, (rows, columns)),
            shape=(len(basis), len(compatible)),
            dtype=np.complex128,
        ).tocsr()
        sectors.append(
            MomentumSector(
                momentum_index=momentum_index,
                momentum=momentum,
                periods=tuple(len(orbit) for orbit in compatible),
                projector=projector,
            )
        )

    return sectors


def character_dimensions(n_sites: int, basis: np.ndarray) -> np.ndarray:
    """Compute dim(k) independently from translation characters.

    The character of T^r is the number of fixed computational basis states.
    Fourier transforming those characters gives the multiplicity of each
    translation eigenvalue.
    """
    fixed_counts = np.asarray(
        [
            sum(
                rotate_state(int(state), n_sites, shift) == int(state)
                for state in basis
            )
            for shift in range(n_sites)
        ],
        dtype=np.float64,
    )
    dimensions: list[int] = []
    translations = np.arange(n_sites)
    for momentum_index in range(n_sites):
        momentum = 2.0 * np.pi * momentum_index / n_sites
        value = np.sum(np.exp(-1j * momentum * translations) * fixed_counts) / n_sites
        if abs(value.imag) > 1e-9 or abs(value.real - round(value.real)) > 1e-9:
            raise AssertionError(f"动量维数的特征标公式未给出整数：{value}")
        dimensions.append(int(round(value.real)))
    return np.asarray(dimensions, dtype=np.int64)


def project_hamiltonian(hamiltonian: csr_matrix, sector: MomentumSector) -> csr_matrix:
    """Return P_k^dagger H P_k as a CSR matrix."""
    return (sector.projector.getH() @ hamiltonian @ sector.projector).tocsr()


def diagonalize_block(block: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return the complete sorted eigensystem of a small momentum block."""
    eigenvalues, eigenvectors = np.linalg.eigh(block.toarray())
    order = np.argsort(eigenvalues.real)
    return eigenvalues[order].real, eigenvectors[:, order]


def lowest_block_eigenpair(
    block: csr_matrix, solver_tolerance: float = 1e-11
) -> tuple[float, np.ndarray]:
    """Return the lowest eigenpair, using ARPACK for nontrivial blocks."""
    eigenvalues, eigenvectors = lowest_block_eigenpairs(
        block, num_eigenvalues=1, solver_tolerance=solver_tolerance
    )
    return float(eigenvalues[0]), eigenvectors[:, 0]


def lowest_block_eigenpairs(
    block: csr_matrix,
    num_eigenvalues: int,
    solver_tolerance: float = 1e-11,
) -> tuple[np.ndarray, np.ndarray]:
    """Return several lowest eigenpairs of a Hermitian momentum block."""
    dimension = block.shape[0]
    if dimension == 0:
        raise ValueError("空动量块没有本征值。")
    if num_eigenvalues <= 0:
        raise ValueError("num_eigenvalues 必须为正整数。")
    keep = min(num_eigenvalues, dimension)
    if dimension <= max(12, keep + 1) or keep == dimension:
        eigenvalues, eigenvectors = diagonalize_block(block)
        return eigenvalues[:keep], eigenvectors[:, :keep]

    initial = np.linspace(1.0, 2.0, dimension, dtype=np.float64).astype(np.complex128)
    initial += 1j * np.linspace(2.0, 1.0, dimension, dtype=np.float64)
    initial /= np.linalg.norm(initial)
    eigenvalues, eigenvectors = eigsh(
        block,
        k=keep,
        which="SA",
        tol=solver_tolerance,
        maxiter=200_000,
        ncv=min(dimension, max(2 * keep + 1, 30)),
        v0=initial,
    )
    order = np.argsort(eigenvalues.real)
    return eigenvalues[order].real, eigenvectors[:, order]


def sparse_max_abs(matrix: csr_matrix) -> float:
    """Maximum absolute stored entry, treating an empty sparse matrix as zero."""
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
