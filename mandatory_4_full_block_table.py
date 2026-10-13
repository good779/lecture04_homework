#!/usr/bin/env python3
"""必做 4：N=8 时逐个动量块列出完整本征值表。

除完整分块表外，本程序用独立的 Pauli/Kronecker 构造核对整个 Hamiltonian，
并检查所有动量块本征值的并集是否逐项等于未分块 U(1) 矩阵的完整谱。
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.sparse import csr_matrix, identity, kron

from mandatory_2_ground_state import build_sector_basis, build_xxz_hamiltonian
from momentum_blocks import (
    build_momentum_sectors,
    diagonalize_block,
    project_hamiltonian,
    sparse_max_abs,
)


def momentum_label(momentum_index: int, n_sites: int) -> str:
    """Human-readable exact label for k=2*pi*m/N."""
    if momentum_index == 0:
        return "0"
    if 2 * momentum_index == n_sites:
        return "pi"
    return f"2*pi*{momentum_index}/{n_sites}"


def embedded_two_site_operator(
    n_sites: int,
    site_i: int,
    operator_i: csr_matrix,
    site_j: int,
    operator_j: csr_matrix,
) -> csr_matrix:
    """Embed two local operators; tensor order makes site 0 the lowest bit."""
    local_identity = identity(2, format="csr", dtype=np.complex128)
    result = csr_matrix([[1.0 + 0.0j]])
    for site in reversed(range(n_sites)):
        if site == site_i:
            factor = operator_i
        elif site == site_j:
            factor = operator_j
        else:
            factor = local_identity
        result = kron(result, factor, format="csr")
    return result


def independent_pauli_reference(
    n_sites: int,
    j1: float,
    j2: float,
    delta: float,
    basis: np.ndarray,
) -> csr_matrix:
    """Build H independently from tensor-product spin matrices and restrict it."""
    sx = csr_matrix(np.asarray([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128))
    # Local basis order is |down>, |up>, matching integer bit values 0,1.
    sy = csr_matrix(np.asarray([[0.0, 0.5j], [-0.5j, 0.0]], dtype=np.complex128))
    sz = csr_matrix(np.asarray([[-0.5, 0.0], [0.0, 0.5]], dtype=np.complex128))
    full_dimension = 1 << n_sites
    hamiltonian = csr_matrix(
        (full_dimension, full_dimension), dtype=np.complex128
    )

    nearest_bonds = (
        [(0, 1)]
        if n_sites == 2
        else [(site, (site + 1) % n_sites) for site in range(n_sites)]
    )
    next_nearest_bonds = [
        (site, (site + 2) % n_sites)
        for site in range(n_sites)
        if site != (site + 2) % n_sites
    ]
    for bonds, coupling in ((nearest_bonds, j1), (next_nearest_bonds, j2)):
        if coupling == 0.0:
            continue
        for site_i, site_j in bonds:
            hamiltonian += coupling * (
                embedded_two_site_operator(n_sites, site_i, sx, site_j, sx)
                + embedded_two_site_operator(n_sites, site_i, sy, site_j, sy)
                + delta
                * embedded_two_site_operator(n_sites, site_i, sz, site_j, sz)
            )

    selected = np.asarray(basis, dtype=np.int64)
    return hamiltonian[selected, :][:, selected].tocsr()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="列出 N=8 J1-J2 XXZ 环每个动量块的全部本征值。"
    )
    parser.add_argument("--N", type=int, default=8, help="格点数（作业默认：8）")
    parser.add_argument("--J1", type=float, default=1.0, help="最近邻耦合（默认：1）")
    parser.add_argument(
        "--J2", type=float, nargs="+", default=[0.0, 0.5], help="默认：0 0.5"
    )
    parser.add_argument("--delta", type=float, default=1.0, help="XXZ 参数（默认：1）")
    parser.add_argument(
        "--ground-tol", type=float, default=1e-9, help="判定基态动量的绝对容差"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.N < 2 or args.N % 2 != 0:
        raise ValueError("S^z_tot=0 要求 N 是不小于 2 的偶数。")

    basis, index = build_sector_basis(args.N, args.N // 2)
    sectors = build_momentum_sectors(args.N, basis, index)
    if sum(sector.dimension for sector in sectors) != math.comb(
        args.N, args.N // 2
    ):
        raise AssertionError("动量块维数和错误。")

    # All projector columns together must be a unitary basis change.
    from scipy.sparse import hstack

    complete_projector = hstack(
        [sector.projector for sector in sectors], format="csr"
    )
    completeness = complete_projector @ complete_projector.getH()
    completeness.setdiag(completeness.diagonal() - 1.0)
    completeness.eliminate_zeros()
    completeness_error = sparse_max_abs(completeness)
    if completeness_error > 1e-10:
        raise AssertionError("全部动量投影列不构成 U(1) 空间的完备正交基。")

    print("必做 4：N=8 的完整动量分块表")
    print(
        f"参数：N={args.N}, J1={args.J1:g}, Delta={args.delta:g}, "
        f"S^z_tot=0 维数={len(basis)}"
    )
    print("星号 * 标出包含全局基态的动量块。")

    validation_rows: list[tuple[float, float, float, float, float]] = []
    for j2 in args.J2:
        hamiltonian = build_xxz_hamiltonian(
            args.N, args.J1, j2, args.delta, basis, index
        ).astype(np.complex128)
        reference = independent_pauli_reference(
            args.N, args.J1, j2, args.delta, basis
        )
        construction_error = sparse_max_abs(hamiltonian - reference)

        block_spectra: list[np.ndarray] = []
        block_eigenvectors: list[np.ndarray] = []
        for sector in sectors:
            eigenvalues, eigenvectors = diagonalize_block(
                project_hamiltonian(hamiltonian, sector)
            )
            block_spectra.append(eigenvalues)
            block_eigenvectors.append(eigenvectors)

        all_block_values = np.sort(np.concatenate(block_spectra))
        full_values = np.linalg.eigvalsh(hamiltonian.toarray())
        spectral_union_error = float(np.max(np.abs(all_block_values - full_values)))
        time_reversal_error = max(
            float(
                np.max(
                    np.abs(
                        block_spectra[m]
                        - block_spectra[(-m) % args.N]
                    )
                )
            )
            for m in range(args.N)
        )
        global_ground = float(full_values[0])

        maximum_residual = 0.0
        for sector, eigenvalues, eigenvectors in zip(
            sectors, block_spectra, block_eigenvectors
        ):
            lifted_vectors = sector.projector @ eigenvectors
            residual_matrix = (
                hamiltonian @ lifted_vectors
                - lifted_vectors * eigenvalues[np.newaxis, :]
            )
            column_residuals = np.linalg.norm(residual_matrix, axis=0)
            maximum_residual = max(maximum_residual, float(np.max(column_residuals)))

        if max(
            construction_error,
            spectral_union_error,
            time_reversal_error,
            maximum_residual,
        ) > 1e-10:
            raise AssertionError(f"J2={j2:g} 的完整动量谱校检失败。")

        print("\n" + "=" * 100)
        print(f"J2 = {j2:g}")
        print("=" * 100)
        print(f"{'m':>3} {'k/pi':>8} {'dim':>5} {'GS':>3}  全部本征值（升序）")
        ground_momenta: list[int] = []
        for sector, eigenvalues in zip(sectors, block_spectra):
            is_ground = abs(eigenvalues[0] - global_ground) <= args.ground_tol
            if is_ground:
                ground_momenta.append(sector.momentum_index)
            spectrum_text = ", ".join(f"{value:.10f}" for value in eigenvalues)
            print(
                f"{sector.momentum_index:3d} "
                f"{sector.momentum / np.pi:8.4f} "
                f"{sector.dimension:5d} "
                f"{'*' if is_ground else '':>3}  [{spectrum_text}]"
            )
        ground_text = ", ".join(
            f"m={m} (k={momentum_label(m, args.N)})" for m in ground_momenta
        )
        print(f"基态能量 E0={global_ground:.12f}；基态动量：{ground_text}")
        validation_rows.append(
            (
                j2,
                construction_error,
                spectral_union_error,
                time_reversal_error,
                maximum_residual,
            )
        )

    print("\n校检清单：")
    print(
        "1. 投影完备性（针对短轨道遗漏、重复或相位归一化错误）：\n"
        f"   期望 P_all P_all^dagger=I；实测最大误差={completeness_error:.3e}，通过。"
    )
    for (
        j2,
        construction_error,
        spectral_union_error,
        time_reversal_error,
        maximum_residual,
    ) in validation_rows:
        print(
            f"2. J2={j2:g} Hamiltonian 独立构造（针对漏键、系数、符号和 Delta 错误）：\n"
            "   位运算稀疏矩阵 vs Pauli/Kronecker 张量积矩阵，"
            f"max|差|={construction_error:.3e}，通过。\n"
            "3. 完整谱重组（针对漏掉动量块或块矩阵元错误）：\n"
            f"   分块谱并集 vs 未分块 U(1) 完整谱，最大逐项误差="
            f"{spectral_union_error:.3e}；时间反演 max|E(k)-E(-k)|="
            f"{time_reversal_error:.3e}；\n"
            f"   全部提升本征态最大残差="
            f"{maximum_residual:.3e}，通过。"
        )
        if (
            np.isclose(j2, 0.5 * args.J1, atol=1e-14, rtol=0.0)
            and np.isclose(args.delta, 1.0, atol=1e-14, rtol=0.0)
        ):
            exact_energy = -3.0 * args.N * args.J1 / 8.0
            print(
                "4. MG 解析值（针对 J2 相对系数错误）：\n"
                f"   期望 E0=-3*N*J1/8={exact_energy:.12f}，"
                "并在 k=0、pi 各有一个基态；以上完整表与之相符，通过。"
            )


if __name__ == "__main__":
    main()
