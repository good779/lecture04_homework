#!/usr/bin/env python3
"""必做 3：N=16 的动量分块维数与每个 k 扇区的最低本征值。

默认计算题目指定的 N=16、J1=1、Delta=1、J2=0/0.5。所有模型参数
均可由命令行传入。程序还执行独立维数公式、平移对易关系、提升后本征态
残差、未分块 U(1) 基态以及 Majumdar-Ghosh 解析解等非平凡校检。
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.sparse.linalg import eigsh

from mandatory_2_ground_state import (
    build_sector_basis,
    build_xxz_hamiltonian,
    product_of_singlets,
)
from momentum_blocks import (
    build_momentum_sectors,
    build_translation_matrix,
    character_dimensions,
    lowest_block_eigenpair,
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


def mg_validation_suite() -> list[dict[str, float]]:
    """Check MG energy, two dimer eigenstates, and their k=0/pi combinations."""
    records: list[dict[str, float]] = []
    for n_sites in (8, 10, 12):
        basis, index = build_sector_basis(n_sites, n_sites // 2)
        hamiltonian = build_xxz_hamiltonian(
            n_sites, 1.0, 0.5, 1.0, basis, index
        )
        translation = build_translation_matrix(n_sites, basis, index)
        exact_energy = -3.0 * n_sites / 8.0

        pairs_0 = [(site, site + 1) for site in range(0, n_sites, 2)]
        pairs_1 = [
            ((site + 1) % n_sites, (site + 2) % n_sites)
            for site in range(0, n_sites, 2)
        ]
        dimer_0 = product_of_singlets(n_sites, pairs_0, index)
        dimer_1 = product_of_singlets(n_sites, pairs_1, index)
        residual_0 = float(
            np.linalg.norm(hamiltonian @ dimer_0 - exact_energy * dimer_0)
        )
        residual_1 = float(
            np.linalg.norm(hamiltonian @ dimer_1 - exact_energy * dimer_1)
        )

        # D0 and D1 are exchanged by T rather than individually being T eigenstates.
        exchange_error = max(
            float(np.linalg.norm(translation @ dimer_0 - dimer_1)),
            float(np.linalg.norm(translation @ dimer_1 - dimer_0)),
        )
        plus = dimer_0 + dimer_1
        minus = dimer_0 - dimer_1
        plus /= np.linalg.norm(plus)
        minus /= np.linalg.norm(minus)
        momentum_error = max(
            float(np.linalg.norm(translation @ plus - plus)),
            float(np.linalg.norm(translation @ minus + minus)),
        )

        numerical_energy = float(
            eigsh(
                hamiltonian,
                k=1,
                which="SA",
                tol=1e-12,
                v0=np.linspace(1.0, 2.0, len(basis)),
            )[0][0]
        )
        energy_error = abs(numerical_energy - exact_energy)
        if max(
            residual_0,
            residual_1,
            exchange_error,
            momentum_error,
            energy_error,
        ) > 1e-9:
            raise AssertionError(f"N={n_sites} 的 Majumdar-Ghosh 校检失败。")

        records.append(
            {
                "N": float(n_sites),
                "energy": numerical_energy,
                "energy_error": energy_error,
                "dimer_residual": max(residual_0, residual_1),
                "exchange_error": exchange_error,
                "momentum_error": momentum_error,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按平移动量分块求 J1-J2 XXZ 环各 k 扇区的最低能量。"
    )
    parser.add_argument("--N", type=int, default=16, help="格点数（默认：16）")
    parser.add_argument("--J1", type=float, default=1.0, help="最近邻耦合（默认：1）")
    parser.add_argument(
        "--J2",
        type=float,
        nargs="+",
        default=[0.0, 0.5],
        help="一个或多个第二近邻耦合（默认：0 0.5）",
    )
    parser.add_argument("--delta", type=float, default=1.0, help="XXZ 参数（默认：1）")
    parser.add_argument(
        "--solver-tol", type=float, default=1e-11, help="ARPACK 容差（默认：1e-11）"
    )
    parser.add_argument(
        "--ground-tol", type=float, default=1e-8, help="判定基态动量的绝对容差"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.N < 2 or args.N % 2 != 0:
        raise ValueError("S^z_tot=0 要求 N 是不小于 2 的偶数。")

    basis, index = build_sector_basis(args.N, args.N // 2)
    expected_dimension = math.comb(args.N, args.N // 2)
    sectors = build_momentum_sectors(args.N, basis, index)
    translation = build_translation_matrix(args.N, basis, index)

    orbit_dimensions = np.asarray([sector.dimension for sector in sectors])
    independent_dimensions = character_dimensions(args.N, basis)
    if not np.array_equal(orbit_dimensions, independent_dimensions):
        raise AssertionError(
            "轨道相容条件与独立的平移特征标公式给出了不同的块维数。"
        )
    if int(np.sum(orbit_dimensions)) != expected_dimension:
        raise AssertionError("所有动量块的维数之和不等于 S^z_tot=0 扇区维数。")

    translation_errors = []
    orthonormality_errors = []
    for sector in sectors:
        phase = np.exp(1j * sector.momentum)
        translation_errors.append(
            sparse_max_abs(translation @ sector.projector - phase * sector.projector)
        )
        gram = sector.projector.getH() @ sector.projector
        gram.setdiag(gram.diagonal() - 1.0)
        gram.eliminate_zeros()
        orthonormality_errors.append(sparse_max_abs(gram))
    max_translation_error = max(translation_errors)
    max_orthonormality_error = max(orthonormality_errors)
    if max(max_translation_error, max_orthonormality_error) > 1e-10:
        raise AssertionError("动量投影基未满足 T 本征方程或正交归一条件。")

    print("必做 3：J1-J2 XXZ 环的动量分块")
    print(
        f"参数：N={args.N}, J1={args.J1:g}, Delta={args.delta:g}, "
        f"S^z_tot=0 维数={len(basis)}"
    )
    print("约定：k=2*pi*m/N，T|psi_k>=exp(+i*k)|psi_k>。")

    all_results: list[
        tuple[float, np.ndarray, np.ndarray, float, float, float]
    ] = []
    for j2 in args.J2:
        hamiltonian = build_xxz_hamiltonian(
            args.N, args.J1, j2, args.delta, basis, index
        )
        commutator_error = sparse_max_abs(
            hamiltonian @ translation - translation @ hamiltonian
        )
        if commutator_error > 1e-12:
            raise AssertionError(f"J2={j2:g} 时 [H,T] 不为零。")

        energies: list[float] = []
        residuals: list[float] = []
        block_hermiticity_errors: list[float] = []
        for sector in sectors:
            block = project_hamiltonian(hamiltonian, sector)
            hermiticity_error = sparse_max_abs(block - block.getH())
            energy, block_vector = lowest_block_eigenpair(block, args.solver_tol)
            lifted = sector.projector @ block_vector
            residual = float(
                np.linalg.norm(hamiltonian @ lifted - energy * lifted)
            )
            energies.append(energy)
            residuals.append(residual)
            block_hermiticity_errors.append(hermiticity_error)

        energy_array = np.asarray(energies)
        residual_array = np.asarray(residuals)
        full_ground_energy = float(
            eigsh(
                hamiltonian,
                k=1,
                which="SA",
                tol=args.solver_tol,
                maxiter=200_000,
                v0=np.linspace(1.0, 2.0, len(basis)),
            )[0][0]
        )
        block_ground_energy = float(np.min(energy_array))
        reconstruction_error = abs(block_ground_energy - full_ground_energy)
        time_reversal_error = max(
            abs(energy_array[m] - energy_array[(-m) % args.N])
            for m in range(args.N)
        )
        max_residual = float(np.max(residual_array))
        max_block_hermiticity = max(block_hermiticity_errors)
        if max(
            reconstruction_error,
            time_reversal_error,
            max_residual,
            max_block_hermiticity,
        ) > 1e-8:
            raise AssertionError(f"J2={j2:g} 的动量分块本征对校检失败。")

        if (
            np.isclose(j2, 0.5 * args.J1, atol=1e-14, rtol=0.0)
            and np.isclose(args.delta, 1.0, atol=1e-14, rtol=0.0)
        ):
            exact_mg_energy = -3.0 * args.N * args.J1 / 8.0
            if abs(block_ground_energy - exact_mg_energy) > 1e-9:
                raise AssertionError("目标系统的 MG 解析能量校检失败。")

        ground_momenta = np.flatnonzero(
            np.abs(energy_array - block_ground_energy) <= args.ground_tol
        )
        print("\n" + "=" * 86)
        print(f"J2 = {j2:g}")
        print("=" * 86)
        print(f"{'m':>3} {'k/pi':>9} {'dim(k)':>9} {'E0(k)':>20} {'lifted residual':>19}")
        for sector, energy, residual in zip(sectors, energy_array, residual_array):
            marker = " *" if sector.momentum_index in ground_momenta else ""
            print(
                f"{sector.momentum_index:3d} "
                f"{sector.momentum / np.pi:9.5f} "
                f"{sector.dimension:9d} "
                f"{energy:20.12f} "
                f"{residual:19.3e}{marker}"
            )
        ground_text = ", ".join(
            f"m={m} (k={momentum_label(int(m), args.N)})"
            for m in ground_momenta
        )
        print(f"* 基态动量：{ground_text}")
        print(
            f"维数和：sum_k dim(k)={int(np.sum(orbit_dimensions))} "
            f"= C({args.N},{args.N // 2})={expected_dimension}"
        )
        all_results.append(
            (
                j2,
                energy_array,
                residual_array,
                commutator_error,
                reconstruction_error,
                time_reversal_error,
            )
        )

    mg_records = mg_validation_suite()

    print("\n校检清单：")
    print(
        "1. 动量块维数（针对轨道周期/相容条件或短轨道重复计数错误）：\n"
        f"   期望由独立特征标公式给出 {independent_dimensions.tolist()}；\n"
        f"   轨道投影实测 {orbit_dimensions.tolist()}，总和 "
        f"{int(np.sum(orbit_dimensions))}，通过。"
    )
    print(
        "2. 动量基（针对相位符号、归一化和平移方向错误）：\n"
        f"   max||T P_k-exp(i*k)P_k||_max={max_translation_error:.3e}，"
        f"max||P_k^dagger P_k-I||_max={max_orthonormality_error:.3e}，通过。"
    )
    for (
        j2,
        _,
        residuals,
        commutator_error,
        reconstruction_error,
        time_reversal_error,
    ) in all_results:
        print(
            f"3. J2={j2:g} 分块本征值（针对漏块、投影矩阵元或本征求解错误）：\n"
            f"   ||[H,T]||_max={commutator_error:.3e}；"
            f"min_k E0(k) 与未分块 U(1) Lanczos 的差="
            f"{reconstruction_error:.3e}；\n"
            f"   时间反演对称性 max|E0(k)-E0(-k)|={time_reversal_error:.3e}；"
            f"   提升回 U(1) 空间后的最大 ||Hv-Ev||={np.max(residuals):.3e}，通过。"
        )
    print(
        "4. Majumdar-Ghosh 解析解（针对 J2 键、Delta 项、符号或相对系数错误）："
    )
    for record in mg_records:
        print(
            f"   N={int(record['N'])}: E0={record['energy']:.12f}, "
            f"|E0+3N/8|={record['energy_error']:.3e}, "
            f"max 二聚体残差={record['dimer_residual']:.3e}, "
            f"T 交换误差={record['exchange_error']:.3e}, "
            f"k=0/pi 组合误差={record['momentum_error']:.3e}，通过。"
        )
    print(
        "   D0、D1 本身不是平移动量本征态；T 将二者互换。"
        "归一化的 D0+D1 与 D0-D1 分别属于 k=0 与 k=pi。"
    )


if __name__ == "__main__":
    main()
