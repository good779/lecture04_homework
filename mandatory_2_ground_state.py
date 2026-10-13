#!/usr/bin/env python3
"""必做 2：在 S^z_tot=0 扇区用稀疏迭代法求 J1-J2 XXZ 环基态。

默认参数对应题目：N=16, J1=1, J2=0/0.5, Delta=1。
矩阵元中的 N、J1、J2、Delta 均来自函数参数，而不是硬编码常数。
"""

from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class GroundStateResult:
    j2: float
    energy: float
    energy_per_site: float
    degeneracy: int
    computed_eigenvalues: np.ndarray
    max_ground_residual: float
    hermiticity_error: float


def build_sector_basis(n_sites: int, n_up: int) -> tuple[np.ndarray, dict[int, int]]:
    """构造固定 n_up 的整数比特基，并返回 state -> row 查找表。"""
    if not 0 <= n_up <= n_sites:
        raise ValueError("n_up 必须满足 0 <= n_up <= N。")

    states: list[int] = []
    for up_sites in itertools.combinations(range(n_sites), n_up):
        state = 0
        for site in up_sites:
            state |= 1 << site
        states.append(state)

    basis = np.asarray(states, dtype=np.int64)
    index = {int(state): row for row, state in enumerate(basis)}
    return basis, index


def interaction_bonds(n_sites: int, j1: float, j2: float) -> list[tuple[int, int, float]]:
    """生成带耦合常数的有向求和项；N=2 的最近邻键按题意只计一次。"""
    if n_sites < 2:
        raise ValueError("本模型要求 N >= 2。")

    bonds: list[tuple[int, int, float]] = []
    if j1 != 0.0:
        if n_sites == 2:
            bonds.append((0, 1, j1))
        else:
            bonds.extend((site, (site + 1) % n_sites, j1) for site in range(n_sites))

    if j2 != 0.0:
        # N=2 时 j+2=j，会成为非物理的自耦合，故跳过 i==j 的项。
        bonds.extend(
            (site, (site + 2) % n_sites, j2)
            for site in range(n_sites)
            if site != (site + 2) % n_sites
        )
    return bonds


def build_xxz_hamiltonian(
    n_sites: int,
    j1: float,
    j2: float,
    delta: float,
    basis: np.ndarray,
    index: dict[int, int],
) -> csr_matrix:
    """在给定固定磁化基上构造 J1-J2 XXZ 稀疏 Hamiltonian。"""
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    bonds = interaction_bonds(n_sites, j1, j2)

    for row, raw_state in enumerate(basis):
        state = int(raw_state)
        diagonal = 0.0
        for site_i, site_j, coupling in bonds:
            bit_i = (state >> site_i) & 1
            bit_j = (state >> site_j) & 1

            # S_i^z S_j^z：平行自旋为 +1/4，反平行为 -1/4。
            diagonal += coupling * delta * (0.25 if bit_i == bit_j else -0.25)

            # S_i^x S_j^x + S_i^y S_j^y
            # = (S_i^+ S_j^- + S_i^- S_j^+)/2，只翻转反平行自旋。
            if bit_i != bit_j:
                flipped = state ^ (1 << site_i) ^ (1 << site_j)
                rows.append(row)
                cols.append(index[flipped])
                values.append(0.5 * coupling)

        rows.append(row)
        cols.append(row)
        values.append(diagonal)

    dimension = len(basis)
    return coo_matrix(
        (values, (rows, cols)), shape=(dimension, dimension), dtype=np.float64
    ).tocsr()


def lowest_eigenpairs(
    hamiltonian: csr_matrix, num_eigenvalues: int, solver_tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """小矩阵用稠密法，仅目标大矩阵使用 ARPACK Lanczos 迭代法。"""
    dimension = hamiltonian.shape[0]
    if dimension <= max(32, num_eigenvalues + 1):
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.toarray())
        order = np.argsort(eigenvalues)
        keep = order[: min(num_eigenvalues, dimension)]
        return eigenvalues[keep], eigenvectors[:, keep]

    k = min(num_eigenvalues, dimension - 2)
    # 固定初始向量使结果可复现；这不包含待求本征态的信息。
    initial = np.linspace(1.0, 2.0, dimension, dtype=np.float64)
    initial /= np.linalg.norm(initial)
    eigenvalues, eigenvectors = eigsh(
        hamiltonian,
        k=k,
        which="SA",
        tol=solver_tolerance,
        maxiter=200_000,
        ncv=min(dimension, max(2 * k + 1, 40)),
        v0=initial,
    )
    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


def max_hermiticity_error(hamiltonian: csr_matrix) -> float:
    difference = hamiltonian - hamiltonian.getH()
    return float(np.max(np.abs(difference.data))) if difference.nnz else 0.0


def product_of_singlets(
    n_sites: int,
    pairs: list[tuple[int, int]],
    index: dict[int, int],
) -> np.ndarray:
    """构造 pairs 中各键上的 singlet 直积态。"""
    amplitudes: dict[int, float] = {0: 1.0}
    for site_a, site_b in pairs:
        updated: dict[int, float] = {}
        for state, amplitude in amplitudes.items():
            updated[state | (1 << site_a)] = updated.get(state | (1 << site_a), 0.0) + (
                amplitude / math.sqrt(2.0)
            )
            updated[state | (1 << site_b)] = updated.get(state | (1 << site_b), 0.0) - (
                amplitude / math.sqrt(2.0)
            )
        amplitudes = updated

    vector = np.zeros(len(index), dtype=np.float64)
    for state, amplitude in amplitudes.items():
        vector[index[state]] = amplitude
    return vector


def check_n2_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """解析校检：N=2 Heisenberg 二聚体谱。"""
    measured: list[float] = []
    for n_up in range(3):
        basis, index = build_sector_basis(2, n_up)
        hamiltonian = build_xxz_hamiltonian(2, 1.0, 0.0, 1.0, basis, index)
        measured.extend(np.linalg.eigvalsh(hamiltonian.toarray()))
    actual = np.sort(np.asarray(measured))
    expected = np.asarray([-0.75, 0.25, 0.25, 0.25])
    if not np.allclose(actual, expected, atol=1e-13, rtol=0.0):
        raise AssertionError(f"N=2 谱校检失败：实测 {actual}，期望 {expected}")
    return actual, expected


def check_n4_ground_energy() -> tuple[float, float]:
    """解析校检：N=4 Heisenberg 环的 E0=-2（取 J1=1）。"""
    basis, index = build_sector_basis(4, 2)
    hamiltonian = build_xxz_hamiltonian(4, 1.0, 0.0, 1.0, basis, index)
    actual = float(np.linalg.eigvalsh(hamiltonian.toarray())[0])
    expected = -2.0
    if not np.isclose(actual, expected, atol=1e-13, rtol=0.0):
        raise AssertionError(f"N=4 基态能校检失败：实测 {actual}，期望 {expected}")
    return actual, expected


def check_mg_dimer_states(
    n_sites: int,
    j1: float,
    hamiltonian: csr_matrix,
    index: dict[int, int],
) -> tuple[float, float, int]:
    """在 MG 点检查两个二聚体覆盖态的本征方程与线性独立性。"""
    pairs_0 = [(site, site + 1) for site in range(0, n_sites, 2)]
    pairs_1 = [((site + 1) % n_sites, (site + 2) % n_sites) for site in range(0, n_sites, 2)]
    dimer_0 = product_of_singlets(n_sites, pairs_0, index)
    dimer_1 = product_of_singlets(n_sites, pairs_1, index)
    exact_energy = -3.0 * n_sites * j1 / 8.0
    residual_0 = float(np.linalg.norm(hamiltonian @ dimer_0 - exact_energy * dimer_0))
    residual_1 = float(np.linalg.norm(hamiltonian @ dimer_1 - exact_energy * dimer_1))
    gram = np.asarray(
        [
            [np.vdot(dimer_0, dimer_0), np.vdot(dimer_0, dimer_1)],
            [np.vdot(dimer_1, dimer_0), np.vdot(dimer_1, dimer_1)],
        ]
    )
    rank = int(np.linalg.matrix_rank(gram, tol=1e-12))
    if max(residual_0, residual_1) > 1e-10 or rank != 2:
        raise AssertionError(
            "MG 二聚体校检失败："
            f"residuals=({residual_0:.3e}, {residual_1:.3e}), Gram rank={rank}"
        )
    return residual_0, residual_1, rank


def solve_one_coupling(
    n_sites: int,
    j1: float,
    j2: float,
    delta: float,
    basis: np.ndarray,
    index: dict[int, int],
    num_eigenvalues: int,
    solver_tolerance: float,
    degeneracy_tolerance: float,
) -> tuple[GroundStateResult, csr_matrix]:
    hamiltonian = build_xxz_hamiltonian(n_sites, j1, j2, delta, basis, index)
    hermiticity_error = max_hermiticity_error(hamiltonian)
    if hermiticity_error > 1e-13:
        raise AssertionError(f"Hermitian 校检失败：max|H-H^dagger|={hermiticity_error:.3e}")

    eigenvalues, eigenvectors = lowest_eigenpairs(
        hamiltonian, num_eigenvalues, solver_tolerance
    )
    ground_energy = float(eigenvalues[0])
    ground_mask = np.abs(eigenvalues - ground_energy) <= degeneracy_tolerance
    degeneracy = int(np.count_nonzero(ground_mask))
    if degeneracy == len(eigenvalues):
        raise RuntimeError("求得的本征值数量不足以确定基态简并度，请增大 --num-eigs。")

    residuals = [
        np.linalg.norm(hamiltonian @ eigenvectors[:, column] - eigenvalues[column] * eigenvectors[:, column])
        for column in np.flatnonzero(ground_mask)
    ]
    max_residual = float(max(residuals))
    if max_residual > 1e-8:
        raise AssertionError(f"Lanczos 本征对残差过大：{max_residual:.3e}")

    result = GroundStateResult(
        j2=j2,
        energy=ground_energy,
        energy_per_site=ground_energy / n_sites,
        degeneracy=degeneracy,
        computed_eigenvalues=eigenvalues,
        max_ground_residual=max_residual,
        hermiticity_error=hermiticity_error,
    )
    return result, hamiltonian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 S^z_tot=0 扇区用 Lanczos/ARPACK 求 J1-J2 XXZ 环基态。"
    )
    parser.add_argument("--N", type=int, default=16, help="格点数（默认：16）")
    parser.add_argument("--J1", type=float, default=1.0, help="最近邻耦合（默认：1.0）")
    parser.add_argument(
        "--J2",
        type=float,
        nargs="+",
        default=[0.0, 0.5],
        help="一个或多个第二近邻耦合（默认：0.0 0.5）",
    )
    parser.add_argument("--delta", type=float, default=1.0, help="XXZ 各向异性（默认：1.0）")
    parser.add_argument(
        "--num-eigs", type=int, default=8, help="用于判断简并度的最低本征值数量（默认：8）"
    )
    parser.add_argument(
        "--solver-tol", type=float, default=1e-11, help="迭代本征求解器容差（默认：1e-11）"
    )
    parser.add_argument(
        "--degeneracy-tol", type=float, default=1e-8, help="判定能级简并的绝对容差（默认：1e-8）"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.N < 2 or args.N % 2 != 0:
        raise ValueError("S^z_tot=0 要求 N 是不小于 2 的偶数。")
    if args.num_eigs < 3:
        raise ValueError("--num-eigs 至少为 3，才能可靠地区分简并基态与第一激发态。")

    n_up = args.N // 2
    basis, index = build_sector_basis(args.N, n_up)
    expected_dimension = math.comb(args.N, n_up)
    if len(basis) != expected_dimension or len(index) != expected_dimension:
        raise AssertionError("固定磁化基维数或查找表有重复/遗漏。")

    print("J1-J2 XXZ 环：S^z_tot=0 扇区基态")
    print(f"参数：N={args.N}, J1={args.J1:g}, Delta={args.delta:g}")
    print(f"扇区维数：C({args.N}, {n_up}) = {len(basis)}")
    print("求解方法：scipy.sparse.linalg.eigsh (ARPACK Lanczos, which='SA')")

    # 先运行不依赖目标大系统的解析校检。
    n2_actual, n2_expected = check_n2_spectrum()
    n4_actual, n4_expected = check_n4_ground_energy()

    results: list[GroundStateResult] = []
    mg_checks: list[tuple[float, float, int] | None] = []
    for j2 in args.J2:
        result, hamiltonian = solve_one_coupling(
            args.N,
            args.J1,
            j2,
            args.delta,
            basis,
            index,
            args.num_eigs,
            args.solver_tol,
            args.degeneracy_tol,
        )
        results.append(result)

        is_mg_point = (
            args.N >= 6
            and args.J1 > 0.0
            and np.isclose(j2, 0.5 * args.J1, atol=1e-14, rtol=0.0)
            and np.isclose(args.delta, 1.0, atol=1e-14, rtol=0.0)
        )
        if is_mg_point:
            exact_energy = -3.0 * args.N * args.J1 / 8.0
            if not np.isclose(result.energy, exact_energy, atol=1e-9, rtol=0.0):
                raise AssertionError(
                    f"MG 解析能量校检失败：实测 {result.energy:.12f}，期望 {exact_energy:.12f}"
                )
            if result.degeneracy != 2:
                raise AssertionError(
                    f"MG 基态简并度校检失败：实测 {result.degeneracy}，期望 2"
                )
            mg_checks.append(check_mg_dimer_states(args.N, args.J1, hamiltonian, index))
        else:
            mg_checks.append(None)

        is_unfrustrated_heisenberg = (
            args.N >= 4
            and args.J1 > 0.0
            and np.isclose(j2, 0.0, atol=1e-14, rtol=0.0)
            and np.isclose(args.delta, 1.0, atol=1e-14, rtol=0.0)
        )
        if is_unfrustrated_heisenberg and result.degeneracy != 1:
            raise AssertionError(
                "无阻挫偶数反铁磁 Heisenberg 环应有唯一基态，"
                f"但数值简并度为 {result.degeneracy}。"
            )

    print("\n数值结果：")
    print(f"{'J2':>8} {'E0':>18} {'E0/N':>18} {'degeneracy':>12}")
    for result in results:
        print(
            f"{result.j2:8g} {result.energy:18.12f} "
            f"{result.energy_per_site:18.12f} {result.degeneracy:12d}"
        )
        spectrum_text = ", ".join(f"{value:.10f}" for value in result.computed_eigenvalues)
        print(f"         用于判定简并度的最低本征值：[{spectrum_text}]")

    print("\n校检清单：")
    print(
        "1. N=2 Heisenberg 谱（检查周期键去重及自旋翻转系数）：\n"
        f"   期望 {n2_expected.tolist()}，实测 {n2_actual.tolist()}，通过。\n"
        "   若把 0-1 键计数两次或 XY 项系数写错，四个能级将不匹配。"
    )
    print(
        "2. N=4 Heisenberg 环基态能（检查 J1 的对角与非对角矩阵元）：\n"
        f"   期望 {n4_expected:.12f}，实测 {n4_actual:.12f}，通过。\n"
        "   若漏键、符号错误或去掉 Delta，E0 将偏离 -2。"
    )
    for result, mg_check in zip(results, mg_checks):
        print(
            f"3. J2={result.j2:g} 目标矩阵 Hermitian 性与 Lanczos 残差：\n"
            f"   max|H-H^dagger|={result.hermiticity_error:.3e}, "
            f"max||Hv-Ev||={result.max_ground_residual:.3e}，通过。\n"
            "   前者检查非对角共轭项是否成对；后者检查迭代本征对是否收敛。"
        )
        if mg_check is not None:
            residual_0, residual_1, rank = mg_check
            exact_energy = -3.0 * args.N * args.J1 / 8.0
            print(
                "4. Majumdar-Ghosh 解析校检（检查 J2 键、符号和相对系数）：\n"
                f"   期望 E0=-3*N*J1/8={exact_energy:.12f}、简并度 2；"
                f"实测 E0={result.energy:.12f}、简并度 {result.degeneracy}。\n"
                f"   两个独立二聚体态残差：{residual_0:.3e}, {residual_1:.3e}；"
                f"Gram 秩={rank}，通过。\n"
                "   若漏掉 J2 键、J2 系数或 Delta 项错误，解析能量或二聚体残差会失败。"
            )
        elif (
            args.N >= 4
            and args.J1 > 0.0
            and np.isclose(result.j2, 0.0, atol=1e-14, rtol=0.0)
            and np.isclose(args.delta, 1.0, atol=1e-14, rtol=0.0)
        ):
            print(
                "4. 简并度理论校检：偶数、无阻挫、反铁磁 Heisenberg 环的基态唯一；\n"
                f"   期望简并度 1，实测 {result.degeneracy}，通过。"
            )


if __name__ == "__main__":
    main()
