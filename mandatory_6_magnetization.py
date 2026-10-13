#!/usr/bin/env python3
"""必做 6：局域磁化、各 S^z_tot 扇区最低能量及 m(h) 曲线。

磁化曲线采用每格点磁化 m(h)=S^z_tot(h)/N。对每个磁场 h，程序直接
比较所有扇区的 E0(S^z_tot)-h*S^z_tot，并由这些直线的下包络给出临界场。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

from mandatory_2_ground_state import (
    build_sector_basis,
    build_xxz_hamiltonian,
    lowest_eigenpairs,
    max_hermiticity_error,
)


@dataclass(frozen=True)
class LocalMagnetization:
    energy: float
    sz: np.ndarray
    sx: np.ndarray
    eigen_residual: float
    singlet_residual: float


@dataclass(frozen=True)
class MagnetizationResult:
    j2: float
    magnetizations: np.ndarray
    sector_energies: np.ndarray
    local: LocalMagnetization
    hull_indices: tuple[int, ...]
    critical_fields: np.ndarray
    max_hermiticity_error: float
    spin_inversion_error: float
    polarized_energy_error: float
    envelope_check_passed: bool


def sector_ground_energy(hamiltonian, solver_tolerance: float) -> float:
    dimension = hamiltonian.shape[0]
    if dimension <= 4:
        return float(np.linalg.eigvalsh(hamiltonian.toarray())[0])
    initial = np.linspace(1.0, 2.0, dimension)
    initial /= np.linalg.norm(initial)
    return float(
        eigsh(
            hamiltonian,
            k=1,
            which="SA",
            tol=solver_tolerance,
            maxiter=200_000,
            ncv=min(dimension, 30),
            v0=initial,
        )[0][0]
    )


def local_magnetization(
    n_sites: int,
    hamiltonian,
    basis: np.ndarray,
    index: dict[int, int],
    solver_tolerance: float,
) -> LocalMagnetization:
    """Evaluate site-resolved Sz and Sx in one normalized true ground state."""
    eigenvalues, eigenvectors = lowest_eigenpairs(
        hamiltonian, num_eigenvalues=4, solver_tolerance=solver_tolerance
    )
    energy = float(eigenvalues[0])
    state_vector = eigenvectors[:, 0].astype(np.complex128)
    state_vector /= np.linalg.norm(state_vector)
    eigen_residual = float(
        np.linalg.norm(hamiltonian @ state_vector - energy * state_vector)
    )

    probabilities = np.abs(state_vector) ** 2
    sz = np.empty(n_sites, dtype=np.float64)
    for site in range(n_sites):
        local_sz = np.asarray(
            [0.5 if (int(state) >> site) & 1 else -0.5 for state in basis]
        )
        sz[site] = float(np.dot(probabilities, local_sz))

    # Embed into the full computational basis and apply S_i^x by a bit flip.
    full_vector = np.zeros(1 << n_sites, dtype=np.complex128)
    full_vector[basis] = state_vector
    full_indices = np.arange(1 << n_sites, dtype=np.int64)
    sx = np.asarray(
        [
            float(
                np.real(
                    0.5
                    * np.vdot(
                        full_vector,
                        full_vector[full_indices ^ (1 << site)],
                    )
                )
            )
            for site in range(n_sites)
        ]
    )

    # In the M=0 sector, ||S_tot^+|psi>||^2=<S_tot^2>.  It must vanish for
    # the SU(2) singlet ground state and is a nontrivial selected-state check.
    raised_basis, raised_index = build_sector_basis(n_sites, n_sites // 2 + 1)
    raised = np.zeros(len(raised_basis), dtype=np.complex128)
    for column, raw_state in enumerate(basis):
        state = int(raw_state)
        for site in range(n_sites):
            if ((state >> site) & 1) == 0:
                raised[raised_index[state | (1 << site)]] += state_vector[column]
    singlet_residual = float(np.linalg.norm(raised))

    if max(
        eigen_residual,
        singlet_residual,
        float(np.max(np.abs(sz))),
        float(np.max(np.abs(sx))),
    ) > 1e-8:
        raise AssertionError("真实基态的本征方程、单态或逐点磁化校检失败。")
    return LocalMagnetization(
        energy=energy,
        sz=sz,
        sx=sx,
        eigen_residual=eigen_residual,
        singlet_residual=singlet_residual,
    )


def lower_convex_hull(
    magnetizations: np.ndarray, energies: np.ndarray, tolerance: float = 1e-12
) -> tuple[tuple[int, ...], np.ndarray]:
    """Indices of the lower convex hull and its consecutive line slopes."""
    hull: list[int] = []
    for new_index in range(len(magnetizations)):
        while len(hull) >= 2:
            left, middle = hull[-2], hull[-1]
            old_slope = (energies[middle] - energies[left]) / (
                magnetizations[middle] - magnetizations[left]
            )
            new_slope = (energies[new_index] - energies[middle]) / (
                magnetizations[new_index] - magnetizations[middle]
            )
            if old_slope < new_slope - tolerance:
                break
            hull.pop()
        hull.append(new_index)
    critical_fields = np.asarray(
        [
            (energies[right] - energies[left])
            / (magnetizations[right] - magnetizations[left])
            for left, right in zip(hull[:-1], hull[1:])
        ],
        dtype=np.float64,
    )
    return tuple(hull), critical_fields


def solve_one_coupling(
    n_sites: int,
    j1: float,
    j2: float,
    delta: float,
    solver_tolerance: float,
) -> MagnetizationResult:
    magnetizations = np.arange(n_sites + 1, dtype=np.float64) - n_sites / 2.0
    energies = np.empty(n_sites + 1, dtype=np.float64)
    hermiticity_errors: list[float] = []
    zero_basis = None
    zero_index = None
    zero_hamiltonian = None

    for n_up in range(n_sites + 1):
        basis, index = build_sector_basis(n_sites, n_up)
        hamiltonian = build_xxz_hamiltonian(
            n_sites, j1, j2, delta, basis, index
        )
        energies[n_up] = sector_ground_energy(hamiltonian, solver_tolerance)
        hermiticity_errors.append(max_hermiticity_error(hamiltonian))
        if n_up == n_sites // 2:
            zero_basis, zero_index, zero_hamiltonian = basis, index, hamiltonian

    assert zero_basis is not None and zero_index is not None
    assert zero_hamiltonian is not None
    local = local_magnetization(
        n_sites,
        zero_hamiltonian,
        zero_basis,
        zero_index,
        solver_tolerance,
    )
    hull_indices, critical_fields = lower_convex_hull(magnetizations, energies)

    if len(critical_fields):
        probe_fields = np.concatenate(
            (
                [critical_fields[0] - 1.0],
                0.5 * (critical_fields[:-1] + critical_fields[1:]),
                [critical_fields[-1] + 1.0],
            )
        )
    else:
        probe_fields = np.asarray([0.0])
    selected_at_probes = [
        int(np.argmin(energies - field * magnetizations))
        for field in probe_fields
    ]
    envelope_check_passed = selected_at_probes == list(hull_indices)
    if not envelope_check_passed:
        raise AssertionError("凸包临界场与逐扇区 Zeeman 能量直接比较不一致。")

    spin_inversion_error = float(np.max(np.abs(energies - energies[::-1])))
    nearest_count = 1 if n_sites == 2 else n_sites
    next_nearest_count = n_sites if n_sites > 2 else 0
    expected_polarized = (
        delta * 0.25 * (nearest_count * j1 + next_nearest_count * j2)
    )
    polarized_energy_error = max(
        abs(energies[0] - expected_polarized),
        abs(energies[-1] - expected_polarized),
    )
    maximum_hermiticity = max(hermiticity_errors)
    if max(
        spin_inversion_error, polarized_energy_error, maximum_hermiticity
    ) > 1e-9:
        raise AssertionError(f"J2={j2:g} 的扇区能量校检失败。")

    if (
        np.isclose(j2, 0.5 * j1, atol=1e-14, rtol=0.0)
        and np.isclose(delta, 1.0, atol=1e-14, rtol=0.0)
    ):
        expected_mg = -3.0 * n_sites * j1 / 8.0
        if abs(energies[n_sites // 2] - expected_mg) > 1e-9:
            raise AssertionError("MG 点 S^z_tot=0 基态能量校检失败。")

    return MagnetizationResult(
        j2=j2,
        magnetizations=magnetizations,
        sector_energies=energies,
        local=local,
        hull_indices=hull_indices,
        critical_fields=critical_fields,
        max_hermiticity_error=maximum_hermiticity,
        spin_inversion_error=spin_inversion_error,
        polarized_energy_error=float(polarized_energy_error),
        envelope_check_passed=envelope_check_passed,
    )


def magnetization_on_grid(
    result: MagnetizationResult, fields: np.ndarray, n_sites: int
) -> np.ndarray:
    zeeman_energies = (
        result.sector_energies[:, np.newaxis]
        - result.magnetizations[:, np.newaxis] * fields[np.newaxis, :]
    )
    minimizing_indices = np.argmin(zeeman_energies, axis=0)
    return result.magnetizations[minimizing_indices] / n_sites


def plot_magnetization(
    n_sites: int,
    results: list[MagnetizationResult],
    output_path: Path,
    requested_h_min: float | None,
    requested_h_max: float | None,
) -> tuple[float, float]:
    all_critical = np.concatenate([result.critical_fields for result in results])
    span = float(np.max(all_critical) - np.min(all_critical))
    margin = max(0.25, 0.08 * span)
    h_min = (
        float(requested_h_min)
        if requested_h_min is not None
        else float(np.min(all_critical) - margin)
    )
    h_max = (
        float(requested_h_max)
        if requested_h_max is not None
        else float(np.max(all_critical) + margin)
    )
    if h_min >= h_max:
        raise ValueError("--h-min 必须小于 --h-max。")
    fields = np.linspace(h_min, h_max, 4001)

    figure, axes = plt.subplots(
        1, len(results), figsize=(6.2 * len(results), 4.8), squeeze=False
    )
    for axis, result in zip(axes[0], results):
        magnetization = magnetization_on_grid(result, fields, n_sites)
        axis.step(fields, magnetization, where="post", color="#276FBF", linewidth=1.8)
        for critical_field in result.critical_fields:
            if h_min <= critical_field <= h_max:
                axis.axvline(critical_field, color="0.75", linewidth=0.6, zorder=0)
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.axvline(0.0, color="black", linewidth=0.6)
        axis.set_title(rf"$J_2={result.j2:g}$")
        axis.set_xlabel(r"field $h$")
        axis.set_ylabel(r"magnetization per site $m=S^z_{tot}/N$")
        axis.set_xlim(h_min, h_max)
        axis.set_ylim(-0.54, 0.54)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(rf"Zero-temperature magnetization curve, $N={n_sites}$")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return h_min, h_max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算 J1-J2 XXZ 环局域磁化与零温 m(h) 曲线。"
    )
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--solver-tol", type=float, default=1e-11)
    parser.add_argument("--h-min", type=float, default=None)
    parser.add_argument("--h-max", type=float, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("mandatory_6_magnetization_curve.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.N < 2 or args.N % 2 != 0:
        raise ValueError("本作业的 S^z_tot=0 真实基态要求 N 为不小于 2 的偶数。")

    results = [
        solve_one_coupling(
            args.N,
            args.J1,
            j2,
            args.delta,
            args.solver_tol,
        )
        for j2 in args.J2
    ]
    h_min, h_max = plot_magnetization(
        args.N, results, args.output, args.h_min, args.h_max
    )

    print("必做 6：局域磁化与磁化曲线")
    print(f"参数：N={args.N}, J1={args.J1:g}, Delta={args.delta:g}")
    for result in results:
        print("\n" + "=" * 74)
        print(f"J2={result.j2:g}：真实基态逐点磁化")
        print("=" * 74)
        print(f"{'site i':>6} {'<S_i^z>':>20} {'<S_i^x>':>20}")
        for site, (sz_value, sx_value) in enumerate(zip(result.local.sz, result.local.sx)):
            print(f"{site:6d} {sz_value:20.12e} {sx_value:20.12e}")
        print(
            f"sum_i <S_i^z>={np.sum(result.local.sz):.3e}, "
            f"max_i |<S_i^x>|={np.max(np.abs(result.local.sx)):.3e}"
        )

        print(f"\nJ2={result.j2:g}：各 S^z_tot 扇区最低能量")
        print(f"{'S^z_tot':>10} {'E0(S^z_tot)':>20}")
        for magnetization, energy in zip(
            result.magnetizations, result.sector_energies
        ):
            print(f"{magnetization:10g} {energy:20.12f}")

        print("\n磁化台阶临界场（下包络相邻扇区）：")
        for left, right, field in zip(
            result.hull_indices[:-1],
            result.hull_indices[1:],
            result.critical_fields,
        ):
            left_m = result.magnetizations[left]
            right_m = result.magnetizations[right]
            print(f"  S^z_tot: {left_m:g} -> {right_m:g}, h_c={field:.12f}")

        print("\n校检清单：")
        print(
            "  1. 真实基态本征残差="
            f"{result.local.eigen_residual:.3e}；"
            f"||S_tot^+|psi>||={result.local.singlet_residual:.3e}，通过。\n"
            "     针对未收敛或误选非单态；失败时残差或逐点磁化不再为零。"
        )
        print(
            "  2. 所有扇区 Hermitian 最大误差="
            f"{result.max_hermiticity_error:.3e}；"
            f"自旋翻转 max|E(M)-E(-M)|={result.spin_inversion_error:.3e}，通过。\n"
            "     针对非对称矩阵、磁化扇区或自旋比特约定错误。"
        )
        print(
            "  3. 全极化态解析能量误差="
            f"{result.polarized_energy_error:.3e}，通过。\n"
            "     针对键数、J1/J2 或 Delta 对角项系数错误。"
        )
        print(
            "  4. 每个相区中点直接最小化 E0(M)-hM，与凸包给出的磁化扇区"
            f"逐段一致={result.envelope_check_passed}，通过。\n"
            "     针对临界场排序、跳过扇区或磁化台阶构造错误。"
        )

    print(f"磁场绘图区间：[{h_min:.6f}, {h_max:.6f}]")
    print(f"图片已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
