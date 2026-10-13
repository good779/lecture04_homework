#!/usr/bin/env python3
"""必做 5：绘制各动量扇区最低能量 E(k)。

默认参数为 N=16、J1=1、Delta=1、J2=0/0.5。程序同时报告：

1. E(k) 曲线的基态动量；
2. 排除所有简并基态后，整个 S^z_tot=0 空间中真正的最低激发动量；
3. 仅从 E(k) 曲线排除基态点后得到的最低非基态扇区。

第 2、3 项通常相同；若最低激发与基态处于同一动量扇区，它们会不同。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mandatory_2_ground_state import (
    build_sector_basis,
    build_xxz_hamiltonian,
    lowest_eigenpairs,
)
from momentum_blocks import (
    build_momentum_sectors,
    lowest_block_eigenpairs,
    project_hamiltonian,
)


@dataclass(frozen=True)
class DispersionResult:
    j2: float
    sector_minima: np.ndarray
    ground_energy: float
    ground_momenta: tuple[int, ...]
    first_excited_energy: float
    first_excited_momenta: tuple[int, ...]
    curve_excited_energy: float
    curve_excited_momenta: tuple[int, ...]
    max_lifted_residual: float
    unblocked_ground_error: float
    unblocked_excited_error: float
    time_reversal_error: float


def momentum_label(momentum_index: int, n_sites: int) -> str:
    if momentum_index == 0:
        return "0"
    if 2 * momentum_index == n_sites:
        return "pi"
    return f"2*pi*{momentum_index}/{n_sites}"


def first_value_above_ground(
    values: np.ndarray, ground_energy: float, tolerance: float
) -> float:
    candidates = values[values > ground_energy + tolerance]
    if len(candidates) == 0:
        raise RuntimeError("本征值数量不足，无法识别第一激发态。")
    return float(np.min(candidates))


def solve_dispersion(
    n_sites: int,
    j1: float,
    j2: float,
    delta: float,
    basis: np.ndarray,
    index: dict[int, int],
    sectors,
    num_eigenvalues: int,
    solver_tolerance: float,
    degeneracy_tolerance: float,
) -> DispersionResult:
    hamiltonian = build_xxz_hamiltonian(
        n_sites, j1, j2, delta, basis, index
    )
    sector_spectra: list[np.ndarray] = []
    maximum_residual = 0.0

    for sector in sectors:
        eigenvalues, eigenvectors = lowest_block_eigenpairs(
            project_hamiltonian(hamiltonian, sector),
            num_eigenvalues=num_eigenvalues,
            solver_tolerance=solver_tolerance,
        )
        lifted = sector.projector @ eigenvectors
        residuals = np.linalg.norm(
            hamiltonian @ lifted - lifted * eigenvalues[np.newaxis, :], axis=0
        )
        maximum_residual = max(maximum_residual, float(np.max(residuals)))
        sector_spectra.append(eigenvalues)

    sector_minima = np.asarray([spectrum[0] for spectrum in sector_spectra])
    ground_energy = float(np.min(sector_minima))
    ground_momenta = tuple(
        int(m)
        for m in np.flatnonzero(
            np.abs(sector_minima - ground_energy) <= degeneracy_tolerance
        )
    )

    excited_candidates: list[tuple[float, int]] = []
    for momentum_index, spectrum in enumerate(sector_spectra):
        for energy in spectrum:
            if energy > ground_energy + degeneracy_tolerance:
                excited_candidates.append((float(energy), momentum_index))
    if not excited_candidates:
        raise RuntimeError("每个动量块需要计算更多本征值才能确定最低激发。")
    first_excited_energy = min(energy for energy, _ in excited_candidates)
    first_excited_momenta = tuple(
        sorted(
            {
                momentum_index
                for energy, momentum_index in excited_candidates
                if abs(energy - first_excited_energy) <= degeneracy_tolerance
            }
        )
    )

    nonground_curve_indices = [
        m for m in range(n_sites) if m not in ground_momenta
    ]
    curve_excited_energy = min(sector_minima[m] for m in nonground_curve_indices)
    curve_excited_momenta = tuple(
        m
        for m in nonground_curve_indices
        if abs(sector_minima[m] - curve_excited_energy) <= degeneracy_tolerance
    )

    unblocked_values, _ = lowest_eigenpairs(
        hamiltonian,
        num_eigenvalues=max(8, num_eigenvalues + len(ground_momenta)),
        solver_tolerance=solver_tolerance,
    )
    unblocked_ground = float(unblocked_values[0])
    unblocked_excited = first_value_above_ground(
        unblocked_values, unblocked_ground, degeneracy_tolerance
    )
    unblocked_ground_error = abs(unblocked_ground - ground_energy)
    unblocked_excited_error = abs(unblocked_excited - first_excited_energy)
    time_reversal_error = max(
        abs(sector_minima[m] - sector_minima[(-m) % n_sites])
        for m in range(n_sites)
    )

    if max(
        maximum_residual,
        unblocked_ground_error,
        unblocked_excited_error,
        time_reversal_error,
    ) > 1e-8:
        raise AssertionError(f"J2={j2:g} 的色散关系校检失败。")

    if (
        np.isclose(j2, 0.5 * j1, atol=1e-14, rtol=0.0)
        and np.isclose(delta, 1.0, atol=1e-14, rtol=0.0)
    ):
        expected = -3.0 * n_sites * j1 / 8.0
        if abs(ground_energy - expected) > 1e-9:
            raise AssertionError("Majumdar-Ghosh 解析基态能量校检失败。")

    return DispersionResult(
        j2=j2,
        sector_minima=sector_minima,
        ground_energy=ground_energy,
        ground_momenta=ground_momenta,
        first_excited_energy=first_excited_energy,
        first_excited_momenta=first_excited_momenta,
        curve_excited_energy=float(curve_excited_energy),
        curve_excited_momenta=curve_excited_momenta,
        max_lifted_residual=maximum_residual,
        unblocked_ground_error=unblocked_ground_error,
        unblocked_excited_error=unblocked_excited_error,
        time_reversal_error=float(time_reversal_error),
    )


def plot_dispersion(
    n_sites: int, results: list[DispersionResult], output_path: Path
) -> None:
    figure, axes = plt.subplots(
        1, len(results), figsize=(6.2 * len(results), 4.8), squeeze=False
    )
    x = 2.0 * np.pi * np.arange(n_sites) / n_sites
    for axis, result in zip(axes[0], results):
        closed_x = np.append(x, 2.0 * np.pi)
        closed_energies = np.append(result.sector_minima, result.sector_minima[0])
        axis.plot(
            closed_x,
            closed_energies,
            "o-",
            color="#276FBF",
            label=r"$E_0(k)$",
        )
        ground_x = x[list(result.ground_momenta)]
        axis.scatter(
            ground_x,
            np.full(len(ground_x), result.ground_energy),
            marker="*",
            s=150,
            color="#C1292E",
            zorder=4,
            label="ground state",
        )
        excited_x = x[list(result.first_excited_momenta)]
        axis.scatter(
            excited_x,
            np.full(len(excited_x), result.first_excited_energy),
            marker="D",
            s=55,
            color="#F28E2B",
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
            label="true lowest excitation",
        )
        if (
            abs(result.curve_excited_energy - result.first_excited_energy) > 1e-8
            or set(result.curve_excited_momenta)
            != set(result.first_excited_momenta)
        ):
            curve_excited_x = x[list(result.curve_excited_momenta)]
            axis.scatter(
                curve_excited_x,
                np.full(len(curve_excited_x), result.curve_excited_energy),
                marker="v",
                s=65,
                color="#59A14F",
                edgecolor="black",
                linewidth=0.5,
                zorder=4,
                label=r"lowest non-GS $E_0(k)$",
            )
        axis.set_title(rf"$J_2={result.j2:g}$")
        axis.set_xlabel(r"momentum $k=2\pi m/N$")
        axis.set_ylabel("energy")
        axis.set_xticks(
            [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi],
            [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
        )
        axis.set_xlim(-0.04 * np.pi, 2.04 * np.pi)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle(rf"Momentum-resolved spectrum, $N={n_sites}$")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 J1-J2 XXZ 环的 E(k)。")
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--num-eigs", type=int, default=4)
    parser.add_argument("--solver-tol", type=float, default=1e-11)
    parser.add_argument("--degeneracy-tol", type=float, default=1e-8)
    parser.add_argument(
        "--output", type=Path, default=Path("mandatory_5_dispersion.png")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.N < 2 or args.N % 2 != 0:
        raise ValueError("S^z_tot=0 要求 N 是不小于 2 的偶数。")
    if args.num_eigs < 2:
        raise ValueError("--num-eigs 至少为 2，才能识别最低激发。")

    basis, index = build_sector_basis(args.N, args.N // 2)
    sectors = build_momentum_sectors(args.N, basis, index)
    results = [
        solve_dispersion(
            args.N,
            args.J1,
            j2,
            args.delta,
            basis,
            index,
            sectors,
            args.num_eigs,
            args.solver_tol,
            args.degeneracy_tol,
        )
        for j2 in args.J2
    ]
    plot_dispersion(args.N, results, args.output)

    print("必做 5：动量分辨色散 E(k)")
    print(f"参数：N={args.N}, J1={args.J1:g}, Delta={args.delta:g}")
    for result in results:
        ground_text = ", ".join(
            momentum_label(m, args.N) for m in result.ground_momenta
        )
        excited_text = ", ".join(
            momentum_label(m, args.N) for m in result.first_excited_momenta
        )
        curve_text = ", ".join(
            momentum_label(m, args.N) for m in result.curve_excited_momenta
        )
        print(f"\nJ2={result.j2:g}")
        print(f"  基态：E0={result.ground_energy:.12f}, k={ground_text}")
        print(
            f"  真正最低激发：E1={result.first_excited_energy:.12f}, "
            f"gap={result.first_excited_energy-result.ground_energy:.12f}, "
            f"k={excited_text}"
        )
        print(
            f"  E(k) 曲线排除基态点后的最低点："
            f"E={result.curve_excited_energy:.12f}, k={curve_text}"
        )
        print(
            "  校检：max 提升残差="
            f"{result.max_lifted_residual:.3e}, "
            f"未分块 E0/E1 误差={result.unblocked_ground_error:.3e}/"
            f"{result.unblocked_excited_error:.3e}, "
            f"max|E(k)-E(-k)|={result.time_reversal_error:.3e}，通过。"
        )
    print(f"\n图片已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
