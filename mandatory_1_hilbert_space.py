#!/usr/bin/env python3
"""必做 1：统计自旋 1/2 链各个总磁化扇区的维数。

基矢约定与题目一致：整数 s 的第 j 个比特表示格点 j，1/0 分别表示
自旋向上/向下。脚本默认计算 N=16，也可通过 --N 修改。
"""

from __future__ import annotations

import argparse
import math


def dimensions_from_binomial(n_sites: int) -> dict[int, int]:
    """由组合数 C(N, n_up) 得到每个 n_up 扇区的维数。"""
    return {n_up: math.comb(n_sites, n_up) for n_up in range(n_sites + 1)}


def dimensions_by_enumeration(n_sites: int) -> dict[int, int]:
    """独立枚举所有比特串，并按其中 1 的个数统计维数。"""
    counts = {n_up: 0 for n_up in range(n_sites + 1)}
    for state in range(1 << n_sites):
        counts[state.bit_count()] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算自旋 1/2 链各 S^z_tot 扇区的 Hilbert 空间维数。"
    )
    parser.add_argument("--N", type=int, default=16, help="格点数（默认：16）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_sites = args.N
    if n_sites <= 0:
        raise ValueError("N 必须是正整数。")
    if n_sites > 24:
        raise ValueError("独立枚举校检需要 2^N 次操作；本脚本限制 N <= 24。")

    combinatorial = dimensions_from_binomial(n_sites)
    enumerated = dimensions_by_enumeration(n_sites)

    # 非平凡独立校检：组合数公式与逐个枚举比特串使用了两种不同算法。
    if combinatorial != enumerated:
        raise AssertionError("校检失败：组合数结果与比特串枚举结果不一致。")

    print(f"N = {n_sites}")
    print("\n各 S^z_tot 扇区维数：")
    print(f"{'n_up':>6} {'S^z_tot':>10} {'dimension':>12}")
    for n_up, dimension in combinatorial.items():
        sz_tot = n_up - n_sites / 2
        print(f"{n_up:6d} {sz_tot:10g} {dimension:12d}")

    total_dimension = sum(combinatorial.values())
    if n_sites % 2 == 0:
        zero_sector_dimension = combinatorial[n_sites // 2]
        print(f"\nS^z_tot = 0 扇区维数 = C({n_sites}, {n_sites // 2})"
              f" = {zero_sector_dimension}")
    else:
        print("\nN 为奇数，不存在 S^z_tot = 0 扇区。")

    print(f"全部 S^z_tot 扇区维数之和 = {total_dimension}")
    print(f"完整 Hilbert 空间维数 2^N = {1 << n_sites}")

    if total_dimension != (1 << n_sites):
        raise AssertionError("校检失败：所有扇区维数之和不等于 2^N。")

    print("\n[校检通过]")
    print("1. 组合数维数与逐一枚举 2^N 个比特串的统计完全一致。")
    print("   针对错误：n_up/S^z_tot 对应关系错误、漏计或重复计数基矢。")
    print("2. 所有扇区维数之和等于 2^N。")
    print("   针对错误：遗漏某个磁化扇区；出错时总和将不等于完整空间维数。")


if __name__ == "__main__":
    main()
