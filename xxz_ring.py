"""
第四讲作业：J1-J2 XXZ 环的对称性、动量分块与 E(k)
本文件实现【必做 1、2】及配套校检（必做校检 1、2 的相关部分）。

模型（自旋 1/2，周期边界）：
      H = J1 * sum_j [S_j^x S_{j+1}^x + S_j^y S_{j+1}^y + Delta S_j^z S_{j+1}^z]
        + J2 * sum_j [S_j^x S_{j+2}^x + S_j^y S_{j+2}^y + Delta S_j^z S_{j+2}^z]

比特约定：
  * s = sum_j b_j 2^j，比特 j 对应格点 j，
    b_j = 1 表示 spin up  (S_j^z = +1/2)，b_j = 0 表示 spin down。
  * S_tot^z = (#up - #down)/2；S_tot^z = 0 扇区即 #up = N/2。
  * 自旋 1/2 恒等式：
        S_i^x S_j^x + S_i^y S_j^y = (1/2)(S_i^+ S_j^- + S_i^- S_j^+)
    在反平行键上 = 「同时翻转两自旋」，矩阵元 J/2；平行键上为 0。
    S_i^z S_j^z：平行 +1/4，反平行 -1/4。
"""

from itertools import combinations
from math import comb

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh


# ==================================================================
# 1. 基矢与查找表
# ==================================================================

def build_basis(N, n_up):
    """构造上自旋数为 n_up 的基矢（升序）以及整数->列号查找表。"""
    tmp = []
    for occ in combinations(range(N), n_up):
        s = 0
        for j in occ:
            s |= (1 << j)
        tmp.append(s)
    states = np.array(sorted(tmp), dtype=np.int64)
    index = {int(s): i for i, s in enumerate(states)}
    return states, index


def build_full_basis(N):
    """全希尔伯特空间基矢 0 .. 2^N-1。"""
    states = np.arange(1 << N, dtype=np.int64)
    index = {int(s): int(s) for s in states}
    return states, index


def sector_dim(N, n_up):
    """S_tot^z = n_up - N/2 扇区维数 = C(N, n_up)。"""
    return comb(N, n_up)


# ==================================================================
# 2. 键表（无序对，去重）
# ==================================================================

def build_bond_pairs(N, distance):
    """返回距离 distance 的无序键列表 [(min,max)]，每个无序对只算一次。

    * 一般 N：近邻/次近邻各 N 条不同键。
    * N=2：(0,1) 与 (1,0) 同一对，去重后只剩 1 条
      —— 对应「N=2 只对 0-1 键求和一次」。
    * N=3 的次近邻与近邻重合、N=4 的 (0,2) 会被数两次，去重后各算一次。
    """
    pairs = set()
    for j in range(N):
        k = (j + distance) % N
        if j == k:
            continue
        pairs.add((min(j, k), max(j, k)))
    return sorted(pairs)


# ==================================================================
# 3. 稀疏哈密顿量
# ==================================================================

def build_hamiltonian(N, J1, J2, Delta, states, index):
    """在 states 张成的扇区内构造 H 的稀疏矩阵（CSR）。"""
    dim = len(states)
    diag = np.zeros(dim, dtype=np.float64)
    rows, cols, vals = [], [], []

    bonds = [(i, j, J1) for (i, j) in build_bond_pairs(N, 1)]
    bonds += [(i, j, J2) for (i, j) in build_bond_pairs(N, 2)]

    for (i, j, J) in bonds:
        if J == 0.0:
            continue
        mi = np.int64(1 << i)
        mj = np.int64(1 << j)
        flip = int(mi | mj)
        for col in range(dim):
            s = int(states[col])
            bi = (s & int(mi)) != 0
            bj = (s & int(mj)) != 0
            if bi == bj:
                diag[col] += J * Delta * 0.25
            else:
                diag[col] -= J * Delta * 0.25
                rows.append(index[s ^ flip])
                cols.append(col)
                vals.append(0.5 * J)

    for col in range(dim):
        rows.append(col)
        cols.append(col)
        vals.append(diag[col])

    return csr_matrix((vals, (rows, cols)), shape=(dim, dim))


# ==================================================================
# 4. 平移算符 T
# ==================================================================

def rotate(s, N, shift=1):
    """T|s> = |rot(s,+1)>：把格点 j 上的自旋搬到 j+1（循环左移比特）。"""
    out = 0
    for j in range(N):
        if (s >> j) & 1:
            out |= 1 << ((j + shift) % N)
    return out


def build_translation(N, states, index):
    """在给定扇区内构造平移算符 T 的置换矩阵（行=像, 列=原像）。"""
    dim = len(states)
    rows = np.empty(dim, dtype=np.int64)
    cols = np.arange(dim)
    for col, s in enumerate(states):
        rows[col] = index[rotate(int(s), N, 1)]
    return csr_matrix((np.ones(dim), (rows, cols)), shape=(dim, dim))


# ==================================================================
# 5. 求解工具
# ==================================================================

def low_eigs(H, k):
    """求最低 k 个本征值；维度很小时用稠密求解。"""
    dim = H.shape[0]
    if dim <= 2 * k + 2:
        w = np.linalg.eigvalsh(H.toarray())
        return np.sort(w)[:k], None
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(dim)
    w, v = eigsh(H, k=k, which="SA", v0=v0, tol=0)
    order = np.argsort(w)
    return w[order], v[:, order]


def degeneracy(evals, tol=1e-8):
    return int(np.sum(np.abs(evals - evals[0]) < tol))


# ==================================================================
# 6. 必做 1、2
# ==================================================================

def task1():
    print("=" * 64)
    print("必做 1：希尔伯特空间维数")
    print("=" * 64)
    N = 16
    d0 = sector_dim(N, N // 2)
    total = sum(sector_dim(N, n) for n in range(N + 1))
    print(f"N = {N}")
    print(f"  S_tot^z = 0 扇区维数 = C({N},{N//2}) = {d0}")
    print(f"  全部扇区维数之和 = 2^{N} = {total}")
    print(f"  校验 sum_n C(N,n) == 2^N : {total == 2 ** N}")


def task2():
    print()
    print("=" * 64)
    print("必做 2：基态能量（仅 U(1) 分块，Lanczos 迭代）")
    print("=" * 64)
    N, J1, Delta = 16, 1.0, 1.0
    results = {}
    for J2 in (0.0, 0.5):
        states, index = build_basis(N, N // 2)
        H = build_hamiltonian(N, J1, J2, Delta, states, index)
        evals, evecs = low_eigs(H, k=8)
        E0 = evals[0]
        deg = degeneracy(evals)
        res = np.linalg.norm(H @ evecs[:, 0] - E0 * evecs[:, 0])
        results[J2] = (E0, deg)
        print(f"J2 = {J2}")
        print(f"  扇区维数 dim          = {H.shape[0]}")
        print(f"  E0                    = {E0:.12f}")
        print(f"  E0/N                  = {E0 / N:.12f}")
        print(f"  基态简并度            = {deg}")
        print(f"  最低 8 个本征值       = {np.round(evals, 8)}")
        print(f"  残差 ||Hv - E0 v||    = {res:.3e}")
        print()
    return results


# ==================================================================
# 7. 校检
# ==================================================================

def check_hermiticity():
    """校检 1a：厄米性 max|H - H^dag| = 0。

    针对错误：非对角矩阵元只写了一半（漏掉共轭方向），或符号不一致。
    失败方式：max|H-H^dag| 变为 O(1)。"""
    N = 12
    states, index = build_basis(N, N // 2)
    H = build_hamiltonian(N, 1.0, 0.5, 1.0, states, index)
    d = abs(H - H.transpose()).max()
    print(f"[校检 厄米性]   max|H - H^dag| = {d:.3e}   pass={d < 1e-14}")


def check_n2():
    """校检 1b：N=2 全空间谱应为 {-3J1/4, +J1/4 (x3)}。

    针对错误：近邻键 (0,1)/(1,0) 未去重导致耦合翻倍。
    失败方式：谱变成 {-3J1/2, +J1/2 (x3)} 之类，数值翻倍。"""
    J1 = 1.0
    states, index = build_full_basis(2)
    H = build_hamiltonian(2, J1, 0.0, 1.0, states, index)
    w = np.sort(np.linalg.eigvalsh(H.toarray()))
    expect = np.array([-3 * J1 / 4, J1 / 4, J1 / 4, J1 / 4])
    err = np.abs(w - expect).max()
    print(f"[校检 N=2]      谱 = {np.round(w, 6)}  期望 = {np.round(expect, 6)}")
    print(f"                 max 误差 = {err:.3e}   pass={err < 1e-12}")


def check_n4():
    """校检 1c：N=4 海森堡环 (J2=0, Delta=1) 基态 E0 = -2 J1。

    针对错误：J1/J2 键表弄错、S^z 项符号或 1/4 系数写错、1/2 翻转矩阵元写错。
    失败方式：E0 偏离 -2J1。"""
    N, J1 = 4, 1.0
    states, index = build_basis(N, N // 2)
    H = build_hamiltonian(N, J1, 0.0, 1.0, states, index)
    w = np.sort(np.linalg.eigvalsh(H.toarray()))
    E0 = w[0]
    print(f"[校检 N=4]      E0 = {E0:.12f}  期望 = {-2 * J1:.12f}"
          f"  pass={abs(E0 + 2 * J1) < 1e-12}")


def check_translation():
    """校检 2：平移不变性 ||[H,T]||max = 0，且 T^N = I。

    针对错误：键表未按周期边界处理，或比特/格点映射不一致。
    失败方式：对易子出现 O(1) 非零元；T^N 偏离单位阵。"""
    N = 12
    states, index = build_basis(N, N // 2)
    H = build_hamiltonian(N, 1.0, 0.5, 1.0, states, index)
    T = build_translation(N, states, index)
    comm = H @ T - T @ H
    c = abs(comm).max()
    Tn = T
    for _ in range(N - 1):
        Tn = Tn @ T
    idn = identity(len(states), format="csr")
    d = abs(Tn - idn).max()
    print(f"[校检 平移]     ||[H,T]||max = {c:.3e}   pass={c < 1e-12}")
    print(f"                ||T^N - I||max = {d:.3e}   pass={d < 1e-12}")


if __name__ == "__main__":
    task1()
    task2()
    print("=" * 64)
    print("校检")
    print("=" * 64)
    check_hermiticity()
    check_n2()
    check_n4()
    check_translation()
