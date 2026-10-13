#!/usr/bin/env python3
"""一次性端到端检验必做 1-6。

每个独立脚本都内置了针对该数值结果的非平凡校检。本程序在临时目录中
依次运行六个入口、确认校检标记和图片有效性，并给出简洁的 PASS/FAIL
汇总。临时图片在退出时自动清理，不覆盖正式结果图。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ValidationCase:
    name: str
    script: str
    expected_markers: tuple[str, ...]
    output_image_name: str | None = None


CASES: tuple[ValidationCase, ...] = (
    ValidationCase(
        "必做 1：组合数与逐比特枚举",
        "mandatory_1_hilbert_space.py",
        ("[校检通过]", "完整 Hilbert 空间维数"),
    ),
    ValidationCase(
        "必做 2：小尺寸解析值、MG 解与 Lanczos 残差",
        "mandatory_2_ground_state.py",
        ("校检清单：", "Majumdar-Ghosh", "通过。"),
    ),
    ValidationCase(
        "必做 3：动量投影、维数与提升本征态",
        "mandatory_3_momentum_blocks.py",
        ("动量块维数", "max||T P_k-exp(i*k)P_k||", "通过。"),
    ),
    ValidationCase(
        "必做 4：独立 Pauli 构造与完整谱重组",
        "mandatory_4_full_block_table.py",
        ("Hamiltonian 独立构造", "完整谱重组", "通过。"),
    ),
    ValidationCase(
        "必做 5：分块/未分块低能谱与 E(k) 图",
        "mandatory_5_dispersion.py",
        ("未分块 E0/E1 误差", "max|E(k)-E(-k)|", "通过。"),
        "dispersion.png",
    ),
    ValidationCase(
        "必做 6：单态、扇区对称性、凸包与 m(h) 图",
        "mandatory_6_magnetization.py",
        ("||S_tot^+|psi>||", "自旋翻转", "凸包给出的磁化扇区", "通过。"),
        "magnetization.png",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一次性检验第四讲作业必做 1-6。")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("validation_report.txt"),
        help="检验报告路径（默认：validation_report.txt）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="每个独立脚本的超时秒数（默认：180）",
    )
    return parser.parse_args()


def validate_png(path: Path) -> tuple[int, int]:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"没有生成有效图片：{path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        if image.format != "PNG" or width < 500 or height < 300:
            raise AssertionError(
                f"图片格式或尺寸异常：format={image.format}, size={image.size}"
            )
    return width, height


def run_case(
    case: ValidationCase,
    project_directory: Path,
    temporary_directory: Path,
    timeout: float,
    environment: dict[str, str],
) -> tuple[bool, float, str]:
    command = [sys.executable, str(project_directory / case.script)]
    image_details = ""
    if case.output_image_name is not None:
        image_path = temporary_directory / case.output_image_name
        command.extend(("--output", str(image_path)))
    else:
        image_path = None

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=project_directory,
            env=environment,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        duration = time.perf_counter() - started
        captured = error.stdout or ""
        return False, duration, f"超时（>{timeout:g}s）\n{captured}"

    duration = time.perf_counter() - started
    if completed.returncode != 0:
        return (
            False,
            duration,
            f"退出状态码 {completed.returncode}\n{completed.stdout}",
        )

    missing_markers = [
        marker for marker in case.expected_markers if marker not in completed.stdout
    ]
    if missing_markers:
        return (
            False,
            duration,
            f"缺少校检输出标记：{missing_markers}\n{completed.stdout}",
        )

    if image_path is not None:
        try:
            width, height = validate_png(image_path)
            image_details = f"；PNG={width}x{height}"
        except Exception as error:  # noqa: BLE001 - report validation failure verbatim
            return False, duration, str(error)
    return True, duration, f"全部内置校检通过{image_details}"


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout 必须为正数。")

    project_directory = Path(__file__).resolve().parent
    report_path = (
        args.report if args.report.is_absolute() else project_directory / args.report
    )
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    report_lines = [
        "第四讲作业必做 1-6 一次性检验报告",
        f"时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Python：{sys.version.split()[0]}",
        "",
    ]
    failures = 0
    with tempfile.TemporaryDirectory(prefix="xxz-homework-validation-") as temp:
        temporary_directory = Path(temp)
        for number, case in enumerate(CASES, start=1):
            passed, duration, details = run_case(
                case,
                project_directory,
                temporary_directory,
                args.timeout,
                environment,
            )
            status = "PASS" if passed else "FAIL"
            line = f"[{status}] {number}. {case.name} ({duration:.2f}s)"
            print(line, flush=True)
            report_lines.extend((line, f"       {details}", ""))
            if not passed:
                failures += 1
                print(details, file=sys.stderr)

    summary = f"总计：{len(CASES) - failures}/{len(CASES)} 项通过"
    report_lines.append(summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(f"{summary}\n报告：{report_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
