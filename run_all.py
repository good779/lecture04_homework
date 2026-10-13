#!/usr/bin/env python3
"""一次运行必做 1-6，显示并保存全部文字输出与两张结果图。

本脚本调用各题独立程序，而不复制其中的物理计算。任一子程序失败时立即
停止，并保留已经产生的文字输出，便于定位问题。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TASKS: tuple[tuple[str, str], ...] = (
    ("必做 1：Hilbert 空间维数", "mandatory_1_hilbert_space.py"),
    ("必做 2：U(1) 扇区基态", "mandatory_2_ground_state.py"),
    ("必做 3：动量分块", "mandatory_3_momentum_blocks.py"),
    ("必做 4：N=8 完整分块表", "mandatory_4_full_block_table.py"),
    ("必做 5：色散关系 E(k)", "mandatory_5_dispersion.py"),
    ("必做 6：局域磁化与 m(h)", "mandatory_6_magnetization.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="依次运行第四讲作业必做 1-6，并汇总全部输出。"
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=Path("all_homework_outputs.txt"),
        help="汇总文字输出文件（默认：all_homework_outputs.txt）",
    )
    return parser.parse_args()


def section_header(task_number: int, title: str, script_name: str) -> str:
    rule = "=" * 88
    return (
        f"{rule}\n"
        f"[{task_number}/{len(TASKS)}] {title}\n"
        f"运行：{script_name}\n"
        f"{rule}"
    )


def write_log(path: Path, sections: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_directory = Path(__file__).resolve().parent
    output_log = (
        args.output_log
        if args.output_log.is_absolute()
        else project_directory / args.output_log
    )
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    collected_sections: list[str] = []
    for task_number, (title, script_name) in enumerate(TASKS, start=1):
        header = section_header(task_number, title, script_name)
        print("\n" + header, flush=True)
        completed = subprocess.run(
            [sys.executable, str(project_directory / script_name)],
            cwd=project_directory,
            env=environment,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        child_output = completed.stdout.rstrip()
        if child_output:
            print(child_output, flush=True)
        collected_sections.append(f"{header}\n{child_output}")
        write_log(output_log, collected_sections)

        if completed.returncode != 0:
            print(
                f"\n运行失败：{script_name} 返回状态码 {completed.returncode}。\n"
                f"已将此前输出保存到：{output_log}",
                file=sys.stderr,
            )
            raise SystemExit(completed.returncode)

    figure_paths = (
        project_directory / "mandatory_5_dispersion.png",
        project_directory / "mandatory_6_magnetization_curve.png",
    )
    missing_figures = [path for path in figure_paths if not path.is_file()]
    if missing_figures:
        missing_text = ", ".join(str(path) for path in missing_figures)
        raise FileNotFoundError(f"所有程序成功，但缺少预期图片：{missing_text}")

    summary = (
        "=" * 88
        + "\n全部必做 1-6 已成功运行。\n"
        + f"汇总文字输出：{output_log}\n"
        + "结果图片：\n"
        + "\n".join(f"  - {path}" for path in figure_paths)
        + "\n"
        + "=" * 88
    )
    print("\n" + summary)
    collected_sections.append(summary)
    write_log(output_log, collected_sections)


if __name__ == "__main__":
    main()
