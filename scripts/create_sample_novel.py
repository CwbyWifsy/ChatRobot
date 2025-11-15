"""Utility to generate a small demo novel for quick RAG smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

SAMPLE_NOVEL = """第一章 缘起\n这里是故事的开端，主角从小村庄踏上旅途。\n\n第二章 初遇\n主角与伙伴第一次相遇，并决定一起面对危机。\n\n第三章 磨砺\n众人经历试炼，逐渐理解彼此的信念。\n"""


def write_sample(output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    target.write_text(SAMPLE_NOVEL, encoding="utf-8")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a short demo novel in UTF-8 encoding. "
            "This is handy when you just want to verify the upload "
            "and chat flow without preparing additional text files."
        )
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Directory where the sample novel will be written.",
    )
    parser.add_argument(
        "--name",
        default="sample.txt",
        help="Filename of the generated novel (default: sample.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = write_sample(args.output, args.name)
    print(f"Sample novel written to {path}")


if __name__ == "__main__":
    main()
