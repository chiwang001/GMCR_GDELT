"""Export existing PNG figures to same-name PDF and SVG files."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PDF and SVG counterparts for PNG figures."
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output" / "31-17-04" / "figures",
        help="Directory containing PNG figures.",
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=300.0,
        help="Resolution metadata written to PDF files.",
    )
    return parser.parse_args()


def export_png(png_path: Path, dpi: float) -> tuple[Path, Path]:
    with Image.open(png_path) as source:
        image = source.convert("RGB")
        pdf_path = png_path.with_suffix(".pdf")
        image.save(pdf_path, "PDF", resolution=dpi)
        width, height = image.size

    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<image width="{width}" height="{height}" preserveAspectRatio="none" '
        f'href="data:image/png;base64,{encoded}"/>'
        "</svg>\n"
    )
    svg_path = png_path.with_suffix(".svg")
    svg_path.write_text(svg, encoding="utf-8")
    return pdf_path, svg_path


def main() -> None:
    args = parse_args()
    if not args.figures_dir.is_dir():
        raise SystemExit(f"Figures directory does not exist: {args.figures_dir}")
    pngs = sorted(args.figures_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No PNG figures found in {args.figures_dir}")
    for png_path in pngs:
        export_png(png_path, args.dpi)
    print(f"Exported {len(pngs)} PNG figures to PDF and SVG in {args.figures_dir}")


if __name__ == "__main__":
    main()
