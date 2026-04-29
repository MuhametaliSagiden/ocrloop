"""Render a sample mixed RU/EN image with decorative bullets for smoke-testing.

Not part of the bot — only used locally to verify the OCR pipeline.
"""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont


def find_font() -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, 28)
    return ImageFont.load_default()


def main(out_path: str) -> None:
    lines = [
        "Project plan / План проекта",
        "",
        "  • Setup environment / Настроить окружение",
        "  • Implement OCR / Реализовать OCR",
        "      ◦ Russian + English",
        "      ◦ Preserve indentation",
        "  ① First milestone — finish bot",
        "  ② Second milestone — deploy",
        "",
        "    Notes:",
        "        — keep formatting",
        "        — strip decorations",
    ]
    font = find_font()
    img = Image.new("RGB", (900, 60 + 36 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((30, 30 + i * 36), line, fill="black", font=font)
    img.save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sample.png")
