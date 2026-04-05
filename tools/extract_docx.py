from __future__ import annotations

import sys
from pathlib import Path

import docx


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python tools/extract_docx.py <input.docx> <output.md>")
        return 2

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    document = docx.Document(str(input_path))

    lines: list[str] = []
    lines.append(f"# Extracted: {input_path.name}")
    lines.append("")

    for para in document.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue

        style_name = (para.style.name or "").lower() if para.style is not None else ""
        if style_name.startswith("heading"):
            # Try to parse heading level like "Heading 1"
            level = 2
            parts = style_name.split()
            if len(parts) == 2 and parts[1].isdigit():
                level = max(1, min(6, int(parts[1])))
            lines.append("#" * level + " " + text)
        else:
            lines.append(text)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
