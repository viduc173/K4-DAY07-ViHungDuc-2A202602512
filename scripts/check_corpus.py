#!/usr/bin/env python3
"""Kiểm tra corpus: metadata bắt buộc, khớp sources.csv, số lượng 5-10 file.

Cách dùng: python scripts/check_corpus.py data/vinuni-registrar
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REQUIRED = ["doc_id", "title", "source_url", "retrieved_at", "document_version", "audience"]


def front_matter(text: str) -> dict[str, str]:
    block = text.split("---")[1]
    pairs = re.findall(r"^(\w+):\s*(.+)$", block, re.M)
    return {key: value.strip().strip('"') for key, value in pairs}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_corpus.py <thu-muc-chu-de>", file=sys.stderr)
        return 2
    folder = Path(sys.argv[1])
    files = sorted(folder.glob("*.md"))
    rows = list(csv.DictReader((folder / "sources.csv").open(encoding="utf-8")))

    ids: list[str] = []
    audiences: dict[str, int] = {}
    for path in files:
        meta = front_matter(path.read_text(encoding="utf-8"))
        ids.append(meta.get("doc_id", ""))
        audiences[meta.get("audience", "?")] = audiences.get(meta.get("audience", "?"), 0) + 1
        missing = [key for key in REQUIRED if key not in meta]
        ok = not missing and meta["doc_id"] == path.stem
        print(f"{path.name:40} {'OK' if ok else 'THIEU/SAI: ' + ','.join(missing or ['doc_id != ten file'])}")

    print("so file :", len(files), "(can 5-10)")
    print("csv     :", "khop" if sorted(r["doc_id"] for r in rows) == sorted(ids) else "LECH")
    print("audience:", audiences)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
