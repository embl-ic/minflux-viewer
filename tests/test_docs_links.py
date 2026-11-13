"""Every relative link in the published documentation resolves.

Covers ``docs/**/*.md`` and the README: the linked file must exist, and a
``#section`` must match a heading of the linked Markdown page (GitHub's anchor
rule: lower case, punctuation dropped, spaces to hyphens).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LINK = re.compile(r"\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def _anchors(markdown: str) -> set[str]:
    return {re.sub(r"[^\w\- ]", "", title.strip().lower()).replace(" ", "-")
            for title in _HEADING.findall(markdown)}


def _pages() -> list[Path]:
    return sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]


def test_every_relative_link_in_the_docs_resolves():
    problems = []
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        for target in _LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, anchor = target.partition("#")
            dest = (page.parent / path).resolve() if path else page.resolve()
            where = f"{page.relative_to(ROOT).as_posix()}: {target}"
            if not dest.exists():
                problems.append(f"missing file  {where}")
            elif anchor and dest.suffix == ".md" and anchor not in _anchors(
                    dest.read_text(encoding="utf-8")):
                problems.append(f"missing anchor  {where}")
    assert not problems, "\n".join(problems)


def test_the_manual_index_reaches_every_page():
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    linked = {(ROOT / "docs" / target.partition("#")[0]).resolve()
              for target in _LINK.findall(index) if not target.startswith("http")}
    pages = {p.resolve() for p in (ROOT / "docs").rglob("*.md")
             if p.name != "README.md" or p.parent != ROOT / "docs"}
    assert pages <= linked, sorted(p.relative_to(ROOT).as_posix() for p in pages - linked)
