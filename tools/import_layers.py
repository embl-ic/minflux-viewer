"""The package's import graph, by layer.

``tests/test_import_layers.py`` holds the rule to it: which layer may import
which, and which import cycles exist. Run this file to print the current
crossings and cycles in the form that test lists them::

    python tools/import_layers.py

Imports are read from the source with ``ast``, never by importing. An import
inside a function is *deferred*; one in a module or class body runs at *import
time*; one under ``if TYPE_CHECKING:`` never runs and is left out. Importing a
module is not counted as importing its parent packages, and a module named in
a string (``importlib.import_module``) is not seen.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "minflux_viewer"
PACKAGE_DIR = ROOT / PACKAGE

IMPORT_TIME = "import time"
DEFERRED = "deferred"

#: The layer of each first-level package or module ("" is the package itself).
LAYER_OF = {
    "": "base",
    "colors": "base",
    "colormaps": "base",
    "utils": "utils",
    "core": "core",
    "msr": "msr",
    "analysis": "analysis",
    "api": "api",
    "scripting": "api",        # binds the api to the running application
    "ui": "ui",
    "plugins": "plugins",
    "__main__": "entry",
}

_EVERYTHING = frozenset(LAYER_OF.values())

#: What each layer may import, at import time or later.
ALLOWED = {
    "base": frozenset({"base"}),
    "utils": frozenset({"base", "utils"}),
    "core": frozenset({"base", "utils", "core"}),
    "msr": frozenset({"base", "utils", "core", "msr"}),
    "analysis": frozenset({"base", "utils", "core", "msr", "analysis"}),
    "api": frozenset({"base", "utils", "core", "msr", "analysis", "api"}),
    "ui": _EVERYTHING,
    "plugins": _EVERYTHING,
    "entry": _EVERYTHING,
}

#: What a layer may import only from inside a function: the api acts on a
#: running application, so it may open its windows but not depend on them.
DEFERRED_ONLY = {"api": frozenset({"ui", "plugins"})}


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str          # IMPORT_TIME or DEFERRED


def module_names(package_dir: Path, package: str) -> dict[str, Path]:
    """Every module of the package, by dotted name."""
    names = {}
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(package_dir).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names[".".join([package, *parts])] = path
    return names


def _deepest_module(name: str, known: set[str]) -> str | None:
    while name and name not in known:
        name = name.rpartition(".")[0]
    return name or None


def _is_type_checking(test: ast.expr) -> bool:
    return ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))


def _imported(node: ast.Import | ast.ImportFrom, module: str,
              is_package: bool) -> list[str]:
    """The dotted names one import statement brings in."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        base = module.split(".")
        if not is_package:
            base = base[:-1]
        base = base[:len(base) - (node.level - 1)]
        origin = ".".join(base + ([node.module] if node.module else []))
    else:
        origin = node.module or ""
    return [origin] + [f"{origin}.{alias.name}" for alias in node.names]


def _module_edges(module: str, path: Path, known: set[str]) -> set[Edge]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_package = path.name == "__init__.py"
    edges: set[Edge] = set()

    def visit(node: ast.AST, kind: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking(child.test):
                for other in child.orelse:          # the else branch does run
                    visit_one(other, kind)
                continue
            visit_one(child, kind)

    def visit_one(node: ast.AST, kind: str) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in _imported(node, module, is_package):
                target = _deepest_module(name, known)
                if target is not None and target != module:
                    edges.add(Edge(module, target, kind))
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            visit(node, DEFERRED)
        else:
            visit(node, kind)

    visit(tree, IMPORT_TIME)
    return edges


def scan(package_dir: Path, package: str) -> set[Edge]:
    """Every import between the package's modules."""
    names = module_names(package_dir, package)
    known = set(names)
    edges: set[Edge] = set()
    for module, path in names.items():
        edges |= _module_edges(module, path, known)
    return edges


def short(name: str, package: str) -> str:
    """``package.core.loader`` -> ``core.loader``; the package itself stays."""
    return name[len(package) + 1:] if name.startswith(package + ".") else name


def first_level(name: str, package: str) -> str:
    rest = short(name, package)
    return "" if rest == name else rest.split(".")[0]


def unplaced(edges: Iterable[Edge], package_dir: Path, package: str) -> set[str]:
    """First-level names no layer claims."""
    names = {first_level(name, package) for name in module_names(package_dir, package)}
    return {name for name in names if name not in LAYER_OF}


def _layer(name: str, package: str) -> str | None:
    return LAYER_OF.get(first_level(name, package))


def crossings(edges: Iterable[Edge], package: str) -> set[str]:
    """Imports the layer rule does not allow, as ``source -> target (kind)``."""
    found = set()
    for edge in edges:
        source, target = _layer(edge.source, package), _layer(edge.target, package)
        if source is None or target is None:
            continue                  # reported by unplaced() instead
        if target in ALLOWED[source]:
            continue
        if edge.kind == DEFERRED and target in DEFERRED_ONLY.get(source, ()):
            continue
        found.add(f"{short(edge.source, package)} -> "
                  f"{short(edge.target, package)} ({edge.kind})")
    return found


def cycles(edges: Iterable[Edge], package: str, *,
           import_time_only: bool = False) -> set[tuple[str, ...]]:
    """The import cycles, each as the sorted names of its modules."""
    graph: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if import_time_only and edge.kind != IMPORT_TIME:
            continue
        graph[edge.source].add(edge.target)
    return {tuple(sorted(short(name, package) for name in component))
            for component in _strongly_connected(graph) if len(component) > 1}


def _strongly_connected(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's algorithm, iteratively, so a deep graph cannot exhaust the stack."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0
    nodes = set(graph) | {target for targets in graph.values() for target in targets}
    for root in sorted(nodes):
        if root in index:
            continue
        work = [(root, iter(sorted(graph.get(root, ()))))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph.get(child, ())))))
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    components.append(component)
    return components


def main() -> int:
    edges = scan(PACKAGE_DIR, PACKAGE)
    missing = unplaced(edges, PACKAGE_DIR, PACKAGE)
    if missing:
        print(f"Not in any layer: {sorted(missing)}")
    print("KNOWN_CROSSINGS = frozenset({")
    for item in sorted(crossings(edges, PACKAGE)):
        print(f"    {item!r},")
    print("})")
    for title, only in (("KNOWN_IMPORT_TIME_CYCLES", True), ("KNOWN_CYCLES", False)):
        print(f"\n{title} = frozenset({{")
        for cycle in sorted(cycles(edges, PACKAGE, import_time_only=only)):
            print(f"    {cycle!r},")
        print("})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
