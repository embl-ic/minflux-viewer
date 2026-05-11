# msr_test_parser.py

import sys
from pathlib import Path
from rich import print
from rich.tree import Tree
from rich.pretty import Pretty
from .main_parser import parse_msr

def build_rich_tree(parsed):
    tree = Tree(f"[bold]Parsed MSR[/bold] (mode: {parsed.get('mode')})")
    for ds in parsed.get("datasets", []):
        ds_label = f"{ds.name} [dim](did: {ds.did})"
        ds_node = tree.add(ds_label)
        add_field_nodes(ds_node, ds.fields)
    return tree

def add_field_nodes(tree_node, fields):
    for f in fields:
        label = f"{f.name} : {f.dtype} {f.shape}"
        print("f in fields, f.name: " + f.name)
        child = tree_node.add(label)
        if f.children:
            print("if f.children true, f.name: " + f.name)
            add_field_nodes(child, f.children)

def main(msr_path, tmp_dir=".tmp_zarr"):
    Path(tmp_dir).mkdir(exist_ok=True)
    parsed = parse_msr(msr_path, tmp_dir, log=print)
    print(build_rich_tree(parsed))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python msr_test_parser.py <file1.msr> [<file2.msr> ...]")
        sys.exit(1)
    tmp = ".tmp_zarr"
    for msr in sys.argv[1:]:
        print(f"\n[green]=== Parsing {msr} ===[/green]")
        main(msr, tmp_dir=tmp)
