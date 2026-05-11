import click
from .io import pick_one_msr, parse_msr_to_tree, process_file

@click.group()
def main():
    """MINFLUX .msr CLI (parse/export)."""

@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--tmp-dir", required=True, type=click.Path(), help="Temporary dir for Zarr + exports")
def parse(input_path, tmp_dir):
    """Parse a single MSR (file or first in folder) and print Zarr tree."""
    msr = pick_one_msr(input_path)
    if not msr:
        click.echo("no msr file found!")
        return
    def log(msg): click.echo(msg)
    recs = parse_msr_to_tree(msr, tmp_dir, log=log)
    for r in recs:
        click.echo(f"did={r['did']} name={r['name']}")
        for f in r["fields"]:
            if f["kind"] == "array":
                click.echo(f"  {f['path']}  {f['shape']} {f['dtype']}")
            else:
                click.echo(f"  {f['path']}/ (group)")

@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--tmp-dir", required=True, type=click.Path())
@click.option("--mode", type=click.Choice(["mfx", "mbm", "both-separate", "both-combined"], case_sensitive=False), default="both-separate")
@click.option("--fmt", "fmts", multiple=True, type=click.Choice(["npy", "mat", "csv"]), default=["npy"])
def export(input_path, tmp_dir, mode, fmts):
    """Export mfx/mbm for each dataset (classic flow)."""
    msr = pick_one_msr(input_path)
    if not msr:
        click.echo("no msr file found!")
        return
    do_mfx = mode in ["mfx", "both-separate", "both-combined"]
    do_mbm = mode in ["mbm", "both-separate", "both-combined"]
    combine = mode == "both-combined"
    def log(msg): click.echo(msg)
    process_file(msr, tmp_dir, do_mfx, do_mbm, combine, list(fmts), log=log)
