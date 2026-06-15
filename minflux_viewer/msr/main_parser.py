# main_parser.py — thin back-compat wrapper; prefer parse_general / parse_msr_general.
from .msr_parser import parse_general


def parse_msr(msr_path: str, tmp: str, log=print) -> dict:
    """Parse an .msr file via the pure-Python msr-reader pipeline."""
    return parse_general(msr_path, tmp, log=log)
