# main_parser.py
from pathlib import Path
from .msr_parser import parse_modern, parse_legacy
from .parser_model import DatasetInfo

def parse_msr(msr_path: str, tmp: str, log=print) -> dict:
    try:
        # Try modern path
        datasets = parse_modern(msr_path, tmp, log)
        if datasets:
            return {"mode": "modern", "datasets": datasets}
        else:
            raise Exception("No datasets found by specpy")
    except Exception as e:
        log(f"[main_parser] legacy fallback: {e}")
        datasets = parse_legacy(msr_path, tmp, log)
        return {"mode": "legacy", "datasets": datasets}
