# parser_model.py
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class FieldInfo:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    logical_shape: str
    kind: str           # "array", "structured", "struct-root", "legacy"
    children: List["FieldInfo"] = field(default_factory=list)

@dataclass
class DatasetInfo:
    did: str
    name: str
    root: str           # Zarr root path; empty for legacy
    fields: List[FieldInfo]
