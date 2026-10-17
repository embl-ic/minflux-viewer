"""Compact streaming numeric columns without routing integer IDs through floats."""

from __future__ import annotations

import re
from array import array
from decimal import Decimal
from typing import Any

import numpy as np

_INTEGER = re.compile(r"^[+-]?\d+$")
_EXACT_FLOAT_INTEGER = 2**53


class NumericColumnBuffer:
    def __init__(self) -> None:
        self._buffer: array = array("q")
        self._minimum = 0
        self._maximum = 0
        self.nonempty = 0
        self.numeric = 0
        self.precision_loss = False

    def append(self, cell: Any, *, excel_numeric: bool = False) -> None:
        text = "" if cell is None else str(cell).strip()
        value: int | float = float("nan")
        if text:
            self.nonempty += 1
            try:
                if _INTEGER.fullmatch(text):
                    value = int(text)
                else:
                    value = float(text)
                    # Decimal/scientific spellings can also encode exact IDs.
                    if np.isfinite(value) and abs(value) >= _EXACT_FLOAT_INTEGER:
                        exact = Decimal(text)
                        if exact == exact.to_integral_value():
                            value = int(exact)
                self.numeric += 1
            except (ValueError, OverflowError):
                value = float("nan")
            if isinstance(cell, (float, np.floating)) or (
                    excel_numeric and isinstance(cell, (int, np.integer))):
                if abs(value) >= _EXACT_FLOAT_INTEGER:
                    self.precision_loss = True
        if isinstance(value, int):
            self._minimum = min(self._minimum, value)
            self._maximum = max(self._maximum, value)
            if value < -(2**63) or value >= 2**64:
                self.precision_loss = True
                value = float("nan")
            elif self._buffer.typecode != "d":
                if value >= 2**63:
                    if self._minimum < 0:
                        self.precision_loss = True
                        value = float(value)
                    elif self._buffer.typecode == "q":
                        self._buffer = array("Q", self._buffer)
                elif value < 0 and self._buffer.typecode == "Q":
                    self.precision_loss = True
                    value = float(value)
            if self._buffer.typecode == "d" and abs(value) >= _EXACT_FLOAT_INTEGER:
                self.precision_loss = True
        if not isinstance(value, int) and self._buffer.typecode != "d":
            if max(abs(self._minimum), abs(self._maximum)) >= _EXACT_FLOAT_INTEGER:
                self.precision_loss = True
            self._buffer = array("d", self._buffer)
        self._buffer.append(value)

    def values(self) -> np.ndarray:
        dtype = {"q": np.int64, "Q": np.uint64, "d": np.float64}[self._buffer.typecode]
        return np.frombuffer(self._buffer, dtype=dtype)
