"""Lazy compatibility for symbols that have moved between internal modules."""

from importlib import import_module


def resolve_moved_symbol(module_name: str, name: str, aliases: dict[str, str]):
    target = aliases.get(name)
    if target is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    module, _, symbol = target.partition(":")
    return getattr(import_module(module), symbol)
