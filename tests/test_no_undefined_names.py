"""Every global name a function reaches for must actually exist.

A name that is only referenced on a branch nobody takes in the tests is
invisible to the suite until a user takes it. That is how
``AttributeComponent`` survived in the MSR reader's viewer-open path after the
provenance stamping moved to ``msr/import_stamp.py`` and took the import with
it: the line runs only for a ``.msr`` whose channels carry MBM beads, so most
files imported fine and one reported ``name 'AttributeComponent' is not
defined`` for every dataset in it.

Python resolves a global at call time, so no amount of importing finds this.
The bytecode does: a ``LOAD_GLOBAL`` naming something that is neither a module
global nor a builtin cannot succeed, whatever the branch.
"""

from __future__ import annotations

import builtins
import dis
import importlib
import pkgutil
import types

import pytest

import minflux_viewer

#: Modules whose import needs a package this application does not depend on.
#: They are developer tools, not part of the viewer, and are reported as
#: skipped rather than silently passed.
OPTIONAL_IMPORTS = {
    "minflux_viewer.msr.cli": "click",
    "minflux_viewer.msr.msr_test_parser": "rich",
}

_BUILTINS = frozenset(dir(builtins))


def _code_objects(code):
    """*code* and every code object nested in it (closures, comprehensions)."""
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


def _defined_here(fn, module) -> bool:
    """⚠ Compare the code object's FILE, not ``__module__``.

    A dataclass' generated ``__repr__`` is built by ``reprlib`` and its
    ``__module__`` is rewritten to the dataclass' own module, so a
    ``__module__`` test drags in ``reprlib``'s globals (``_thread``) and
    ``contextlib``'s (``_GeneratorContextManager``) and reports them missing.
    """
    return fn.__code__.co_filename == getattr(module, "__file__", None)


def _functions(module):
    """Functions written in *module*: at module level and on its classes."""
    for value in list(vars(module).values()):
        if isinstance(value, types.FunctionType) and _defined_here(value, module):
            yield value
        elif isinstance(value, type) and value.__module__ == module.__name__:
            for member in list(vars(value).values()):
                fn = getattr(member, "__func__", member)   # unwrap static/class
                if isinstance(fn, types.FunctionType) and _defined_here(fn, module):
                    yield fn


def unresolved_globals(module) -> list[tuple[str, str, int]]:
    """``(qualname, name, line)`` for each global *module* cannot resolve."""
    known = set(vars(module)) | _BUILTINS
    found = []
    for fn in _functions(module):
        for code in _code_objects(fn.__code__):
            for ins in dis.get_instructions(code):
                if ins.opname == "LOAD_GLOBAL" and ins.argval not in known:
                    found.append((code.co_qualname, ins.argval,
                                  ins.positions.lineno))
    return found


def _module_names():
    return sorted(m.name for m in pkgutil.walk_packages(
        minflux_viewer.__path__, f"{minflux_viewer.__name__}."))


@pytest.mark.parametrize("name", _module_names())
def test_module_reaches_for_no_name_it_does_not_have(name):
    dependency = OPTIONAL_IMPORTS.get(name)
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if dependency and dependency in str(exc):
            pytest.skip(f"needs the optional '{dependency}' package")
        raise

    missing = unresolved_globals(module)
    assert not missing, "\n".join(
        f"{name}:{line}  {qual} reads undefined global {nm!r}"
        for qual, nm, line in missing)


def test_the_check_would_catch_a_lost_import():
    """The whole point, on a module written for the purpose."""
    module = types.ModuleType("probe")
    module.__file__ = "probe.py"
    code = compile(
        "def only_on_a_rare_branch(flag):\n"
        "    if flag:\n"
        "        return AttributeComponent({})\n"
        "    return None\n",
        "probe.py", "exec")
    exec(code, vars(module))

    assert [nm for _q, nm, _l in unresolved_globals(module)] == [
        "AttributeComponent"]
    # ...and stops complaining once the name is there
    module.AttributeComponent = dict
    assert unresolved_globals(module) == []
