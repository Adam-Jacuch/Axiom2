from __future__ import annotations

import argparse
import ast
import inspect
import keyword
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import jax.nn as jnn
import jax.numpy as jnp

from axiom import core, nn


# These operations change physical array layout without updating Axiom topology.
IGNORE_LIST = {
    "append",
    "argwhere",
    "array",
    "asarray",
    "choose",
    "column_stack",
    "concatenate",
    "copy",
    "delete",
    "diag",
    "diagonal",
    "dstack",
    "empty",
    "empty_like",
    "expand_dims",
    "full",
    "full_like",
    "hstack",
    "insert",
    "moveaxis",
    "ndim",
    "ones",
    "ones_like",
    "put",
    "repeat",
    "reshape",
    "resize",
    "rollaxis",
    "shape",
    "size",
    "split",
    "squeeze",
    "stack",
    "swapaxes",
    "take",
    "tile",
    "transpose",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
}

# These remain available even when the generator is run outside this repository.
DEFAULT_AXIS_NAMES = {
    "a",
    "b",
    "c",
    "d",
    "h",
    "k",
    "q",
    "s",
    "t",
    "v",
    "w",
    "x",
    "y",
}

SCAN_SUFFIXES = {".md", ".py", ".pyi", ".rst"}
SKIP_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
}
MARKDOWN_AXIS_PATTERN = re.compile(r"(?<![\w.])ax\.([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class DynamicMethod:
    name: str
    func: Any
    native_axiom: bool = False

    @property
    def requires_axis(self) -> bool:
        try:
            return "axis" in inspect.signature(self.func).parameters
        except (TypeError, ValueError):
            return False


def get_callables(module: ModuleType) -> dict[str, Any]:
    """Return public callable functions while excluding dtype/classes."""
    callables: dict[str, Any] = {}
    for name in dir(module):
        if name.startswith("_") or not name.isidentifier() or keyword.iskeyword(name):
            continue
        try:
            value = getattr(module, name)
        except Exception:
            continue
        if callable(value) and not inspect.isclass(value) and not inspect.ismodule(value):
            callables[name] = value
    return callables


def collect_dynamic_methods() -> dict[str, DynamicMethod]:
    methods = {
        name: DynamicMethod(name, func)
        for name, func in {**get_callables(jnp), **get_callables(jnn)}.items()
        if name not in IGNORE_LIST
    }

    # Only decorated Axiom functions accept a TargetedTensor as their first input.
    for name, func in vars(nn).items():
        if (
            name.isidentifier()
            and not name.startswith("_")
            and callable(func)
            and getattr(func, "_is_axiom_nn", False)
        ):
            methods[name] = DynamicMethod(name, func, native_axiom=True)
    return methods


def _is_axis_namespace(node: ast.AST, aliases: set[str], module_aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in aliases
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "ax"
        and isinstance(node.value, ast.Name)
        and node.value.id in module_aliases
    )


def _axes_from_python(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    aliases: set[str] = set()
    module_aliases: set[str] = set()
    axis_class_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in {"axiom", "axiom.core", "core"}:
                for imported in node.names:
                    if imported.name == "ax":
                        aliases.add(imported.asname or imported.name)
                    elif imported.name == "Axis":
                        axis_class_aliases.add(imported.asname or imported.name)
        elif isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name in {"axiom", "axiom.core"}:
                    module_aliases.add(imported.asname or imported.name)

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_axis_namespace(
            node.value, aliases, module_aliases
        ):
            names.add(node.attr)
            continue

        if not isinstance(node, ast.Call) or not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
            continue

        if _is_axis_namespace(node.func, aliases, module_aliases):
            names.add(first_arg.value)
        elif isinstance(node.func, ast.Name) and node.func.id in axis_class_aliases:
            names.add(first_arg.value)
    return names


def _iter_scan_files(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if (
                path in seen
                or not path.is_file()
                or path.suffix.lower() not in SCAN_SUFFIXES
                or any(part in SKIP_DIRECTORIES for part in path.parts)
            ):
                continue
            seen.add(path)
            yield path


def discover_axis_names(roots: Iterable[Path]) -> set[str]:
    names = set(DEFAULT_AXIS_NAMES)
    for path in _iter_scan_files(roots):
        source = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() in {".py", ".pyi"}:
            names.update(_axes_from_python(source))
        else:
            names.update(MARKDOWN_AXIS_PATTERN.findall(source))
    return names


def _safe_default(value: Any) -> Any:
    if value is None or value is Ellipsis or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else Ellipsis
    if isinstance(value, (bytes, tuple, list, dict, set)):
        try:
            ast.literal_eval(repr(value))
        except (SyntaxError, ValueError):
            return Ellipsis
        return value
    if isinstance(value, frozenset) and not value:
        return value
    return Ellipsis


def _annotation_name(annotation: Any, default: Any, native_axiom: bool) -> str:
    if annotation is inspect.Parameter.empty:
        return "typing.Any"

    if isinstance(annotation, str):
        name = annotation.strip("'\"")
    elif annotation in {bool, bytes, float, int, str}:
        name = annotation.__name__
    else:
        name = inspect.formatannotation(annotation)

    if native_axiom:
        native_names = {
            "Axis": "Axis",
            "Tensor": "Tensor",
            "TargetedTensor": "TargetedTensor",
            "Tie": "Tie",
            "bool": "bool",
            "bytes": "bytes",
            "float": "float",
            "int": "int",
            "str": "str",
        }
        rendered = native_names.get(name, "typing.Any")
    else:
        jax_names = {
            "Any": "typing.Any",
            "ArrayLike": "ArrayLike",
            "ArrayLike | None": "ArrayLike | None",
            "DTypeLike": "DTypeLike",
            "DTypeLike | None": "DTypeLike | None",
            "None": "None",
            "bool": "bool",
            "bytes": "bytes",
            "float": "float",
            "int": "int",
            "str": "str",
        }
        rendered = jax_names.get(name, "typing.Any")

    if default is None and rendered not in {"None", "typing.Any"} and "| None" not in rendered:
        rendered += " | None"
    return rendered


def _bound_signature(method: DynamicMethod, inject_axis: bool) -> str:
    try:
        parameters = list(inspect.signature(method.func).parameters.values())
    except (TypeError, ValueError):
        return "(self, /, *args: typing.Any, **kwargs: typing.Any)"

    consumed_index = next(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ),
        None,
    )
    if consumed_index is None:
        return "(self, /, *args: typing.Any, **kwargs: typing.Any)"
    del parameters[consumed_index]

    if inject_axis:
        parameters = [parameter for parameter in parameters if parameter.name != "axis"]

    rendered_parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY)]
    for parameter in parameters:
        default = parameter.default
        if default is not inspect.Parameter.empty:
            default = _safe_default(default)
        rendered_parameters.append(
            parameter.replace(
                annotation=_annotation_name(
                    parameter.annotation, parameter.default, method.native_axiom
                ),
                default=default,
            )
        )

    try:
        signature = inspect.Signature(rendered_parameters)
    except ValueError:
        return "(self, /, *args: typing.Any, **kwargs: typing.Any)"
    return str(signature).replace("Ellipsis", "...")


def _valid_axis_names(names: Iterable[str]) -> list[str]:
    reserved = {
        name for name in vars(core._AxisNamespace) if not name.startswith("_")
    }
    return sorted(
        name
        for name in names
        if name.isidentifier()
        and not keyword.iskeyword(name)
        and not name.startswith("_")
        and name not in reserved
    )


def _render_axis_properties(axis_names: Iterable[str], return_type: str) -> list[str]:
    lines: list[str] = []
    for name in axis_names:
        lines.extend(
            [
                "    @property",
                f'    def {name}(self) -> "{return_type}": ...',
            ]
        )
    return lines


def _render_surface(
    class_name: str,
    return_type: str,
    axis_return_type: str,
    concrete_class: type[Any],
    targeted: bool,
    methods: dict[str, DynamicMethod],
    axis_names: Sequence[str],
) -> list[str]:
    concrete_members = set(vars(concrete_class))
    visible_axes = [
        name for name in axis_names if name not in concrete_members
    ]
    lines = [f"class {class_name}:"]
    lines.extend(_render_axis_properties(visible_axes, axis_return_type))

    for name, method in sorted(methods.items()):
        if (
            name in concrete_members
            or name in visible_axes
            or (method.native_axiom and not targeted)
            or (method.requires_axis and not targeted)
        ):
            continue
        signature = _bound_signature(method, inject_axis=targeted)
        lines.append(f'    def {name}{signature} -> "{return_type}": ...')

    if len(lines) == 1:
        lines.append("    pass")
    lines.append("")
    return lines


def render_stubs(
    scan_roots: Iterable[Path] = (),
    extra_axes: Iterable[str] = (),
) -> str:
    methods = collect_dynamic_methods()
    axis_names = _valid_axis_names(
        discover_axis_names(scan_roots) | set(extra_axes)
    )

    lines = [
        '"""Generated typing mixins for Axiom dynamic APIs. Do not edit by hand."""',
        "from __future__ import annotations",
        "",
        "import typing",
        "",
        "if typing.TYPE_CHECKING:",
        "    from jax.typing import ArrayLike, DTypeLike",
        "",
        "    from .core import Axis, Bundle, TargetedBundle, TargetedTensor, Tensor, Tie",
        "",
        "",
        "class AxisNamespaceStubs:",
    ]
    lines.extend(_render_axis_properties(axis_names, "Axis"))
    lines.append("")

    surfaces = [
        (
            "NNTensorStubs",
            "Tensor",
            "TargetedTensor",
            core.Tensor,
            False,
        ),
        (
            "NNTargetedTensorStubs",
            "Tensor",
            "TargetedTensor",
            core.TargetedTensor,
            True,
        ),
        (
            "NNBundleStubs",
            "Bundle",
            "TargetedBundle",
            core.Bundle,
            False,
        ),
        (
            "NNTargetedBundleStubs",
            "Bundle",
            "TargetedBundle",
            core.TargetedBundle,
            True,
        ),
    ]
    for surface in surfaces:
        lines.extend(
            _render_surface(
                *surface,
                methods=methods,
                axis_names=axis_names,
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def generate(
    output_path: Path | None = None,
    scan_roots: Iterable[Path] = (),
    extra_axes: Iterable[str] = (),
) -> Path:
    output_path = output_path or Path(__file__).with_name("_nn_stubs.py")
    content = render_stubs(scan_roots=scan_roots, extra_axes=extra_axes)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Axiom IDE typing mixins.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("_nn_stubs.py"),
        help="Destination file (default: src/axiom/_nn_stubs.py).",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        type=Path,
        dest="scan_roots",
        help="Project path to scan for ax.<axis> usage. May be repeated.",
    )
    parser.add_argument(
        "--axis",
        action="append",
        default=[],
        dest="extra_axes",
        help="Additional axis property to generate. May be repeated.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the destination does not match generated output.",
    )
    args = parser.parse_args(argv)

    scan_roots = args.scan_roots or [Path.cwd()]
    content = render_stubs(scan_roots=scan_roots, extra_axes=args.extra_axes)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != content:
            print(f"{args.output} is out of date.")
            return 1
        print(f"{args.output} is up to date.")
        return 0

    args.output.write_text(content, encoding="utf-8")
    method_count = len(collect_dynamic_methods())
    axis_count = len(
        _valid_axis_names(discover_axis_names(scan_roots) | set(args.extra_axes))
    )
    print(
        f"Generated {args.output} with {method_count} dynamic functions "
        f"and {axis_count} axis properties."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
