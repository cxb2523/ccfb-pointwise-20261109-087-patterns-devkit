"""Offline validation of pipeline graph yaml files.

The validation never touches the network: no login, upload, or component
directory requests are made. All checks run against the raw yaml document.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import typer
import yaml
from typer import Argument

from patterns.cli.commands._common import load_yaml
from patterns.cli.services import versions
from patterns.cli.services.output import abort, sprint

# The node types the devkit knows how to create locally. Keep this list in one
# place so the check and its error message can never drift apart.
ALLOWED_TYPES: Tuple[str, ...] = ("function", "component", "webhook", "table")

# Fixed ordering for the different kinds of errors. Combined with component
# document order this makes multi-problem output deterministic.
class ErrorType(Enum):
    duplicate_id = "duplicate_id"
    unknown_type = "unknown_type"
    invalid_dependency = "invalid_dependency"
    ambiguous_dependency = "ambiguous_dependency"
    missing_dependency = "missing_dependency"
    self_dependency = "self_dependency"
    dependency_cycle = "dependency_cycle"


_COMPONENT_SECTIONS = ("functions", "stores")


class _Component:
    """A single component declaration as it appears in the raw document."""

    def __init__(self, order: int, section: str, index: int, raw: dict):
        # Order of appearance across the whole document (functions then stores).
        self.order = order
        self.section = section
        self.index = index
        self.raw = raw
        self.id = raw.get("id")

    @property
    def label(self) -> str:
        """Stable, human-readable location of the declaration."""
        singular = self.section[:-1]  # functions -> function, stores -> store
        return f"{singular}[{self.index}]"

    @property
    def type(self) -> str:
        """Return the declared type, or derive it from the declaration fields.

        An explicit ``type`` field wins. Otherwise the type is inferred from
        ``uses`` / ``webhook`` / ``table`` / ``node_file``; a declaration with
        none of these resolves to ``"unknown"``.
        """
        explicit = self.raw.get("type")
        if explicit is not None:
            return str(explicit)
        if self.raw.get("uses"):
            return "component"
        if self.raw.get("webhook"):
            return "webhook"
        if self.raw.get("table"):
            return "table"
        if self.raw.get("node_file"):
            return "function"
        return "unknown"


class _ValidationError:
    def __init__(
        self,
        component: Optional[_Component],
        error_type: ErrorType,
        message: str,
    ):
        self.component = component
        # Components without an id sort after every declared component.
        self.order = component.order if component is not None else float("inf")
        self.error_type = error_type
        self.message = message


def validate_graph_config(config: Any) -> List[_ValidationError]:
    """Validate a parsed graph yaml document and collect every error found.

    Analysis never short-circuits on a bad declaration: it is reported and
    skipped so that remaining components can still be checked.
    """
    errors: List[_ValidationError] = []
    if not isinstance(config, dict):
        abort(
            "Invalid graph configuration: expected a mapping with "
            "'functions' and/or 'stores' entries"
        )

    components = _collect_components(config)

    # Build the component id index straight from the raw document. A duplicated
    # id is reported and marked ambiguous; the graph must never silently pick
    # one of its instances.
    id_index, ambiguous_ids = _build_id_index(components, errors)

    for component in components:
        if component.type not in ALLOWED_TYPES:
            errors.append(
                _ValidationError(
                    component,
                    ErrorType.unknown_type,
                    f"{component.label}: unknown component type "
                    f"'{component.type}'; allowed types are: "
                    f"{', '.join(ALLOWED_TYPES)}",
                )
            )
        # Type problems must not block dependency analysis.

    # Normalize every dependencies declaration once; every later graph check
    # reuses this single adjacency structure.
    adjacency = _build_adjacency(components, errors)

    _check_edges(components, id_index, ambiguous_ids, adjacency, errors)
    _check_cycles(components, id_index, ambiguous_ids, adjacency, errors)

    errors.sort(key=lambda e: (e.order, e.error_type.value))
    return errors


def _collect_components(config: dict) -> List[_Component]:
    components: List[_Component] = []
    order = 0
    for section in _COMPONENT_SECTIONS:
        entries = config.get(section)
        if entries is None:
            continue
        if not isinstance(entries, list):
            abort(f"Invalid graph configuration: '{section}' must be a list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                abort(
                    f"Invalid graph configuration: {section[:-1]}[{index}] "
                    "must be a mapping"
                )
            components.append(_Component(order, section, index, entry))
            order += 1
    return components


def _build_id_index(
    components: List[_Component], errors: List[_ValidationError]
) -> Tuple[Dict[str, _Component], Set[str]]:
    id_index: Dict[str, _Component] = {}
    ambiguous_ids: Set[str] = set()
    for component in components:
        component_id = component.id
        if component_id is None:
            continue
        if component_id in ambiguous_ids:
            # Third and later copies get the same stable report.
            errors.append(_duplicate_id_error(component, component_id))
            continue
        existing = id_index.get(component_id)
        if existing is not None:
            ambiguous_ids.add(component_id)
            for duplicate in (existing, component):
                errors.append(_duplicate_id_error(duplicate, component_id))
            continue
        id_index[component_id] = component
    return id_index, ambiguous_ids


def _duplicate_id_error(
    component: _Component, component_id: str
) -> _ValidationError:
    return _ValidationError(
        component,
        ErrorType.duplicate_id,
        f"{component.label}: duplicate component id '{component_id}' "
        "(id is ambiguous)",
    )


def _build_adjacency(
    components: List[_Component], errors: List[_ValidationError]
) -> Dict[int, Set[str]]:
    """Normalize every dependencies declaration into one adjacency mapping.

    The mapping is keyed by component document order rather than id so that
    declarations sharing a duplicated id can never overwrite each other's
    edges. A missing field is treated as an empty set. Strings and lists are
    both accepted. Malformed declarations are reported and contribute no
    edges, so the rest of the graph can still be analyzed.
    """
    adjacency: Dict[int, Set[str]] = {}
    for component in components:
        if component.id is None:
            # Nothing can reference a component without an id, and it cannot be
            # the source of a reliable edge, so it gets no adjacency entry.
            continue
        raw_deps = component.raw.get("dependencies")
        deps: Set[str] = set()
        if raw_deps is None:
            adjacency[component.order] = deps
            continue
        if isinstance(raw_deps, str):
            raw_deps = [raw_deps]
        elif not isinstance(raw_deps, list):
            errors.append(
                _ValidationError(
                    component,
                    ErrorType.invalid_dependency,
                    f"{component.label}: 'dependencies' must be a component id "
                    "string or a list of component id strings",
                )
            )
            adjacency[component.order] = deps
            continue

        for dep in raw_deps:
            if not isinstance(dep, str) or not dep:
                errors.append(
                    _ValidationError(
                        component,
                        ErrorType.invalid_dependency,
                        f"{component.label}: dependency declarations must be "
                        "non-empty component id strings",
                    )
                )
            else:
                deps.add(dep)
        adjacency[component.order] = deps
    return adjacency


def _check_edges(
    components: List[_Component],
    id_index: Dict[str, _Component],
    ambiguous_ids: Set[str],
    adjacency: Dict[int, Set[str]],
    errors: List[_ValidationError],
) -> None:
    for component in components:
        if component.id is None:
            continue
        for dep_id in sorted(adjacency[component.order]):
            if dep_id in ambiguous_ids:
                # Never resolve an ambiguous id to one of its instances: doing
                # so would let cycle detection run on a fabricated topology.
                errors.append(
                    _ValidationError(
                        component,
                        ErrorType.ambiguous_dependency,
                        f"{component.label}: dependency '{dep_id}' is ambiguous "
                        "because the id is declared more than once",
                    )
                )
            elif dep_id not in id_index:
                errors.append(
                    _ValidationError(
                        component,
                        ErrorType.missing_dependency,
                        f"{component.label}: dependency '{dep_id}' does not "
                        "match any component id",
                    )
                )
            elif dep_id == component.id:
                errors.append(
                    _ValidationError(
                        component,
                        ErrorType.self_dependency,
                        f"{component.label}: component depends on itself "
                        f"('{dep_id}')",
                    )
                )
            # A resolvable edge stays in adjacency for cycle detection.


def _check_cycles(
    components: List[_Component],
    id_index: Dict[str, _Component],
    ambiguous_ids: Set[str],
    adjacency: Dict[int, Set[str]],
    errors: List[_ValidationError],
) -> None:
    """Find dependency cycles via iterative DFS over the single adjacency.

    Ambiguous ids and already-rejected edges are excluded so cycles are only
    reported on topology that actually resolves.
    """
    valid_ids = {cid for cid in id_index if cid not in ambiguous_ids}
    graph: Dict[str, Set[str]] = {}
    for component in components:
        if component.id not in valid_ids:
            continue
        graph[component.id] = {
            dep
            for dep in adjacency.get(component.order, set())
            if dep in valid_ids and dep != component.id
        }

    white, gray, black = 0, 1, 2
    color: Dict[str, int] = {cid: white for cid in valid_ids}
    cycles: Set[frozenset] = set()

    def visit(start: str) -> None:
        stack: List[Tuple[str, Iterator[str]]] = [
            (start, iter(sorted(graph[start])))
        ]
        color[start] = gray
        path = [start]
        while stack:
            node, node_iter = stack[-1]
            advanced = False
            for neighbor in node_iter:
                if color[neighbor] == gray:
                    # Back edge: everything from neighbor onward in the path
                    # participates in the cycle.
                    cycles.add(frozenset(path[path.index(neighbor):]))
                elif color[neighbor] == white:
                    color[neighbor] = gray
                    path.append(neighbor)
                    stack.append((neighbor, iter(sorted(graph[neighbor]))))
                    advanced = True
                    break
            if not advanced:
                color[node] = black
                stack.pop()
                path.pop()

    for cid in sorted(valid_ids):
        if color[cid] == white:
            visit(cid)

    for cycle in sorted(cycles, key=lambda members: sorted(members)):
        cycle_ids = sorted(cycle)
        rendered = " -> ".join(cycle_ids + [cycle_ids[0]])
        for cid in cycle_ids:
            component = id_index[cid]
            errors.append(
                _ValidationError(
                    component,
                    ErrorType.dependency_cycle,
                    f"{component.label}: component is part of a dependency "
                    f"cycle: {rendered}",
                )
            )


def validate(
    config_path: Path = Argument(
        None,
        exists=True,
        help="Path to the pipeline configuration yaml to validate (graph.yml)",
        show_default=False,
    ),
):
    """Validate a pipeline configuration yaml offline

    Checks component ids, types, and dependency declarations without logging in
    or contacting the Patterns server. Exits non-zero when the configuration
    contains errors.
    """
    # Fully offline: don't even phone home for a devkit version check.
    versions.DISABLE_VERSION_CHECK = True

    try:
        config = load_yaml(config_path)
    except yaml.YAMLError:
        # Keep raw parser stack traces out of the user-facing output.
        abort(f"Invalid yaml in '{config_path}': could not parse the file")

    errors = validate_graph_config(config)
    if not errors:
        sprint(f"[success]Configuration '{config_path}' is valid")
        return

    sprint(f"[error]Configuration '{config_path}' has {len(errors)} error(s):")
    for error in errors:
        sprint(f"  [error]- {error.message}")
    raise typer.Exit(1)
