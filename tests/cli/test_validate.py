from pathlib import Path

from tests.cli.base import request_mocker, run_cli, set_tmp_dir


def _write_graph(tmp_path: Path, text: str, name: str = "graph.yml") -> Path:
    set_tmp_dir(tmp_path)
    path = tmp_path / name
    path.write_text(text.lstrip())
    return path


def test_validate_valid_config(tmp_path: Path):
    path = _write_graph(
        tmp_path,
        """
title: demo
functions:
  - node_file: ingest.py
    id: ingest
    type: function
  - uses: patterns/component@v1
    id: comp
    dependencies: ingest
  - webhook: hook
    id: hook
    dependencies:
      - ingest
      - comp
stores:
  - table: things
    id: things
""",
    )
    # No mocked endpoints beyond the auth base: any network call would raise.
    with request_mocker():
        result = run_cli(f"validate {path}")
    assert result.exit_code == 0
    assert "is valid" in result.output
    assert "Traceback" not in result.output


def test_validate_unknown_type_lists_allowed_values(tmp_path: Path):
    path = _write_graph(
        tmp_path,
        """
functions:
  - node_file: ingest.py
    id: ingest
    type: not_a_real_type
""",
    )
    with request_mocker():
        result = run_cli(f"validate {path}")
    assert result.exit_code == 1
    assert "unknown component type 'not_a_real_type'" in result.output
    assert "allowed types are" in result.output
    for allowed in ("function", "component", "webhook", "table"):
        assert allowed in result.output
    # A type failure must not prevent dependency checks from running; here the
    # dependency section is simply absent, which must be treated as empty.
    assert "dependency" not in result.output


def test_validate_yaml_syntax_error_uses_cli_message(tmp_path: Path):
    path = _write_graph(
        tmp_path,
        """
functions:
  - id: ingest
   node_file: ingest.py
""",
    )
    with request_mocker():
        result = run_cli(f"validate {path}")
    assert result.exit_code == 1
    assert "Invalid yaml" in result.output
    assert "could not parse" in result.output
    # Raw parser stack traces must never reach the user.
    assert "Traceback" not in result.output


def test_validate_missing_dependency(tmp_path: Path):
    path = _write_graph(
        tmp_path,
        """
functions:
  - node_file: ingest.py
    id: ingest
    dependencies: ghost
""",
    )
    with request_mocker():
        result = run_cli(f"validate {path}")
    assert result.exit_code == 1
    assert "dependency 'ghost' does not match any component id" in result.output


def test_validate_self_dependency(tmp_path: Path):
    path = _write_graph(
        tmp_path,
        """
functions:
  - node_file: ingest.py
    id: ingest
    dependencies: [ingest]
""",
    )
    with request_mocker():
        result = run_cli(f"validate {path}")
    assert result.exit_code == 1
    assert "depends on itself" in result.output
    # A self dependency must not be misreported as a cycle.
    assert "cycle" not in result.output


def test_validate_duplicate_id_is_ambiguous(tmp_path: Path):
    path = _write_graph(
        tmp_path,
        """
functions:
  - node_file: a.py
    id: shared
  - node_file: b.py
    id: shared
  - node_file: c.py
    id: consumer
    dependencies: shared
""",
    )
    with request_mocker():
        result = run_cli(f"validate {path}")
    assert result.exit_code == 1
    # Both declarations are flagged, and the id is called out as ambiguous.
    assert result.output.count("duplicate component id 'shared'") == 2
    assert "id is ambiguous" in result.output
    assert "dependency 'shared' is ambiguous" in result.output
    # The ambiguous edge is excluded from the graph, so no fabricated cycle.
    assert "cycle" not in result.output


def test_validate_dependency_cycle(tmp_path: Path):
    path = _write_graph(
        tmp_path,
        """
functions:
  - node_file: a.py
    id: a
    dependencies: [b]
  - node_file: b.py
    id: b
    dependencies: [c]
  - node_file: c.py
    id: c
    dependencies: [a]
  - node_file: d.py
    id: d
""",
    )
    with request_mocker():
        result = run_cli(f"validate {path}")
    assert result.exit_code == 1
    assert "dependency cycle" in result.output
    # Every member of the cycle is reported, and the unrelated node is not.
    assert result.output.count("dependency cycle") == 3
    assert "function[3]" not in result.output


def test_validate_multiple_problems_have_stable_order(tmp_path: Path):
    path = _write_graph(
        tmp_path,
        """
functions:
  - node_file: a.py
    id: a
    type: bogus
    dependencies: [missing_node]
  - node_file: b.py
    id: b
    dependencies: a
  - node_file: a2.py
    id: a
  - node_file: c.py
    id: c
    dependencies: [c]
  - node_file: d.py
    id: d
    dependencies: [e]
  - node_file: e.py
    id: e
    dependencies: [d]
""",
    )
    with request_mocker():
        first = run_cli(f"validate {path}")
        second = run_cli(f"validate {path}")
    assert first.exit_code == 1
    assert second.exit_code == 1
    # Repeated runs must produce byte-identical, ordering-stable output.
    assert first.output == second.output

    # Rich wraps long lines to the console width, so normalize whitespace
    # before checking wording and relative order.
    normalized = " ".join(first.output.split())
    expected = [
        "function[0]: duplicate component id 'a' (id is ambiguous)",
        "function[0]: dependency 'missing_node' does not match any component id",
        "function[0]: unknown component type 'bogus'; allowed types are: "
        "function, component, webhook, table",
        "function[1]: dependency 'a' is ambiguous because the id is declared "
        "more than once",
        "function[2]: duplicate component id 'a' (id is ambiguous)",
        "function[3]: component depends on itself ('c')",
        "function[4]: component is part of a dependency cycle: d -> e -> d",
        "function[5]: component is part of a dependency cycle: d -> e -> d",
    ]
    positions = [normalized.index(message) for message in expected]
    assert positions == sorted(positions)
    assert normalized.count("dependency cycle: d -> e -> d") == 2


def test_validate_missing_dependencies_field_is_empty_set(tmp_path: Path):
    path = _write_graph(
        tmp_path,
        """
functions:
  - node_file: a.py
    id: a
stores:
  - table: t
    id: t
""",
    )
    with request_mocker():
        result = run_cli(f"validate {path}")
    assert result.exit_code == 0
    assert "is valid" in result.output


def test_validate_malformed_dependencies_is_skipped_not_fatal(tmp_path: Path):
    path = _write_graph(
        tmp_path,
        """
functions:
  - node_file: a.py
    id: a
    dependencies: 7
  - node_file: b.py
    id: b
    dependencies: [a]
""",
    )
    with request_mocker():
        result = run_cli(f"validate {path}")
    assert result.exit_code == 1
    assert "'dependencies' must be a component id string" in result.output
    # The valid edge on the other component is still analyzed.
    assert "Traceback" not in result.output


def test_validate_non_mapping_root(tmp_path: Path):
    path = _write_graph(tmp_path, "just a scalar string\n")
    with request_mocker():
        result = run_cli(f"validate {path}")
    assert result.exit_code == 1
    assert "Invalid graph configuration" in result.output
    assert "Traceback" not in result.output
