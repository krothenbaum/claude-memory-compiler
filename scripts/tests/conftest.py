"""Test config: put scripts/ on sys.path and provide a temp-KB fixture."""

import sys
from pathlib import Path

# scripts/ must be importable as top-level (modules use `from config import ...`)
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pytest


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


@pytest.fixture
def kb(tmp_path: Path):
    """A tiny knowledge base on disk.

    Graph (concept -> links):
      a -> b, c
      b -> a, d
      c -> a, d
      d -> b, c
      hub -> e, f, g, h  (high degree bridge)
      e -> hub ; f -> hub ; g -> hub ; h -> hub
    So: (b, c) share bridge a AND bridge d -> 2 specific bridges, not adjacent.
        (e, f), (e, g), ... share only the hub bridge.
    """
    concepts = tmp_path / "concepts"
    connections = tmp_path / "connections"
    connections.mkdir(parents=True, exist_ok=True)

    _write(concepts / "a.md", "links [[concepts/b]] [[concepts/c]]\n")
    _write(concepts / "b.md", "links [[concepts/a]] [[concepts/d]]\n")
    _write(concepts / "c.md", "links [[concepts/a]] [[concepts/d]]\n")
    _write(concepts / "d.md", "links [[concepts/b]] [[concepts/c]]\n")
    _write(concepts / "hub.md",
           "links [[concepts/e]] [[concepts/f]] [[concepts/g]] [[concepts/h]]\n")
    _write(concepts / "e.md", "links [[concepts/hub]]\n")
    _write(concepts / "f.md", "links [[concepts/hub]]\n")
    _write(concepts / "g.md", "links [[concepts/hub]]\n")
    _write(concepts / "h.md", "links [[concepts/hub]]\n")

    return {"root": tmp_path, "concepts": concepts, "connections": connections}
