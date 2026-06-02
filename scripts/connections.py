"""Swanson 2-hop connection discovery for the knowledge base.

Builds the concept wikilink graph, finds non-obvious 2-hop bridge candidates,
filters out hub-driven noise, scores by bridge rarity, and (in a full run)
asks an LLM to confirm and write only genuine connections.
"""

from __future__ import annotations

import re
from pathlib import Path

CONCEPT_LINK_RE = re.compile(r"\[\[concepts/([a-z0-9-]+)\]\]")


def build_graph(concepts_dir: Path) -> dict[str, set[str]]:
    """Build an undirected adjacency map from concept [[concepts/slug]] links.

    Every concept file becomes a node (even with no links). Self-links are
    ignored. Linked-but-fileless targets still become nodes (valid bridges).
    """
    adj: dict[str, set[str]] = {}
    for path in sorted(concepts_dir.glob("*.md")):
        slug = path.stem
        adj.setdefault(slug, set())
        text = path.read_text(encoding="utf-8")
        for m in CONCEPT_LINK_RE.finditer(text):
            tgt = m.group(1)
            if tgt != slug:
                adj[slug].add(tgt)
                adj.setdefault(tgt, set()).add(slug)
    return adj
