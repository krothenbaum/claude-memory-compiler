"""Swanson 2-hop connection discovery for the knowledge base.

Builds the concept wikilink graph, finds non-obvious 2-hop bridge candidates,
filters out hub-driven noise, scores by bridge rarity, and (in a full run)
asks an LLM to confirm and write only genuine connections.
"""

from __future__ import annotations

import re
from pathlib import Path

CONCEPT_LINK_RE = re.compile(r"\[\[concepts/([a-z0-9-]+)\]\]")
CONNECTS_RE = re.compile(r'"concepts/([a-z0-9-]+)"')


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


def load_existing_pairs(connections_dir: Path) -> set[frozenset[str]]:
    """Return the set of concept pairs that already have a connection article.

    Reads the `connects:` slugs from each connection article's frontmatter and
    records every pairwise combination, so the candidate generator can skip
    pairs that are already connected.
    """
    pairs: set[frozenset[str]] = set()
    if not connections_dir.exists():
        return pairs
    for path in sorted(connections_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # startswith("---") guarantees split yields >=2 parts, so [1] is safe;
        # a missing closing --- just makes [1] the whole body, which is harmless.
        head = text.split("---")[1] if text.startswith("---") else ""
        members = sorted(set(CONNECTS_RE.findall(head)))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add(frozenset((members[i], members[j])))
    return pairs
