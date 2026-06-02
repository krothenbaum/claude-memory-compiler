"""Swanson 2-hop connection discovery for the knowledge base.

Builds the concept wikilink graph, finds non-obvious 2-hop bridge candidates,
filters out hub-driven noise, scores by bridge rarity, and (in a full run)
asks an LLM to confirm and write only genuine connections.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from config import CONCEPTS_DIR, CONNECTIONS_DIR

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


@dataclass
class Candidate:
    """A candidate connection. Fields `a` and `c` satisfy a < c (lexicographic)."""
    a: str
    c: str
    bridges: list[str]
    score: float


def candidate_pairs(
    adj: dict[str, set[str]],
    existing: set[frozenset[str]],
    *,
    hub_degree: int = 12,
    min_bridges: int = 2,
) -> list[Candidate]:
    """Generate Swanson 2-hop connection candidates, ranked by bridge rarity.

    A candidate is a pair (a, c) that is NOT directly linked, NOT already
    connected, and shares at least `min_bridges` common neighbors after dropping
    bridges and endpoints whose degree exceeds `hub_degree`. Score sums each
    bridge's rarity (1 / log2(degree + 2)), so links through specific, low-degree
    concepts rank above links through generic hubs. Sorted descending by score.
    """
    deg = {n: len(neighbors) for n, neighbors in adj.items()}
    shared: dict[tuple[str, str], list[str]] = defaultdict(list)
    for b, neighbors in adj.items():
        if deg[b] > hub_degree:
            continue  # hub bridge: connects everything to everything, low signal
        for a, c in combinations(sorted(neighbors), 2):
            if c in adj.get(a, set()):
                continue  # directly linked -> obvious, not a discovery
            if frozenset((a, c)) in existing:
                continue  # already has a connection article
            shared[(a, c)].append(b)

    candidates: list[Candidate] = []
    for (a, c), bridges in shared.items():
        if len(bridges) < min_bridges:
            continue
        if deg.get(a, 0) > hub_degree or deg.get(c, 0) > hub_degree:
            continue  # hub endpoint: too generic to yield a specific connection
        score = sum(1 / math.log2(deg[b] + 2) for b in bridges)
        candidates.append(Candidate(a=a, c=c, bridges=sorted(bridges), score=score))

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates


def build_candidates(
    hub_degree: int = 12, min_bridges: int = 2
) -> list[Candidate]:
    """Build ranked candidates from the live knowledge base on disk."""
    adj = build_graph(CONCEPTS_DIR)
    existing = load_existing_pairs(CONNECTIONS_DIR)
    return candidate_pairs(
        adj, existing, hub_degree=hub_degree, min_bridges=min_bridges
    )


def _print_candidates(cands: list[Candidate], top: int) -> None:
    print(f"{len(cands)} candidate pairs (showing top {min(top, len(cands))}):\n")
    for cand in cands[:top]:
        print(f"[{cand.score:.2f}] {cand.a}  <->  {cand.c}")
        print(f"        via: {', '.join(cand.bridges)}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Swanson 2-hop connection discovery")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print ranked candidates without calling the LLM")
    parser.add_argument("--top", type=int, default=15,
                        help="Max candidates to send to the gate (full run) or display (dry-run) (default 15)")
    parser.add_argument("--hub-degree", type=int, default=12,
                        help="Bridges/endpoints above this degree are dropped (default 12)")
    parser.add_argument("--min-bridges", type=int, default=2,
                        help="Minimum shared bridges required (default 2)")
    args = parser.parse_args()

    cands = build_candidates(hub_degree=args.hub_degree, min_bridges=args.min_bridges)

    if args.dry_run:
        _print_candidates(cands, args.top)
        return

    import asyncio
    asyncio.run(synthesize_connections(cands[: args.top]))


if __name__ == "__main__":
    main()
