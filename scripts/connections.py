"""Swanson 2-hop connection discovery for the knowledge base.

Builds the concept wikilink graph, finds non-obvious 2-hop bridge candidates,
filters out hub-driven noise, scores by bridge rarity, and (in a full run)
asks an LLM to confirm and write only genuine connections.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from config import AGENTS_FILE, CONCEPTS_DIR, CONNECTIONS_DIR, KNOWLEDGE_DIR, now_iso
from utils import notify_terminal, read_wiki_index

CONCEPT_LINK_RE = re.compile(r"\[\[concepts/([a-z0-9-]+)\]\]")
CONNECTS_RE = re.compile(r'"concepts/([a-z0-9-]+)"')

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = Path(__file__).resolve().parent / "compile.log"

logger = logging.getLogger("connections")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_fh)


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


def _format_candidates(cands: list[Candidate]) -> str:
    lines = []
    for i, cand in enumerate(cands, 1):
        lines.append(
            f"{i}. [[concepts/{cand.a}]] <-> [[concepts/{cand.c}]]  "
            f"(shared bridges: {', '.join(cand.bridges)})"
        )
    return "\n".join(lines)


async def synthesize_connections(cands: list[Candidate]) -> float:
    """Ask the LLM to confirm and write only genuine connections.

    Returns the API cost. The model is instructed to be conservative: write a
    connection article only when a specific, non-obvious, reusable insight links
    the pair, and to reject co-occurrence-only pairs with a one-line reason.

    The caller is responsible for bounding `cands` (main passes cands[:top]).
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    if not cands:
        print("No candidates to synthesize.")
        return 0.0

    schema = AGENTS_FILE.read_text(encoding="utf-8")
    wiki_index = read_wiki_index()
    timestamp = now_iso()

    prompt = f"""You are a connection synthesizer for a personal knowledge base.
Your job is to evaluate candidate relationships between EXISTING concept articles
and write connection articles ONLY for the genuine, non-obvious ones.

## Schema (AGENTS.md)

{schema}

## Current Wiki Index

{wiki_index}

## Candidate Pairs (Swanson 2-hop bridges, hub-filtered)

Each pair below was found because the two concepts are not directly linked but
share specific intermediate concepts (bridges). A shared bridge is a HINT, not
proof. Many candidates are mere co-occurrence (concepts that appeared in the same
line of work but share no transferable idea).

{_format_candidates(cands)}

## Your Task

For EACH candidate pair:
1. `Read` both concept articles (and a bridge if helpful) to understand them.
2. Decide: is there a SPECIFIC, NON-OBVIOUS, REUSABLE insight that links them?
   - YES: write a connection article in `knowledge/connections/` using the exact
     Connection Article format from the schema (frontmatter with `title`,
     `connects:` listing both `concepts/<slug>`, `project:` as a YAML block list
     (e.g. `project:` on its own line then `  - <project-slug>` lines), `sources:`,
     `created`/`updated` set to {timestamp[:10]}; body sections: The Connection,
     Key Insight, Evidence, Related Concepts with [[wikilinks]]).
   - NO: do not write anything; record a one-line rejection reason instead.

Be conservative. When in doubt, REJECT. A co-occurrence within one project or
initiative is NOT a connection. Only write when the relationship would teach a
reader something they could not get from either article alone.

### Mandatory bookkeeping before you stop:
- For every connection article you create, add a row to `knowledge/index.md`
  (Article | Project | Summary | Compiled From | Updated).
- Append ONE entry to `knowledge/log.md`:
  ```
  ## [{timestamp}] connections | swanson-pass
  - Candidates evaluated: <n>
  - Connections created: [[connections/x]], [[connections/y]] (or: none)
  - Rejected (co-occurrence / too weak): <pair> - <reason>; <pair> - <reason>
  ```
  This log entry is REQUIRED even if you create zero connections.

### File paths:
- Write connection articles to: {CONNECTIONS_DIR}
- Update index at: {KNOWLEDGE_DIR / 'index.md'}
- Append log at: {KNOWLEDGE_DIR / 'log.md'}
"""

    def _on_stderr(line: str) -> None:
        logger.debug("[cli stderr] %s", line.rstrip())

    log_md_path = KNOWLEDGE_DIR / "log.md"
    log_md_before = log_md_path.stat().st_size if log_md_path.exists() else 0

    cost = 0.0
    notify_terminal(f"connections pass started ({len(cands)} candidates)")
    logger.info("Begin connections pass (%d candidates)", len(cands))
    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=str(ROOT_DIR),
                system_prompt={"type": "preset", "preset": "claude_code"},
                allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
                permission_mode="bypassPermissions",
                max_turns=80,  # higher than compile.py (60): a pass evaluates many pairs, each needing 2+ Reads
                stderr=_on_stderr,
                extra_args={"debug-to-stderr": None},
                setting_sources=[],
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        pass  # model writes files directly
            elif isinstance(message, ResultMessage):
                cost = message.total_cost_usd or 0.0
                print(f"  Cost: ${cost:.4f}")
                logger.info("connections pass cost=$%.4f", cost)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  Error in connections pass: {e}")
        logger.error("Exception in connections pass: %s\n%s", e, tb)
        notify_terminal(f"connections pass failed: {e}")
        return 0.0

    log_md_after = log_md_path.stat().st_size if log_md_path.exists() else 0
    if log_md_after <= log_md_before:
        print("  Warning: knowledge/log.md did not grow; the gate may have skipped its required audit entry.")
        logger.warning("connections pass did not append to log.md (size %d -> %d)", log_md_before, log_md_after)
        notify_terminal("connections pass: WARNING log.md not updated (audit entry missing)")

    notify_terminal(f"connections pass complete (${cost:.4f})")
    return cost


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
