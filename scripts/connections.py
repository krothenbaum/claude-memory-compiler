"""Swanson 2-hop connection discovery for the knowledge base.

Builds the concept wikilink graph, finds non-obvious 2-hop bridge candidates,
filters out hub-driven noise, scores by bridge rarity, and (in a full run)
asks an LLM to confirm and write only genuine connections.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from config import AGENTS_FILE, CONCEPTS_DIR, CONNECTIONS_DIR, KNOWLEDGE_DIR, load_config, now_iso
from providers import (
    ClaudeProvider,
    CodexProvider,
    ProviderResult,
    ProviderRouter,
    TaskKind,
    WorkspaceRequest,
)
from staging import (
    ApplyBookkeeping,
    StageValidationError,
    apply_validated_stage,
    create_fallback_stage,
    create_stage,
    discard_stage,
    validate_stage,
)
from utils import notify_terminal, read_wiki_index

CONCEPT_LINK_RE = re.compile(r"\[\[concepts/([a-z0-9-]+)\]\]")
CONNECTS_RE = re.compile(r'''["']concepts/([a-z0-9-]+)["']''')

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = Path(__file__).resolve().parent / "compile.log"

logger = logging.getLogger("connections")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_fh)


def _default_workspace_router(config, fallback_workspace_factory):
    return ProviderRouter(
        CodexProvider(task_models=config.task_models),
        ClaudeProvider(model=config.claude_model),
        fallback_workspace_factory=fallback_workspace_factory,
    )


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


def build_connections_prompt(
    cands: list[Candidate], schema: str, wiki_index: str, timestamp: str
) -> str:
    """Build the provider-neutral, stage-relative connection prompt."""
    return f"""You are a connection synthesizer for a personal knowledge base.
Evaluate the candidates conservatively and create a connection article only for
a specific, non-obvious, reusable insight.

## Schema

{schema}

## Current index

{wiki_index}

## Candidates

{_format_candidates(cands)}

For each candidate, read the staged concept articles. For accepted candidates:
- create an article below knowledge/connections/ using the schema;
- add its row to knowledge/index.md;
- cite it in one append-only build-log entry headed
  `## [{timestamp}] connections | swanson-pass`.

The log entry is required even if every candidate is rejected. Never edit concept
articles, AGENTS.md, daily sources, or scripts/state.json. All paths are relative
to this disposable staged workspace."""


async def synthesize_connections(
    cands: list[Candidate],
    *,
    router: object | None = None,
    router_factory: object | None = None,
    memory_home: Path | str | None = None,
) -> float:
    """Ask the LLM to confirm and write only genuine connections.

    Returns the API cost. The model is instructed to be conservative: write a
    connection article only when a specific, non-obvious, reusable insight links
    the pair, and to reject co-occurrence-only pairs with a one-line reason.

    The caller is responsible for bounding `cands` (main passes cands[:top]).
    """
    if not cands:
        print("No candidates to synthesize.")
        return 0.0

    home = Path(memory_home).expanduser().resolve() if memory_home is not None else ROOT_DIR
    schema = (home / "AGENTS.md").read_text(encoding="utf-8")
    wiki_index = (home / "knowledge/index.md").read_text(encoding="utf-8")
    timestamp = now_iso()
    prompt = build_connections_prompt(cands, schema, wiki_index, timestamp)
    relevant_slugs = sorted(
        {slug for cand in cands for slug in (cand.a, cand.c, *cand.bridges)}
    )
    relevant_paths = tuple(
        f"knowledge/concepts/{slug}.md"
        for slug in relevant_slugs
        if (home / f"knowledge/concepts/{slug}.md").exists()
    )
    stage = create_stage(
        home, "connections", "codex", relevant_articles=relevant_paths
    )
    allowed = (
        "knowledge/connections/*.md",
        "knowledge/index.md",
        "knowledge/log.md",
    )
    environment = dict(os.environ)
    environment["AI_MEMORY_HOME"] = str(home)
    environment.pop("CLAUDE_MEMORY_HOME", None)
    config = load_config(environment)
    fallback_holder = []

    def fallback_factory(request: WorkspaceRequest) -> WorkspaceRequest:
        fallback = create_fallback_stage(stage, attempt_id="claude")
        fallback_holder.append(fallback)
        return WorkspaceRequest(
            request.task, request.prompt, fallback.root, request.timeout_seconds,
            request.output_schema, request.allowed_paths,
        )

    if router is not None and router_factory is not None:
        raise ValueError("provide router or router_factory, not both")
    if router_factory is not None:
        provider_router = router_factory(fallback_factory)
    elif router is not None:
        provider_router = router
    else:
        provider_router = _default_workspace_router(config, fallback_factory)
    request = WorkspaceRequest(
        TaskKind.CONNECTIONS,
        prompt,
        stage.root,
        config.job_timeout_seconds,
        allowed_paths=allowed,
    )
    notify_terminal(f"connections pass started ({len(cands)} candidates)")
    logger.info("Begin connections pass (%d candidates)", len(cands))
    try:
        result = await provider_router.edit_workspace(request)
        if result.outcome != "success":
            discard_stage(fallback_holder[-1] if fallback_holder else stage)
            return 0.0
        selected = fallback_holder[-1] if result.provider == "claude" and fallback_holder else stage
        try:
            validated = validate_stage(
                selected,
                allowed_paths=allowed,
                task=TaskKind.CONNECTIONS,
                expected_candidate_count=len(cands),
            )
        except StageValidationError as validation_error:
            if result.provider != "codex" or (router is not None and router_factory is None):
                discard_stage(selected)
                return 0.0
            failed = ProviderResult(
                "codex", result.model, TaskKind.CONNECTIONS, "invalid_output",
                reason=str(validation_error),
            )
            result = await provider_router.edit_workspace(request, codex_attempt=failed)
            if result.outcome != "success" or not fallback_holder:
                if stage.root.exists():
                    discard_stage(stage)
                return 0.0
            selected = fallback_holder[-1]
            validated = validate_stage(
                selected,
                allowed_paths=allowed,
                task=TaskKind.CONNECTIONS,
                expected_candidate_count=len(cands),
            )
        apply_validated_stage(validated, validated.before, ApplyBookkeeping())
    except Exception as exc:
        logger.exception("connections provider failed")
        for candidate in [*fallback_holder, stage]:
            if candidate.root.exists():
                discard_stage(candidate)
        notify_terminal(f"connections pass failed: {exc}")
        return 0.0
    finally:
        for candidate in [*fallback_holder, stage]:
            if candidate.root.exists():
                discard_stage(candidate)
    notify_terminal("connections pass complete")
    return 0.0


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
