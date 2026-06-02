import connections


def test_build_graph_is_symmetric_and_excludes_self(kb):
    adj = connections.build_graph(kb["concepts"])
    assert adj["a"] == {"b", "c"}
    assert "a" in adj["b"]          # symmetric: a->b implies b->a
    assert "a" in adj["c"]
    assert adj["hub"] == {"e", "f", "g", "h"}
    assert adj["d"] == {"b", "c"}
    assert "a" not in adj["a"]      # no self-links
    # every concept file is a node, even leaf nodes
    assert set(adj) == {"a", "b", "c", "d", "hub", "e", "f", "g", "h"}


def test_load_existing_pairs_reads_connects_frontmatter(kb):
    conn = kb["connections"] / "b-and-c.md"
    conn.write_text(
        "---\n"
        'title: "Connection: B and C"\n'
        "connects:\n"
        '  - "concepts/b"\n'
        '  - "concepts/c"\n'
        "---\n\n# Connection: B and C\n",
        encoding="utf-8",
    )
    pairs = connections.load_existing_pairs(kb["connections"])
    assert frozenset(("b", "c")) in pairs
    assert frozenset(("a", "d")) not in pairs


def test_load_existing_pairs_empty_when_no_connections(kb):
    assert connections.load_existing_pairs(kb["connections"]) == set()


def test_load_existing_pairs_handles_single_quoted_slugs(kb):
    conn = kb["connections"] / "a-and-d.md"
    conn.write_text(
        "---\n"
        'title: "Connection: A and D"\n'
        "connects:\n"
        "  - 'concepts/a'\n"
        "  - 'concepts/d'\n"
        "---\n\n# Connection: A and D\n",
        encoding="utf-8",
    )
    pairs = connections.load_existing_pairs(kb["connections"])
    assert frozenset(("a", "d")) in pairs


def _pair_set(cands):
    return {frozenset((c.a, c.c)) for c in cands}


def test_candidate_excludes_adjacent_and_finds_bridged_pair(kb):
    adj = connections.build_graph(kb["concepts"])
    cands = connections.candidate_pairs(adj, set(), hub_degree=12, min_bridges=2)
    # (b, c) are not directly linked but share bridges a and d -> a candidate
    assert frozenset(("b", "c")) in _pair_set(cands)
    # (a, b) ARE directly linked -> never a candidate
    assert frozenset(("a", "b")) not in _pair_set(cands)


def test_candidate_excludes_existing_pairs(kb):
    adj = connections.build_graph(kb["concepts"])
    existing = {frozenset(("b", "c"))}
    cands = connections.candidate_pairs(adj, existing, hub_degree=12, min_bridges=2)
    assert frozenset(("b", "c")) not in _pair_set(cands)


def test_candidate_drops_hub_bridges(kb):
    # hub has degree 4. With hub_degree=3 it is dropped as a bridge, so the
    # hub-only pairs (e,f),(e,g),... disappear.
    adj = connections.build_graph(kb["concepts"])
    cands = connections.candidate_pairs(adj, set(), hub_degree=3, min_bridges=1)
    assert frozenset(("e", "f")) not in _pair_set(cands)


def test_candidate_requires_min_bridges(kb):
    adj = connections.build_graph(kb["concepts"])
    # With min_bridges=2 and hub allowed, hub-only pairs (1 bridge) are excluded.
    cands = connections.candidate_pairs(adj, set(), hub_degree=12, min_bridges=2)
    assert frozenset(("e", "f")) not in _pair_set(cands)


def test_candidate_scores_specific_bridges_higher():
    # specific (low-degree) bridges should outscore a single high-degree bridge.
    adj = {
        "x": {"p", "q"}, "y": {"p", "q"}, "p": {"x", "y"}, "q": {"x", "y"},
        "m": {"hh"}, "n": {"hh"},
        "hh": {"m", "n", "r", "s", "t", "u", "v", "w"},
        "r": {"hh"}, "s": {"hh"}, "t": {"hh"}, "u": {"hh"}, "v": {"hh"}, "w": {"hh"},
    }
    cands = connections.candidate_pairs(adj, set(), hub_degree=99, min_bridges=1)
    by_pair = {frozenset((c.a, c.c)): c.score for c in cands}
    assert by_pair[frozenset(("x", "y"))] > by_pair[frozenset(("m", "n"))]
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)


def test_candidate_bridges_are_sorted(kb):
    adj = connections.build_graph(kb["concepts"])
    cands = connections.candidate_pairs(adj, set(), hub_degree=12, min_bridges=2)
    bc = next(c for c in cands if frozenset((c.a, c.c)) == frozenset(("b", "c")))
    assert bc.bridges == ["a", "d"]
