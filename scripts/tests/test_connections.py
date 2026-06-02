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
