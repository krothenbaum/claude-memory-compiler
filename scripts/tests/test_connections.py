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
