"""End-to-end JS parity: crategraph_forceatlas2.layout() vs the real
graphology-layout-forceatlas2 (via tests/node_reference/reference.mjs), run
on identical payloads (same graph, same seeded init, same settings, same
iteration count).

Both pipelines store node state in f32 (see Task 3/8 notes), and Task 3
established that every settings profile this project ships uses libm::pow's
exactly-bit-matching special-cased paths (y in {0, 0.5, 1, 2}); the only known
1-ULP divergence from V8 is in pow's general path, which infer_settings()
never reaches. So the assertion here is EXACT equality (np.array_equal), not
a tolerance -- see kernel bit-parity gate 10756aa (Task 8, 16/16 fixtures
bit-exact vs graphology 0.10.1) for the underlying guarantee this test is
extending end-to-end through the public Python API.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import crategraph_forceatlas2 as cfa2

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "node_reference" / "reference.mjs"
_NODE_MODULES = _HERE / "node_reference" / "node_modules"
_REQUIRED = ("graphology", "graphology-layout-forceatlas2")

_node_absent = (
    shutil.which("node") is None
    or not _SCRIPT.exists()
    or any(not (_NODE_MODULES / m).exists() for m in _REQUIRED)
)
requires_node = pytest.mark.skipif(
    _node_absent,
    reason="node or graphology deps absent (run `npm ci` in tests/node_reference)",
)


def _random_undirected_edges(n_nodes: int, n_edges: int, seed: int) -> list[list[int]]:
    """A fixed, deduped, self-loop-free (u < v) random edge set. Dedup matters
    because graphology's mergeUndirectedEdge (used by reference.mjs) silently
    dedupes parallel edges, while the Rust matrix builder does not -- a raw
    duplicate would make the two pipelines see different edge counts for
    "the same" payload."""
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    edges: list[list[int]] = []
    while len(edges) < n_edges:
        u, v = (int(x) for x in rng.integers(0, n_nodes, size=2))
        if u == v:
            continue
        if u > v:
            u, v = v, u
        key = (u, v)
        if key in seen:
            continue
        seen.add(key)
        edges.append([u, v])
    return edges


def _run_reference(payload: dict, tmp_path: Path) -> np.ndarray:
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps(payload))
    out = subprocess.run(
        ["node", str(_SCRIPT), str(payload_file)],
        capture_output=True, check=True, timeout=120,
    ).stdout.decode()
    return np.array(json.loads(out), dtype=np.float64)


@requires_node
def test_parity_small_random_graph(tmp_path):
    n_nodes = 200
    edges = _random_undirected_edges(n_nodes, n_edges=600, seed=1)
    init = np.random.default_rng(42).random((n_nodes, 2))
    settings = cfa2.infer_settings(n_nodes)
    assert settings["barnesHutOptimize"] is False  # exercises the direct-repulsion path
    iterations = 60

    expected = _run_reference(
        dict(n_nodes=n_nodes, edges=edges, init=init.tolist(),
             iterations=iterations, settings=settings),
        tmp_path,
    )
    actual = cfa2.layout(
        n_nodes, edges, init=init, iterations=iterations, threads=1, **settings,
    )
    assert np.array_equal(actual, expected)


@requires_node
def test_parity_barnes_hut_large_random_graph(tmp_path):
    n_nodes = 2500
    edges = _random_undirected_edges(n_nodes, n_edges=6000, seed=2)
    init = np.random.default_rng(42).random((n_nodes, 2))
    settings = cfa2.infer_settings(n_nodes)
    assert settings["barnesHutOptimize"] is True  # exercises the Barnes-Hut path
    iterations = 30

    expected = _run_reference(
        dict(n_nodes=n_nodes, edges=edges, init=init.tolist(),
             iterations=iterations, settings=settings),
        tmp_path,
    )
    actual = cfa2.layout(
        n_nodes, edges, init=init, iterations=iterations, threads=1, **settings,
    )
    assert np.array_equal(actual, expected)
