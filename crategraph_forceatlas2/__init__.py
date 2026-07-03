"""Faithful Rust port of graphology-layout-forceatlas2.

Exposes `layout()`, the public entry point that validates its inputs and
delegates to the compiled `_native.layout_raw` binding, and
`infer_settings()`, a sane-defaults helper for picking a settings profile
from a graph's node count.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from crategraph_forceatlas2 import _native

# Mirrors `cfa2_kernel::settings::Settings` field-for-field (camelCase, the
# convention graphology-layout-forceatlas2 itself uses, so callers can reuse
# settings dicts written for the JS library).
VALID_SETTINGS = frozenset(
    {
        "linLogMode",
        "outboundAttractionDistribution",
        "adjustSizes",
        "edgeWeightInfluence",
        "scalingRatio",
        "strongGravityMode",
        "gravity",
        "slowDown",
        "barnesHutOptimize",
        "barnesHutTheta",
    }
)


def infer_settings(n_nodes: int) -> dict:
    """Returns a validated, sane-defaults settings profile for a graph with
    `n_nodes` nodes: strong gravity with `gravity=0.05` and
    `scalingRatio=10`, Barnes-Hut approximation switched on once the graph
    is large enough (past 2000 nodes) to benefit from it.
    """
    return {
        "outboundAttractionDistribution": False,
        "barnesHutOptimize": n_nodes > 2000,
        "barnesHutTheta": 0.5,
        "scalingRatio": 10,
        "strongGravityMode": True,
        "gravity": 0.05,
        "slowDown": 1,
    }


def layout(
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
    *,
    init: np.ndarray | None = None,
    weights: Sequence[float] | None = None,
    iterations: int | None = None,
    seed: int = 42,
    threads: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    **settings: object,
) -> np.ndarray:
    """Runs ForceAtlas2 and returns the final `(n_nodes, 2)` float64
    positions.

    `edges` is a sequence of `(source, target)` node-index pairs, each in
    `[0, n_nodes)`, with no self-loops. `weights` (optional) is a matching
    per-edge sequence of finite, non-negative floats, defaulting to `1.0`.
    `init` (optional) is an `(n_nodes, 2)` array of finite starting
    coordinates; without it, `np.random.default_rng(seed).random((n_nodes,
    2))` is used. `iterations` defaults to `min(200, 50 + n_nodes // 100)`
    when omitted; `threads` (>=1) selects how many threads parallelise the
    Barnes-Hut repulsion phase, defaulting to `1` (sequential). `progress`,
    if given, is called after every iteration as `progress(i, iterations)`
    (1-based `i`); raising from it aborts the layout and propagates the
    exception. Remaining keyword arguments are ForceAtlas2 settings, named
    per graphology's camelCase convention (see `VALID_SETTINGS`).
    """
    unknown = set(settings) - VALID_SETTINGS
    if unknown:
        raise ValueError(
            f"unknown settings: {', '.join(sorted(unknown))}; "
            f"valid settings are: {', '.join(sorted(VALID_SETTINGS))}"
        )

    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")
    if n_nodes == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if len(edges) == 0:
        edges_arr = np.zeros((0, 2), dtype=np.int64)
    else:
        edges_arr = np.asarray(edges, dtype=np.int64)
        if edges_arr.ndim != 2 or edges_arr.shape[1] != 2:
            raise ValueError(f"edges must have shape (m, 2), got {edges_arr.shape}")
        if edges_arr.min() < 0 or edges_arr.max() >= n_nodes:
            raise ValueError(
                f"edge endpoint out of range: indices must be in [0, {n_nodes}), "
                f"got an index range of [{edges_arr.min()}, {edges_arr.max()}]"
            )
        if np.any(edges_arr[:, 0] == edges_arr[:, 1]):
            raise ValueError("edges must not contain self-loops")
    edges_u32 = edges_arr.astype(np.uint32)
    n_edges = edges_arr.shape[0]

    if weights is None:
        weights_arr = None
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape != (n_edges,):
            raise ValueError(f"weights must have shape ({n_edges},), got {weights_arr.shape}")
        if not np.all(np.isfinite(weights_arr)) or np.any(weights_arr < 0):
            raise ValueError("weights must be finite and non-negative")

    if init is None:
        init_arr = np.random.default_rng(seed).random((n_nodes, 2))
    else:
        init_arr = np.asarray(init, dtype=np.float64)
        if init_arr.shape != (n_nodes, 2):
            raise ValueError(f"init must have shape ({n_nodes}, 2), got {init_arr.shape}")
        if not np.all(np.isfinite(init_arr)):
            raise ValueError("init must contain only finite values")

    if threads is None:
        threads = 1
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError(f"threads must be a positive integer, got {threads!r}")

    if iterations is None:
        iterations = min(200, 50 + n_nodes // 100)
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
        raise ValueError(f"iterations must be a non-negative integer, got {iterations!r}")

    return _native.layout_raw(
        n_nodes,
        edges_u32,
        weights_arr,
        init_arr,
        iterations,
        threads,
        dict(settings),
        progress,
    )
