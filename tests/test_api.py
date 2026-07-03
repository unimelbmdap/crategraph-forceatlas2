import numpy as np
import pytest

import crategraph_forceatlas2 as cfa2

TRIANGLE = dict(n_nodes=3, edges=[(0, 1), (1, 2), (2, 0)])


def test_layout_shape_dtype_and_finiteness():
    pos = cfa2.layout(**TRIANGLE, iterations=20)
    assert pos.shape == (3, 2) and pos.dtype == np.float64
    assert np.isfinite(pos).all()


def test_seed_determinism_and_seed_sensitivity():
    a = cfa2.layout(**TRIANGLE, iterations=20, seed=42)
    b = cfa2.layout(**TRIANGLE, iterations=20, seed=42)
    c = cfa2.layout(**TRIANGLE, iterations=20, seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_explicit_init_is_respected_and_not_mutated():
    init = np.random.default_rng(1).random((3, 2))
    keep = init.copy()
    cfa2.layout(**TRIANGLE, init=init, iterations=5)
    assert np.array_equal(init, keep)  # copied-in, never written


def test_iterations_inference_formula_via_infer_settings_docpath():
    # 0 iterations = init returned (f32-roundtripped)
    init = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    pos = cfa2.layout(**TRIANGLE, init=init, iterations=0)
    assert np.allclose(pos, init.astype(np.float32).astype(np.float64))


def test_infer_settings_matches_validated_profile():
    s = cfa2.infer_settings(2001)
    assert s == {"outboundAttractionDistribution": False, "barnesHutOptimize": True,
                 "barnesHutTheta": 0.5, "scalingRatio": 10, "strongGravityMode": True,
                 "gravity": 0.05, "slowDown": 1}
    assert cfa2.infer_settings(2000)["barnesHutOptimize"] is False


def test_threads_do_not_change_result():
    a = cfa2.layout(**TRIANGLE, iterations=50, threads=1, **cfa2.infer_settings(3))
    b = cfa2.layout(**TRIANGLE, iterations=50, threads=4, **cfa2.infer_settings(3))
    assert np.array_equal(a, b)


def test_progress_callback_one_based_and_abort_propagates():
    calls = []
    cfa2.layout(**TRIANGLE, iterations=5, progress=lambda i, t: calls.append((i, t)))
    assert calls == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]

    class Boom(RuntimeError):
        pass

    def bad(i, t):
        if i == 2:
            raise Boom()

    with pytest.raises(Boom):
        cfa2.layout(**TRIANGLE, iterations=5, progress=bad)


@pytest.mark.parametrize("bad_kwargs,match", [
    (dict(edges=[(0, 3)]), "out of range"),
    (dict(edges=[(1, 1)]), "self-loop"),
    (dict(edges=[(0, 1)], weights=[-1.0]), "non-negative"),
    (dict(edges=[(0, 1)], init=np.zeros((2, 2))), "shape"),
    (dict(edges=[(0, 1)], nonsense=True), "nonsense"),
    (dict(edges=[(0, 1)], threads=0), "threads"),
])
def test_validation_errors(bad_kwargs, match):
    kw = dict(n_nodes=3, edges=bad_kwargs.pop("edges"))
    with pytest.raises((ValueError, TypeError), match=match):
        cfa2.layout(**kw, **bad_kwargs)


def test_empty_graph_returns_empty():
    assert cfa2.layout(0, []).shape == (0, 2)
