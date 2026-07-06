# AGENTS.md

> This file helps AI coding agents understand the repository.
> It is agent-agnostic and should be kept in version control.

## Project Overview

`crategraph-forceatlas2` is a fast ForceAtlas2 graph-layout library for Python, implemented
as a faithful Rust port of the JavaScript
[`graphology-layout-forceatlas2`](https://github.com/graphology/graphology/tree/master/src/layout-forceatlas2)
package. A pure-Rust kernel does the numerical work; a PyO3 binding exposes it to Python; a thin
Python wrapper handles input validation and presents the public API. Parity with the reference
JS implementation is a core design goal and is enforced by the test suite.

## Tech Stack

- **Languages:** Rust (kernel + binding), Python 3.10+ (public API)
- **Build backend:** [maturin](https://www.maturin.rs/) (builds the Rust extension into a Python wheel)
- **Python↔Rust bridge:** [PyO3](https://pyo3.rs/) 0.22 (abi3-py310) + [rust-numpy](https://github.com/PyO3/rust-numpy) 0.22
- **Rust package manager:** Cargo (workspace with two member crates)
- **Python package manager:** uv (see `uv.lock`)
- **Key Rust dependencies:** `rayon` (parallel Barnes-Hut repulsion), `libm` (bit-exact math)
- **Key Python dependency:** `numpy>=1.24`
- **Parity reference:** Node.js + `graphology-layout-forceatlas2` (dev-only, under `tests/node_reference/`)

## Directory Structure

```
kernel/                     # cfa2-kernel: pure-Rust ForceAtlas2 implementation
  src/
    layout.rs               #   run_layout() — top-level driver
    iterate.rs              #   one simulation step (attraction, repulsion, gravity)
    quadtree.rs             #   Barnes-Hut quadtree for approximate repulsion
    matrices.rs             #   node/edge matrix layout + graph_to_matrices()
    js_math.rs              #   bit-exact reimplementations of JS Math functions
    settings.rs             #   Settings struct + defaults
  tests/parity.rs           #   Rust-side golden-fixture parity tests
bindings/                   # cfa2-bindings: PyO3 cdylib (_native.layout_raw)
  src/lib.rs
crategraph_forceatlas2/     # Python package
  __init__.py               #   layout(), infer_settings(), input validation
tests/                      # Python test suite
  test_api.py               #   public API behaviour + validation
  test_parity.py            #   parity against the JS reference
  node_reference/           #   Node.js fixtures + generators (dev-only)
.github/workflows/          # CI (ci.yml) and wheel/sdist release (wheels.yml)
```

## Module Guide

- **`kernel/` (`cfa2-kernel`)** — the algorithm. No Python awareness; operates on flat matrices.
  All numerical behaviour, including bit-exact JS math parity, lives here.
- **`bindings/` (`cfa2-bindings`)** — the `_native` extension module. Copies numpy arrays into
  Rust-owned buffers before releasing the GIL, runs the kernel, widens f32 results to f64. Trusts
  its inputs; does no validation.
- **`crategraph_forceatlas2/`** — the public Python API. `layout()` validates everything and
  delegates to `_native.layout_raw`; `infer_settings()` picks a sane settings profile by node count.

## Getting Started

```bash
# Build the Rust extension in place, into the active Python environment
maturin develop
```

## Testing

- **Location:** `tests/` (Python), `kernel/tests/` and inline `#[cfg(test)]` (Rust)
- **Run (Rust):** `cargo test --workspace`
- **Run (Python):** `pytest tests/`
- **Notes:** Parity is mandatory. `tests/test_parity.py` and `kernel/tests/parity.rs` compare
  output against golden fixtures generated from the JS reference in `tests/node_reference/`
  (regenerate via the `gen_*.mjs` scripts after `npm ci` there). `threads=1` reproduces the JS
  library bit-for-bit; the parallel path may reorder floating-point sums.

## Key Commands

| Task | Command |
| --- | --- |
| Build extension in place | `maturin develop` |
| Rust tests | `cargo test --workspace` |
| Rust lint (CI-enforced) | `cargo clippy --workspace --all-targets -- -D warnings` |
| Python tests | `pytest tests/` |
| Regenerate JS parity fixtures | `cd tests/node_reference && npm ci && node gen_kernel_fixtures.mjs` |

## Documentation

- `README.md` — quick start, development commands, algorithm citation.
- `LICENSE` — MIT.
- Rust module docstrings (`//!` headers) explain each kernel file and the binding's GIL discipline.
- Python docstrings in `crategraph_forceatlas2/__init__.py` document the public API and every argument.

## Architecture Notes

Call path: `crategraph_forceatlas2.layout()` (validate) → `_native.layout_raw` (marshal arrays,
release GIL) → `cfa2_kernel::run_layout` → per-iteration `iterate()` (attraction + Barnes-Hut
repulsion via `quadtree` + gravity). Settings use graphology's camelCase names so JS settings
dicts port directly. Determinism and JS parity are load-bearing: `js_math.rs` reimplements JS
`Math` behaviour bit-for-bit, and the golden-fixture tests guard against drift. Prefer changes
that preserve single-threaded parity with the reference implementation.

## Acknowledgements

The initial development of this repository was carried out with the assistance of the **Fable 5**
Anthropic model.
