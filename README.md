# crategraph-forceatlas2

A fast ForceAtlas2 graph layout for Python, implemented as a Rust port of the JavaScript
[`graphology-layout-forceatlas2`](https://github.com/graphology/graphology/tree/master/src/layout-forceatlas2)
package, with credit to its original authors.

The ForceAtlas2 algorithm is described in
[Jacomy M, Venturini T, Heymann S, Bastian M (2014). "ForceAtlas2, a Continuous
Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi
Software." PLoS ONE 9(6): e98679](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679).

The initial development of this repository was carried out with the assistance of the **Fable 5**
Anthropic model. See [AGENTS.md](AGENTS.md) for an overview of the repository aimed at coding agents.

## Quick start

```bash
pip install crategraph-forceatlas2
```

```python
import crategraph_forceatlas2 as cfa2

edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
positions = cfa2.layout(4, edges)  # (4, 2) numpy array of x, y coordinates
```

Settings use graphology's camelCase names, e.g.
`cfa2.layout(4, edges, iterations=100, gravity=0.05, scalingRatio=10)`.
Layouts are deterministic; pass `seed=` to vary them.

## Development

```bash
# Build the Rust extension in place, into the active Python environment
maturin develop

# Run the Rust test suite
cargo test --workspace

# Run the Python test suite
pytest tests/
```
