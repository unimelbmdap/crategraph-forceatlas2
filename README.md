# crategraph-forceatlas2

A fast ForceAtlas2 graph layout for Python, implemented as a Rust port of the JavaScript
[`graphology-layout-forceatlas2`](https://github.com/graphology/graphology/tree/master/src/layout-forceatlas2)
package, with credit to its original authors.

The ForceAtlas2 algorithm is described in
[Jacomy M, Venturini T, Heymann S, Bastian M (2014). "ForceAtlas2, a Continuous
Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi
Software." PLoS ONE 9(6): e98679](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679).

## Development

```bash
# Build the Rust extension in place, into the active Python environment
maturin develop

# Run the Rust test suite
cargo test --workspace

# Run the Python test suite
pytest tests/
```
