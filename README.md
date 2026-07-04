# crategraph-forceatlas2

A fast ForceAtlas2 graph layout for Python, implemented as a Rust port of the JavaScript
[`graphology-layout-forceatlas2`](https://github.com/graphology/graphology/tree/master/src/layout-forceatlas2)
package, with credit to its original authors.

## Development

```bash
# Build the Rust extension in place, into the active Python environment
maturin develop

# Run the Rust test suite
cargo test --workspace

# Run the Python test suite
pytest tests/
```
