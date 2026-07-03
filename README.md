# crategraph-forceatlas2

A fast, MIT-licensed ForceAtlas2 graph layout for Python, implemented as a
faithful Rust port of the JavaScript
[`graphology-layout-forceatlas2`](https://github.com/graphology/graphology-layout-forceatlas2)
package, with credit to its original authors. It exists to provide a
ForceAtlas2 implementation that is not encumbered by the GPL licence of the
commonly used `fa2` Python package. This package is not yet published.

## Development

```bash
# Build the Rust extension in place, into the active Python environment
maturin develop

# Run the Rust test suite
cargo test --workspace

# Run the Python test suite
pytest tests/
```
