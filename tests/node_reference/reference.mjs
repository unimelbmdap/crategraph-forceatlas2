// End-to-end parity oracle: runs the REAL graphology-layout-forceatlas2 on a
// JSON payload and prints index-aligned [x, y] pairs as JSON on stdout. Test
// tests/test_parity.py builds the identical payload, runs it through this
// script AND through crategraph_forceatlas2.layout(), and asserts the
// results are bit-for-bit identical.
//
// Payload contract (copied from cdl2-rocrates'
// benchmarks/graphology_layout/layout.mjs, which is proven):
//   { n_nodes, edges, init, iterations, settings }
import { readFileSync } from "node:fs";
import Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";

const payloadPath = process.argv[2];
if (!payloadPath) {
  process.stderr.write("usage: node reference.mjs <payload.json>\n");
  process.exit(2);
}

const { n_nodes, edges, init, iterations, settings } = JSON.parse(
  readFileSync(payloadPath, "utf8"),
);

// Undirected graph; node keys are the integer indices (graphology stores them
// as strings, which is fine since we read back by the same index).
const graph = new Graph({ type: "undirected" });
for (let i = 0; i < n_nodes; i++) {
  graph.addNode(i, { x: init[i][0], y: init[i][1] });
}
for (const [u, v] of edges) {
  graph.mergeUndirectedEdge(u, v); // dedupes parallel/back edges
}

// ONE deliberate change from the cdl2-rocrates original: that script hardcodes
// `slowDown: 1` merged OVER the payload settings (`{ ...settings, slowDown: 1
// }`), which would silently clobber any slowDown the payload asked for. Here
// the payload settings are spread LAST so the payload fully controls the
// merge; `slowDown: 1` is only a fallback for a payload that omits it. In
// practice this repo's `infer_settings()` always includes `slowDown: 1`
// explicitly, so the fallback never fires, but the payload is authoritative.
const positions = forceAtlas2(graph, {
  iterations,
  settings: { slowDown: 1, ...settings },
});

const out = [];
for (let i = 0; i < n_nodes; i++) {
  out.push([positions[i].x, positions[i].y]);
}
process.stdout.write(JSON.stringify(out));
