// Runs graphology's iterate() N times on a payload and prints the NodeMatrix
// as u32 bit patterns (f32 bits) — the parity oracle for the Rust kernel.
import { readFileSync } from "node:fs";
import Graph from "graphology";
import helpers from "graphology-layout-forceatlas2/helpers.js";
import iterate from "graphology-layout-forceatlas2/iterate.js";
import DEFAULTS from "graphology-layout-forceatlas2/defaults.js";

const { n_nodes, edges, weights, init, settings, iterations } =
  JSON.parse(readFileSync(process.argv[2], "utf8"));

const graph = new Graph({ type: "undirected" });
for (let i = 0; i < n_nodes; i++) graph.addNode(i, { x: init[i][0], y: init[i][1] });
edges.forEach(([u, v], k) =>
  graph.mergeUndirectedEdge(u, v, weights ? { weight: weights[k] } : {}));

// Codex review fixes:
// - graphToByteArrays calls getEdgeWeight UNCONDITIONALLY (helpers.js:154), so
//   passing null crashes — unweighted fixtures need a constant-1 getter.
// - iterate() reads settings directly with NO defaults merge (index.js:55 does
//   the merge in normal use) — partial settings would produce NaN.
const getWeight = weights ? (_, attr) => attr.weight ?? 1 : () => 1;
const { nodes: NodeMatrix, edges: EdgeMatrix } = helpers.graphToByteArrays(
  graph, getWeight);
const merged = { ...DEFAULTS, ...settings };
for (let i = 0; i < iterations; i++) iterate(merged, NodeMatrix, EdgeMatrix);

const u32 = new Uint32Array(NodeMatrix.buffer);
process.stdout.write(JSON.stringify(Array.from(u32, (b) => b.toString())));
