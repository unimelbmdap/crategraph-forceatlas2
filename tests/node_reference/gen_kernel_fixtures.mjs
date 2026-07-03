// Dev-only: builds the bit-parity fixture set by running the REAL
// graphology-layout-forceatlas2 (via dump_state.mjs, spawned per fixture) on
// deterministic payloads, and writes fixtures/kernel_goldens.json. The Rust
// integration test (kernel/tests/parity.rs) replays the same payloads
// through the Rust kernel and asserts the resulting NodeMatrix is bit-for-
// bit identical (`to_bits()` on every f32 slot).
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { writeFileSync, unlinkSync } from "node:fs";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DUMP_STATE = path.join(__dirname, "dump_state.mjs");
const FIXTURES_DIR = path.join(__dirname, "fixtures");
const OUT_FILE = path.join(FIXTURES_DIR, "kernel_goldens.json");

// Small inline LCG (Numerical Recipes constants), seed 42. This is only used
// to build the deterministic fixture INPUTS (coords/edges/weights) that get
// embedded verbatim into kernel_goldens.json -- it has no bearing on
// parity itself, since both the Node dump and the Rust kernel run on the
// exact same embedded literals.
function makeLcg(seed) {
  let state = seed >>> 0;
  return function rand() {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}
const rand = makeLcg(42);
const coord = () => rand() * 10 - 5; // spread coords over [-5, 5)

// Builds a random undirected graph: `n` nodes, `targetEdges` unique edges,
// canonicalised (u < v) and de-duplicated as they're generated -- the Rust
// builder does not dedupe, so a raw duplicate would make graphToByteArrays
// (which uses mergeUndirectedEdge, dedupes) and graph_to_matrices compute
// different edge sets for the same fixture (Codex finding).
function buildRandomGraph(n, targetEdges) {
  const init = Array.from({ length: n }, () => [coord(), coord()]);
  const seen = new Set();
  const edges = [];
  while (edges.length < targetEdges) {
    let u = Math.floor(rand() * n);
    let v = Math.floor(rand() * n);
    if (u === v) continue;
    if (u > v) [u, v] = [v, u];
    const key = `${u},${v}`;
    if (seen.has(key)) continue;
    seen.add(key);
    edges.push([u, v]);
  }
  return { init, edges };
}

// --- Fixture graphs -----------------------------------------------------

const triangleInit = [[coord(), coord()], [coord(), coord()], [coord(), coord()]];
const triangleEdges = [[0, 1], [1, 2], [0, 2]];

const starInit = Array.from({ length: 9 }, () => [coord(), coord()]);
const starEdges = Array.from({ length: 8 }, (_, i) => [0, i + 1]);

const random60 = buildRandomGraph(60, 90);
const ewiWeights = random60.edges.map(() => rand() * 4 + 0.5); // [0.5, 4.5)

const coincidentInit = Array.from({ length: 6 }, () => [1, 1]);
// No edges: this fixture is purely about the BH quad-tree subdivision
// fallback (SUBDIVISION_ATTEMPTS) when every node lands in the same point,
// not about attraction.
const coincidentEdges = [];

// --- Fixture definitions -------------------------------------------------

const fixtures = [
  {
    name: "triangle_1it",
    n_nodes: 3,
    edges: triangleEdges,
    weights: null,
    init: triangleInit,
    settings: {},
    iterations: 1,
  },
  {
    name: "star_1it",
    n_nodes: 9,
    edges: starEdges,
    weights: null,
    init: starInit,
    settings: {},
    iterations: 1,
  },
  {
    name: "random60_pairwise_20it",
    n_nodes: 60,
    edges: random60.edges,
    weights: null,
    init: random60.init,
    settings: {},
    iterations: 20,
  },
  {
    name: "random60_bh_20it",
    n_nodes: 60,
    edges: random60.edges,
    weights: null,
    init: random60.init,
    settings: { barnesHutOptimize: true },
    iterations: 20,
  },
  {
    name: "crategraph_profile_20it",
    n_nodes: 60,
    edges: random60.edges,
    weights: null,
    init: random60.init,
    settings: {
      strongGravityMode: true,
      gravity: 0.05,
      scalingRatio: 10,
      slowDown: 1,
    },
    iterations: 20,
  },
  {
    name: "ewi_half_5it",
    n_nodes: 60,
    edges: random60.edges,
    weights: ewiWeights,
    init: random60.init,
    settings: { edgeWeightInfluence: 0.5 },
    iterations: 5,
  },
  {
    name: "ewi_two_5it",
    n_nodes: 60,
    edges: random60.edges,
    weights: ewiWeights,
    init: random60.init,
    settings: { edgeWeightInfluence: 2 },
    iterations: 5,
  },
  {
    name: "coincident_1it",
    n_nodes: 6,
    edges: coincidentEdges,
    weights: null,
    init: coincidentInit,
    settings: { barnesHutOptimize: true },
    iterations: 1,
  },
];

// All 8 attraction-ladder combinations (adjustSizes x linLogMode x
// outboundAttractionDistribution -- iterate.js:613 onward), on the random60
// graph, 5 iterations each.
for (let bits = 0; bits < 8; bits++) {
  const adjustSizes = Boolean(bits & 0b100);
  const linLogMode = Boolean(bits & 0b010);
  const outboundAttractionDistribution = Boolean(bits & 0b001);
  const label = `${bits & 0b100 ? 1 : 0}${bits & 0b010 ? 1 : 0}${bits & 0b001 ? 1 : 0}`;
  fixtures.push({
    name: `attr_${label}_5it`,
    n_nodes: 60,
    edges: random60.edges,
    weights: null,
    init: random60.init,
    settings: { adjustSizes, linLogMode, outboundAttractionDistribution },
    iterations: 5,
  });
}

// --- Run each fixture through the real graphology and collect goldens ---

const goldens = fixtures.map((fixture, i) => {
  const tmpPath = path.join(FIXTURES_DIR, `tmp_payload_${i}.json`);
  writeFileSync(tmpPath, JSON.stringify(fixture));
  try {
    const stdout = execFileSync(process.execPath, [DUMP_STATE, tmpPath], {
      encoding: "utf8",
    });
    const expected_bits = JSON.parse(stdout);
    console.log(`  ${fixture.name}: ${expected_bits.length} slots`);
    return { ...fixture, expected_bits };
  } finally {
    unlinkSync(tmpPath);
  }
});

writeFileSync(OUT_FILE, JSON.stringify(goldens, null, 2));
console.log(`wrote ${OUT_FILE} (${goldens.length} fixtures)`);
