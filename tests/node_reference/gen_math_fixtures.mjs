// Dev-only: dump Math.log / Math.pow results as u64 bit patterns so the Rust
// kernel can assert bit-identity with V8's fdlibm-derived implementations.
import { writeFileSync } from "node:fs";

const f = new Float64Array(1);
const u = new BigUint64Array(f.buffer);
const bits = (x) => { f[0] = x; return u[0].toString(); };

const logInputs = [1e-300, 0.1, 0.5, 1.0000001, 2, Math.E, 10, 12345.6789, 1e12];
const powPairs = [[2, 2], [0.5, 2], [1.7, 2], [3.14159, 2], [2, 0.5], [10, -3],
                  [1.0000001, 100], [7, 1.5]];
writeFileSync("fixtures/math_goldens.json", JSON.stringify({
  log: logInputs.map((x) => [bits(x), bits(Math.log(x))]),
  pow: powPairs.map(([x, y]) => [bits(x), bits(y), bits(Math.pow(x, y))]),
}, null, 2));
console.log("wrote fixtures/math_goldens.json");
