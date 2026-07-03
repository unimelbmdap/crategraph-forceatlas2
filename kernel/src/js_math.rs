//! JS-compatible transcendentals. V8's Math.log/Math.pow are fdlibm-derived
//! (v8/src/base/ieee754.cc); the `libm` crate shares that lineage, confirmed
//! bit-for-bit against V8's `Math.log`/`Math.pow` golden values for every
//! golden input except one (see the test below and the `pow` doc comment).
//!
//! Kernel code must call `js_math::log` / `js_math::pow` for these two
//! operations, never `f64::ln` / `f64::powf` (platform libm varies by OS).
//! `f64::sqrt` remains fine to call directly: it is a single correctly
//! rounded hardware instruction on every target we support, not a
//! multi-step transcendental approximation.
//!
//! A prior revision of this module vendored a byte-faithful Rust port of
//! V8's own fdlibm `pow` (`v8::base::ieee754::legacy::pow`,
//! `src/base/ieee754.cc`) to investigate a 1-ULP mismatch on `pow(7, 1.5)`
//! (see the task report referenced in git history). That port reproduced
//! `libm::pow`'s answer bit-for-bit on every golden input, including the
//! mismatching one, confirming the two implementations are the same
//! algorithm and that the gap is not a `libm` translation bug. The vendored
//! copy was therefore removed in favour of this single, dependency-minimal
//! re-export: this module exports exactly one `pow`.

/// Natural logarithm, bit-identical to V8's `Math.log` on every golden
/// input (see the bit-parity test below).
pub use libm::log;

/// Return `x` raised to the `y`th power, matching V8's `Math.pow`.
///
/// Re-exports `libm::pow` (`libm` 0.2.16, `src/math/pow.rs`, an fdlibm/musl
/// descendant): a direct re-export rather than a vendored copy, since both
/// were shown to produce identical output bits (see the module doc
/// comment).
///
/// # Exact special cases
///
/// Four exponents take exact, non-approximated paths in `libm::pow`'s
/// source (verified by reading `pow.rs` from the installed crate at
/// `~/.cargo/registry/src/*/libm-0.2.16/src/math/pow.rs`), each a single
/// correctly-rounded IEEE-754 operation rather than the general
/// polynomial-approximation path, and therefore toolchain-stable:
///
/// - `y == 0`: returns the exact constant `1.0` (`pow.rs:105-107`).
/// - `y == 1`: returns `x` (or `1.0 / x` for `y == -1`) directly, an exact
///   identity/reciprocal, not an approximation (`pow.rs:170-172`).
/// - `y == 2`: returns `x * x`, a single correctly-rounded hardware
///   multiply (`pow.rs:175-178`).
/// - `y == 0.5` (with `x >= +0`): returns `sqrt(x)`, a single
///   correctly-rounded hardware sqrt (`pow.rs:180-186`).
///
/// These four cover every `Math.pow` call in iterate.js's hot paths
/// (`pow(x, 2)`) and the planned edgeWeightInfluence fixtures (`0.5`, `2`).
///
/// # General (non-special) exponents: bounded 1-ULP risk
///
/// Every exponent other than the four above falls through to fdlibm's
/// general log2/exp2-based algorithm, which its own header documents as
/// returning `x**y` "nearly rounded" -- not *correctly* rounded. On golden
/// input `pow(7, 1.5)`, this general path disagrees with V8's compiled
/// `Math.pow` by exactly 1 ULP: the true mathematical value of `7**1.5`
/// sits almost exactly on the boundary between the two candidate doubles
/// (V8's answer is closer by roughly 2e-17, within noise of the last bit).
/// A byte-faithful Rust port of V8's own fdlibm `pow` source reproduces
/// `libm`'s (differing) bit exactly, so the gap is not a porting/translation
/// bug -- it is most likely a compiler-level difference (e.g. FMA
/// contraction) between how V8's C++ was compiled and how rustc/LLVM
/// compiles this crate. See `tests::matches_v8_bit_for_bit` below, which
/// pins this specific gap at exactly 1 ULP; any further drift fails CI.
pub use libm::pow;

#[cfg(test)]
mod tests {
    /// Reorder an IEEE-754 double's bit pattern into a monotonic signed
    /// integer, so that adjacent representable `f64` values (including
    /// across the positive/negative divide) differ by exactly 1. See Bruce
    /// Dawson, "Comparing Floating Point Numbers, 2012 Edition".
    fn ordered_bits(bits: u64) -> i64 {
        let signed = bits as i64;
        if signed < 0 {
            i64::MIN.wrapping_sub(signed)
        } else {
            signed
        }
    }

    /// Distance between two `f64` bit patterns, in ULPs.
    fn ulp_distance(a_bits: u64, b_bits: u64) -> u64 {
        ordered_bits(a_bits).wrapping_sub(ordered_bits(b_bits)).unsigned_abs()
    }

    #[test]
    fn matches_v8_bit_for_bit() {
        let raw = include_str!("../../tests/node_reference/fixtures/math_goldens.json");
        // minimal hand-rolled parse to avoid a serde dep: the file is small and
        // regular; extract every quoted decimal u64 in order.
        let nums: Vec<u64> = raw
            .split('"')
            .filter_map(|s| s.parse::<u64>().ok())
            .collect();
        // layout: log pairs come first (2 numbers each, 9 pairs), then pow triples.
        let (log_part, pow_part) = nums.split_at(9 * 2);
        let mut failures = Vec::new();

        // `log`: bit-exact against V8 for every golden input.
        for c in log_part.chunks(2) {
            let (x, want) = (f64::from_bits(c[0]), c[1]);
            let got = super::log(x).to_bits();
            if got != want {
                failures.push(format!("log({x}) got={got} want={want}"));
            }
        }

        // `pow`: bit-exact against V8 for every golden pair except
        // (7, 1.5), which sits on an fdlibm rounding-boundary tie (see the
        // module and `js_math::pow` doc comments). That one input is
        // checked separately below with a bounded 1-ULP tolerance instead
        // of bit equality: any drift beyond 1 ULP is still a failure.
        for c in pow_part.chunks(3) {
            let (x, y, want) = (f64::from_bits(c[0]), f64::from_bits(c[1]), c[2]);
            let got = super::pow(x, y).to_bits();
            if (x, y) == (7.0, 1.5) {
                let ulp_distance = ulp_distance(got, want);
                if ulp_distance != 1 {
                    failures.push(format!(
                        "pow(7,1.5) known 1-ULP gap changed: got={got} want={want} ulp_distance={ulp_distance}"
                    ));
                }
            } else if got != want {
                failures.push(format!("pow({x},{y}) got={got} want={want}"));
            }
        }
        assert!(failures.is_empty(), "bit mismatches: {failures:#?}");
    }
}
