//! Bit-parity gate: runs the Rust kernel over the same payloads that
//! `tests/node_reference/gen_kernel_fixtures.mjs` fed to the REAL
//! `graphology-layout-forceatlas2` 0.10.1 (via `dump_state.mjs`), and
//! asserts every `NodeMatrix` f32 slot is bit-for-bit identical
//! (`to_bits()`) after the same number of iterations.
//!
//! This is the correctness gate the whole port has been building toward:
//! if this test is green, the Rust kernel is byte-faithful to the JS
//! original on every fixture below, not merely "close".

use cfa2_kernel::matrices::graph_to_matrices;
use cfa2_kernel::settings::Settings;
use serde::Deserialize;

/// Mirrors `graphology-layout-forceatlas2/defaults.js`'s camelCase keys.
/// Every field is optional so a fixture only needs to specify the settings
/// it overrides -- exactly like the JS side's `{...DEFAULTS, ...settings}`
/// merge in `dump_state.mjs`.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct FixtureSettings {
    lin_log_mode: Option<bool>,
    outbound_attraction_distribution: Option<bool>,
    adjust_sizes: Option<bool>,
    edge_weight_influence: Option<f64>,
    scaling_ratio: Option<f64>,
    strong_gravity_mode: Option<bool>,
    gravity: Option<f64>,
    slow_down: Option<f64>,
    barnes_hut_optimize: Option<bool>,
    barnes_hut_theta: Option<f64>,
}

impl FixtureSettings {
    fn into_settings(self) -> Settings {
        let d = Settings::default();
        Settings {
            lin_log_mode: self.lin_log_mode.unwrap_or(d.lin_log_mode),
            outbound_attraction_distribution: self
                .outbound_attraction_distribution
                .unwrap_or(d.outbound_attraction_distribution),
            adjust_sizes: self.adjust_sizes.unwrap_or(d.adjust_sizes),
            edge_weight_influence: self.edge_weight_influence.unwrap_or(d.edge_weight_influence),
            scaling_ratio: self.scaling_ratio.unwrap_or(d.scaling_ratio),
            strong_gravity_mode: self.strong_gravity_mode.unwrap_or(d.strong_gravity_mode),
            gravity: self.gravity.unwrap_or(d.gravity),
            slow_down: self.slow_down.unwrap_or(d.slow_down),
            barnes_hut_optimize: self.barnes_hut_optimize.unwrap_or(d.barnes_hut_optimize),
            barnes_hut_theta: self.barnes_hut_theta.unwrap_or(d.barnes_hut_theta),
        }
    }
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
    n_nodes: usize,
    edges: Vec<(u32, u32)>,
    weights: Option<Vec<f64>>,
    init: Vec<(f64, f64)>,
    settings: FixtureSettings,
    iterations: usize,
    expected_bits: Vec<String>,
}

fn load_fixtures() -> Vec<Fixture> {
    let raw = include_str!("../../tests/node_reference/fixtures/kernel_goldens.json");
    serde_json::from_str(raw).expect("kernel_goldens.json must parse")
}

#[test]
fn kernel_matches_graphology_bit_for_bit() {
    let fixtures = load_fixtures();
    assert!(!fixtures.is_empty(), "fixture file must not be empty");

    let mut failures = Vec::new();

    for fixture in fixtures {
        let (mut nm, em) = graph_to_matrices(
            fixture.n_nodes,
            &fixture.edges,
            fixture.weights.as_deref(),
            &fixture.init,
        );

        let settings = fixture.settings.into_settings();
        for _ in 0..fixture.iterations {
            cfa2_kernel::iterate(&settings, &mut nm, &em);
        }

        let expected: Vec<u32> = fixture
            .expected_bits
            .iter()
            .map(|s| s.parse::<u32>().expect("expected_bits must be u32 decimal strings"))
            .collect();

        assert_eq!(
            nm.len(),
            expected.len(),
            "fixture {}: NodeMatrix length mismatch (got {}, want {})",
            fixture.name,
            nm.len(),
            expected.len()
        );

        // Report only the FIRST mismatching slot per fixture, to bisect
        // from the earliest divergence.
        let mut first_mismatch = None;
        for (slot, (&got_f32, &want_bits)) in nm.iter().zip(expected.iter()).enumerate() {
            let got_bits = got_f32.to_bits();
            if got_bits != want_bits {
                first_mismatch = Some((slot, got_bits, want_bits));
                break;
            }
        }

        if let Some((slot, got_bits, want_bits)) = first_mismatch {
            let node = slot / cfa2_kernel::matrices::PPN;
            let offset = slot % cfa2_kernel::matrices::PPN;
            let name = &fixture.name;
            let got_f32 = f32::from_bits(got_bits);
            let want_f32 = f32::from_bits(want_bits);
            failures.push(format!(
                "fixture `{name}`: first mismatch at node {node} slot-offset {offset} (flat index {slot}): \
                 got bits={got_bits:#010x} ({got_f32}), want bits={want_bits:#010x} ({want_f32})"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "bit-parity failures ({} fixture(s)):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
