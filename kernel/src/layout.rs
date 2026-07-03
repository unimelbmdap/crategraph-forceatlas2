//! Multi-iteration layout driver.
//!
//! `iterate::iterate` (Tasks 1-8) runs exactly one ForceAtlas2 iteration and
//! is verified bit-exact against graphology-layout-forceatlas2 (the
//! `kernel/tests/parity.rs` gate). This module adds the piece graphology's
//! JS side leaves to its caller: repeatedly running iterations, invoking a
//! per-iteration callback (so a host can report progress or abort), and --
//! new versus the JS -- parallelising the Barnes-Hut repulsion phase across
//! threads with `rayon` when asked to.
//!
//! Parallelism is deliberately narrow: only the Barnes-Hut per-node
//! repulsion loop (`iterate::bh_node_repulsion`) is split across threads.
//! Every other phase (init, gravity, attraction, apply, tree build, and the
//! O(n^2) pairwise repulsion fallback) stays sequential, matching the plan's
//! non-negotiable contract. Because each node's Barnes-Hut walk reads only a
//! read-only snapshot of other nodes' positions/mass and writes only its own
//! DX/DY (via the same per-addition f32 store discipline the sequential path
//! uses), splitting the loop across `threads` can never change the result:
//! `threads_do_not_change_the_result` below asserts bit-identical output
//! across thread counts.
use rayon::prelude::*;
use rayon::ThreadPool;

use crate::iterate::{
    apply_forces_phase, attraction_phase, bh_node_repulsion, gravity_phase, init_phase, iterate,
    repulsion_phase,
};
use crate::matrices::PPN;
use crate::quadtree;
use crate::settings::Settings;

/// Signals that the caller's `on_iteration` callback requested an abort.
///
/// Mirrors a JS caller throwing from inside a progress callback: graphology
/// itself has no such hook, but every host embedding this kernel (Task 10)
/// needs one to cancel long layouts, so `run_layout` treats an `Err` return
/// from the callback as "stop now" and propagates it as this unit-struct
/// error rather than a full error enum, since there is exactly one way to
/// abort and no extra detail to carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutAborted;

/// Runs `iterations` full ForceAtlas2 iterations over `nm`/`em` in place.
///
/// `on_iteration(i, iterations)` is called after every iteration with a
/// 1-based iteration number; returning `Err(LayoutAborted)` stops the loop
/// immediately (no further iterations run) and `run_layout` propagates the
/// same error.
///
/// `threads` selects the Barnes-Hut repulsion strategy: `1` (or `0`) runs
/// the existing, fully sequential [`iterate`] verbatim -- bypassing `rayon`
/// entirely -- while `> 1` builds a scoped `rayon` thread pool once (reused
/// across every iteration) and, when `s.barnes_hut_optimize` is set, splits
/// the per-node Barnes-Hut repulsion walk across it. `s.barnes_hut_optimize
/// == false` always runs the O(n^2) pairwise repulsion sequentially
/// regardless of `threads`, per the plan's contract that only the
/// Barnes-Hut loop parallelises.
pub fn run_layout(
    s: &Settings,
    nm: &mut [f32],
    em: &[f32],
    iterations: u32,
    threads: usize,
    mut on_iteration: impl FnMut(u32, u32) -> Result<(), LayoutAborted>,
) -> Result<(), LayoutAborted> {
    if threads <= 1 {
        for i in 1..=iterations {
            iterate(s, nm, em);
            on_iteration(i, iterations)?;
        }
        return Ok(());
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build rayon thread pool");

    for i in 1..=iterations {
        run_iteration_parallel(&pool, s, nm, em);
        on_iteration(i, iterations)?;
    }
    Ok(())
}

/// Runs one iteration with the Barnes-Hut repulsion phase (when enabled)
/// split across `pool`'s threads; every other phase runs sequentially on the
/// calling thread, an exact phase-for-phase mirror of [`iterate`].
fn run_iteration_parallel(pool: &ThreadPool, s: &Settings, nm: &mut [f32], em: &[f32]) {
    let outbound_att_compensation = init_phase(s, nm);

    if s.barnes_hut_optimize {
        let rm = quadtree::build(nm);
        let coefficient = s.scaling_ratio;
        let theta_squared = s.barnes_hut_theta * s.barnes_hut_theta;

        // Read-only snapshot of every node's current slots, taken before the
        // parallel loop: rust's aliasing rules forbid reading the full
        // NodeMatrix while it is borrowed mutably in chunks below, and this
        // is numerically identical to reading `nm` directly since no other
        // node's X/Y/mass changes during repulsion (only DX/DY, and only the
        // owning chunk's own DX/DY, are written).
        let snapshot: Vec<f64> = nm.iter().map(|&v| f64::from(v)).collect();

        pool.install(|| {
            nm.par_chunks_mut(PPN).enumerate().for_each(|(i, own)| {
                let n = i * PPN;
                bh_node_repulsion(s, own, n, &snapshot, &rm, coefficient, theta_squared);
            });
        });
    } else {
        // O(n^2) pairwise fallback stays sequential: each pair mutates both
        // endpoints' DX/DY, so it cannot be split into disjoint per-node
        // writes the way the Barnes-Hut walk can.
        repulsion_phase(s, nm, &[]);
    }

    gravity_phase(s, nm);
    attraction_phase(s, nm, em, outbound_att_compensation);
    apply_forces_phase(s, nm);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::graph_to_matrices;

    #[test]
    fn threads_do_not_change_the_result() {
        // BH per-node repulsion writes only the node's own DX/DY against an
        // immutable tree -- no reductions -- so ANY thread count must produce
        // bit-identical output.
        let coords: Vec<(f64, f64)> =
            (0..300).map(|i| ((i as f64 * 0.37) % 7.0, (i as f64 * 0.73) % 5.0)).collect();
        let edges: Vec<(u32, u32)> = (0..299).map(|i| (i, i + 1)).collect();
        let s = Settings { barnes_hut_optimize: true, ..Settings::default() };
        let build = || graph_to_matrices(300, &edges, None, &coords);
        let (mut a, em) = build();
        let (mut b, _) = build();
        run_layout(&s, &mut a, &em, 50, 1, |_, _| Ok(())).unwrap();
        run_layout(&s, &mut b, &em, 50, 4, |_, _| Ok(())).unwrap();
        assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                   b.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn callback_sees_every_iteration_and_can_abort() {
        let (mut nm, em) = graph_to_matrices(2, &[(0, 1)], None, &[(0.0, 0.0), (1.0, 0.0)]);
        let mut seen = vec![];
        run_layout(&Settings::default(), &mut nm, &em, 5, 1,
                   |i, total| { seen.push((i, total)); Ok(()) }).unwrap();
        assert_eq!(seen, vec![(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]); // 1-based
        let err = run_layout(&Settings::default(), &mut nm, &em, 5, 1,
                             |i, _| if i == 2 { Err(LayoutAborted) } else { Ok(()) });
        assert!(err.is_err());
    }
}
