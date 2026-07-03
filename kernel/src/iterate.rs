//! Faithful port of graphology-layout-forceatlas2's single-iteration
//! function (`iterate.js`), part 1: the init phase and both repulsion
//! modes.
//!
//! - `init_phase` is `iterate.js:71-90`: the dx/dy -> old_dx/old_dy swap
//!   (with zeroing) and the `outboundAttCompensation` accumulation.
//! - `repulsion_phase` is `iterate.js:360-564`: the Barnes-Hut per-node
//!   tree walk (`iterate.js:368-497`, branching on `RegionMatrix`) and the
//!   O(n^2) pairwise fallback (`iterate.js:501-564`), each further split on
//!   `adjustSizes`.
//!
//! `NodeMatrix` is `Vec<f32>` (mirrors JS `Float32Array`): every store that
//! corresponds to a JS `+=`/`-=` on the matrix is read-widen-add-narrow
//! *per addition*, via [`add_f32`], matching `Float32Array` semantics
//! exactly -- this matters because the Barnes-Hut walk applies multiple
//! force contributions to the same node's DX/DY across tree regions.
//! `RegionMatrix` is `Vec<f64>` (a plain JS array of doubles, per
//! `crate::quadtree`), read directly with no widening.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::similar_names
)]

use crate::matrices::{
    EDGE_SOURCE, EDGE_TARGET, EDGE_WEIGHT, MAX_FORCE, NODE_CONVERGENCE, NODE_DX, NODE_DY,
    NODE_FIXED, NODE_MASS, NODE_OLD_DX, NODE_OLD_DY, NODE_SIZE, NODE_X, NODE_Y, PPE, PPN,
    REGION_FIRST_CHILD, REGION_MASS, REGION_MASS_CENTER_X, REGION_MASS_CENTER_Y,
    REGION_NEXT_SIBLING, REGION_NODE, REGION_SIZE,
};
use crate::settings::Settings;

/// Applies `delta` to `nm[idx]` via read-widen-add-narrow, i.e. the same
/// single rounding step a JS `Float32Array[idx] += delta` performs. Every
/// `+=`/`-=` onto the `NodeMatrix` in `iterate.js` must go through this
/// (a `-=` is just `add_f32(nm, idx, -delta)`, since `x - y == x + (-y)`
/// exactly in IEEE-754, no extra rounding introduced).
fn add_f32(nm: &mut [f32], idx: usize, delta: f64) {
    nm[idx] = (f64::from(nm[idx]) + delta) as f32;
}

/// Resets per-iteration force accumulators and (optionally) computes the
/// outbound-attraction-distribution compensation, a verbatim port of
/// `iterate.js:71-90`.
///
/// Returns `outboundAttCompensation`. When
/// `s.outbound_attraction_distribution` is `false`, the JS leaves that
/// variable `undefined` (it is only ever read from the attraction phase
/// under the same flag in Task 7); this port returns `0.0` in that case,
/// an unused placeholder rather than a meaningful zero.
pub(crate) fn init_phase(s: &Settings, nm: &mut [f32]) -> f64 {
    let order = nm.len();

    // Resetting positions & computing max values (iterate.js:75-80).
    let mut n = 0;
    while n < order {
        nm[n + NODE_OLD_DX] = nm[n + NODE_DX];
        nm[n + NODE_OLD_DY] = nm[n + NODE_DY];
        nm[n + NODE_DX] = 0.0;
        nm[n + NODE_DY] = 0.0;
        n += PPN;
    }

    // If outbound attraction distribution, compensate (iterate.js:83-90).
    let mut outbound_att_compensation = 0.0f64;
    if s.outbound_attraction_distribution {
        let mut n = 0;
        while n < order {
            outbound_att_compensation += f64::from(nm[n + NODE_MASS]);
            n += PPN;
        }
        outbound_att_compensation /= (order / PPN) as f64;
    }

    outbound_att_compensation
}

/// Applies repulsion forces to every node's DX/DY, a verbatim port of
/// `iterate.js:360-564`.
///
/// Branches on `s.barnes_hut_optimize`: when set, walks the `RegionMatrix`
/// built by [`crate::quadtree::build`] per node (`iterate.js:368-497`);
/// otherwise runs the O(n^2) pairwise loop (`iterate.js:501-564`). Both
/// branches further split on `s.adjust_sizes` (JS's `adjustSizes`,
/// anti-collision repulsion).
///
/// The Barnes-Hut per-node walk reads only the tree (`rm`) plus its own
/// node's slots and writes only its own node's DX/DY -- deliberately kept
/// that way (no cross-node writes) so Task 9's per-node parallelism is
/// sound.
pub(crate) fn repulsion_phase(s: &Settings, nm: &mut [f32], rm: &[f64]) {
    let order = nm.len();
    let coefficient = s.scaling_ratio;

    if s.barnes_hut_optimize {
        let theta_squared = s.barnes_hut_theta * s.barnes_hut_theta;

        // Applying repulsion through regions (iterate.js:368-497).
        let mut n = 0;
        while n < order {
            // Computing leaf quad nodes iteration.
            let mut r: usize = 0; // Starting with root region.
            loop {
                if rm[r + REGION_FIRST_CHILD] >= 0.0 {
                    // The region has sub-regions.

                    // We run the Barnes Hut test to see if we are at the
                    // right distance.
                    let distance = crate::js_math::pow(
                        f64::from(nm[n + NODE_X]) - rm[r + REGION_MASS_CENTER_X],
                        2.0,
                    ) + crate::js_math::pow(
                        f64::from(nm[n + NODE_Y]) - rm[r + REGION_MASS_CENTER_Y],
                        2.0,
                    );

                    let region_size = rm[r + REGION_SIZE];

                    if (4.0 * region_size * region_size) / distance < theta_squared {
                        // We treat the region as a single body, and we
                        // repulse.
                        let x_dist = f64::from(nm[n + NODE_X]) - rm[r + REGION_MASS_CENTER_X];
                        let y_dist = f64::from(nm[n + NODE_Y]) - rm[r + REGION_MASS_CENTER_Y];

                        if s.adjust_sizes {
                            //-- Linear Anti-collision Repulsion.
                            if distance > 0.0 {
                                let factor = (coefficient
                                    * f64::from(nm[n + NODE_MASS])
                                    * rm[r + REGION_MASS])
                                    / distance;

                                add_f32(nm, n + NODE_DX, x_dist * factor);
                                add_f32(nm, n + NODE_DY, y_dist * factor);
                            } else if distance < 0.0 {
                                // JS parity: unreachable in practice (distance is a sum of squares).
                                let factor = (-coefficient
                                    * f64::from(nm[n + NODE_MASS])
                                    * rm[r + REGION_MASS])
                                    / f64::sqrt(distance);

                                add_f32(nm, n + NODE_DX, x_dist * factor);
                                add_f32(nm, n + NODE_DY, y_dist * factor);
                            }
                        } else {
                            //-- Linear Repulsion.
                            if distance > 0.0 {
                                let factor = (coefficient
                                    * f64::from(nm[n + NODE_MASS])
                                    * rm[r + REGION_MASS])
                                    / distance;

                                add_f32(nm, n + NODE_DX, x_dist * factor);
                                add_f32(nm, n + NODE_DY, y_dist * factor);
                            }
                        }

                        // When this is done, we iterate. We have to look
                        // at the next sibling.
                        let next_sibling = rm[r + REGION_NEXT_SIBLING];
                        if next_sibling < 0.0 {
                            break; // No next sibling: we have finished the tree.
                        }
                        r = next_sibling as usize;
                        continue;
                    }
                    // The region is too close and we have to look at
                    // sub-regions.
                    r = rm[r + REGION_FIRST_CHILD] as usize;
                    continue;
                }
                // The region has no sub-region.
                // If there is a node r[0] and it is not n, then repulse.
                let rn = rm[r + REGION_NODE];

                if rn >= 0.0 && rn as usize != n {
                    let rn_idx = rn as usize;
                    let x_dist = f64::from(nm[n + NODE_X]) - f64::from(nm[rn_idx + NODE_X]);
                    let y_dist = f64::from(nm[n + NODE_Y]) - f64::from(nm[rn_idx + NODE_Y]);

                    let distance = x_dist * x_dist + y_dist * y_dist;

                    if s.adjust_sizes {
                        //-- Linear Anti-collision Repulsion.
                        if distance > 0.0 {
                            let factor = (coefficient
                                * f64::from(nm[n + NODE_MASS])
                                * f64::from(nm[rn_idx + NODE_MASS]))
                                / distance;

                            add_f32(nm, n + NODE_DX, x_dist * factor);
                            add_f32(nm, n + NODE_DY, y_dist * factor);
                        } else if distance < 0.0 {
                            // JS parity: unreachable in practice (distance is a sum of squares).
                            let factor = (-coefficient
                                * f64::from(nm[n + NODE_MASS])
                                * f64::from(nm[rn_idx + NODE_MASS]))
                                / f64::sqrt(distance);

                            add_f32(nm, n + NODE_DX, x_dist * factor);
                            add_f32(nm, n + NODE_DY, y_dist * factor);
                        }
                    } else {
                        //-- Linear Repulsion.
                        if distance > 0.0 {
                            let factor = (coefficient
                                * f64::from(nm[n + NODE_MASS])
                                * f64::from(nm[rn_idx + NODE_MASS]))
                                / distance;

                            add_f32(nm, n + NODE_DX, x_dist * factor);
                            add_f32(nm, n + NODE_DY, y_dist * factor);
                        }
                    }
                }

                // When this is done, we iterate. We have to look at the
                // next sibling.
                let next_sibling = rm[r + REGION_NEXT_SIBLING];
                if next_sibling < 0.0 {
                    break; // No next sibling: we have finished the tree.
                }
                r = next_sibling as usize;
                continue;
            }

            n += PPN;
        }
    } else {
        // Square iteration (iterate.js:501-564).
        let mut n1 = 0;
        while n1 < order {
            let mut n2 = 0;
            while n2 < n1 {
                // Common to both methods.
                let x_dist = f64::from(nm[n1 + NODE_X]) - f64::from(nm[n2 + NODE_X]);
                let y_dist = f64::from(nm[n1 + NODE_Y]) - f64::from(nm[n2 + NODE_Y]);

                if s.adjust_sizes {
                    //-- Anticollision Linear Repulsion.
                    let distance = f64::sqrt(x_dist * x_dist + y_dist * y_dist)
                        - f64::from(nm[n1 + NODE_SIZE])
                        - f64::from(nm[n2 + NODE_SIZE]);

                    if distance > 0.0 {
                        let factor = (coefficient
                            * f64::from(nm[n1 + NODE_MASS])
                            * f64::from(nm[n2 + NODE_MASS]))
                            / distance
                            / distance;

                        // Updating nodes' dx and dy.
                        add_f32(nm, n1 + NODE_DX, x_dist * factor);
                        add_f32(nm, n1 + NODE_DY, y_dist * factor);

                        add_f32(nm, n2 + NODE_DX, -(x_dist * factor));
                        add_f32(nm, n2 + NODE_DY, -(y_dist * factor));
                    } else if distance < 0.0 {
                        let factor = 100.0
                            * coefficient
                            * f64::from(nm[n1 + NODE_MASS])
                            * f64::from(nm[n2 + NODE_MASS]);

                        // Updating nodes' dx and dy.
                        add_f32(nm, n1 + NODE_DX, x_dist * factor);
                        add_f32(nm, n1 + NODE_DY, y_dist * factor);

                        add_f32(nm, n2 + NODE_DX, -(x_dist * factor));
                        add_f32(nm, n2 + NODE_DY, -(y_dist * factor));
                    }
                } else {
                    //-- Linear Repulsion.
                    let distance = f64::sqrt(x_dist * x_dist + y_dist * y_dist);

                    if distance > 0.0 {
                        let factor = (coefficient
                            * f64::from(nm[n1 + NODE_MASS])
                            * f64::from(nm[n2 + NODE_MASS]))
                            / distance
                            / distance;

                        // Updating nodes' dx and dy.
                        add_f32(nm, n1 + NODE_DX, x_dist * factor);
                        add_f32(nm, n1 + NODE_DY, y_dist * factor);

                        add_f32(nm, n2 + NODE_DX, -(x_dist * factor));
                        add_f32(nm, n2 + NODE_DY, -(y_dist * factor));
                    }
                }

                n2 += PPN;
            }
            n1 += PPN;
        }
    }
}

/// Applies gravity forces to every node's DX/DY, a verbatim port of
/// `iterate.js:567-591`.
///
/// Branches on `s.strong_gravity_mode`: the strong branch pulls every node
/// toward the origin with a distance-independent force, while the linear
/// branch (graphology's "Linear Anti-collision Repulsion n" comment,
/// verbatim) divides by distance.
pub(crate) fn gravity_phase(s: &Settings, nm: &mut [f32]) {
    let order = nm.len();
    let g = s.gravity / s.scaling_ratio;
    let coefficient = s.scaling_ratio;

    let mut n = 0;
    while n < order {
        let mut factor = 0.0f64;

        // Common to both methods (iterate.js:574-577).
        let x_dist = f64::from(nm[n + NODE_X]);
        let y_dist = f64::from(nm[n + NODE_Y]);
        let distance =
            f64::sqrt(crate::js_math::pow(x_dist, 2.0) + crate::js_math::pow(y_dist, 2.0));

        if s.strong_gravity_mode {
            //-- Strong gravity.
            if distance > 0.0 {
                factor = coefficient * f64::from(nm[n + NODE_MASS]) * g;
            }
        } else {
            //-- Linear Anti-collision Repulsion n.
            if distance > 0.0 {
                factor = (coefficient * f64::from(nm[n + NODE_MASS]) * g) / distance;
            }
        }

        // Updating node's dx and dy.
        add_f32(nm, n + NODE_DX, -(x_dist * factor));
        add_f32(nm, n + NODE_DY, -(y_dist * factor));

        n += PPN;
    }
}

/// Applies attraction forces along every edge to both endpoints' DX/DY, a
/// verbatim port of `iterate.js:593-688`.
///
/// `outbound_att_compensation` is the value returned by [`init_phase`]
/// (JS's `outboundAttCompensation`, only meaningful when
/// `s.outbound_attraction_distribution` is set).
///
/// The JS body is a flat if/else ladder over `adjustSizes` x `linLogMode` x
/// `outboundAttractionDistribution` (8 leaves, 2 of which collapse to the
/// same "set distance=1" shape); every leaf is kept distinct here rather
/// than merged, matching the brief's "keep it flat" instruction.
pub(crate) fn attraction_phase(
    s: &Settings,
    nm: &mut [f32],
    em: &[f32],
    outbound_att_compensation: f64,
) {
    let size = em.len();

    // iterate.js:595-596.
    let coefficient = 1.0
        * if s.outbound_attraction_distribution {
            outbound_att_compensation
        } else {
            1.0
        };

    let mut e = 0;
    while e < size {
        let n1 = em[e + EDGE_SOURCE] as usize;
        let n2 = em[e + EDGE_TARGET] as usize;
        let w = f64::from(em[e + EDGE_WEIGHT]);

        // Edge weight influence (iterate.js:606).
        let ewc = crate::js_math::pow(w, s.edge_weight_influence);

        // Common measures (iterate.js:609-610).
        let x_dist = f64::from(nm[n1 + NODE_X]) - f64::from(nm[n2 + NODE_X]);
        let y_dist = f64::from(nm[n1 + NODE_Y]) - f64::from(nm[n2 + NODE_Y]);

        let mut factor = 0.0f64;
        let mut distance;

        // Applying attraction to nodes (iterate.js:612-677).
        if s.adjust_sizes {
            distance = f64::sqrt(x_dist * x_dist + y_dist * y_dist)
                - f64::from(nm[n1 + NODE_SIZE])
                - f64::from(nm[n2 + NODE_SIZE]);

            if s.lin_log_mode {
                if s.outbound_attraction_distribution {
                    //-- LinLog Degree Distributed Anti-collision Attraction.
                    if distance > 0.0 {
                        factor = (-coefficient * ewc * crate::js_math::log(1.0 + distance))
                            / distance
                            / f64::from(nm[n1 + NODE_MASS]);
                    }
                } else {
                    //-- LinLog Anti-collision Attraction.
                    if distance > 0.0 {
                        factor =
                            (-coefficient * ewc * crate::js_math::log(1.0 + distance)) / distance;
                    }
                }
            } else if s.outbound_attraction_distribution {
                //-- Linear Degree Distributed Anti-collision Attraction.
                if distance > 0.0 {
                    factor = (-coefficient * ewc) / f64::from(nm[n1 + NODE_MASS]);
                }
            } else {
                //-- Linear Anti-collision Attraction.
                if distance > 0.0 {
                    factor = -coefficient * ewc;
                }
            }
        } else {
            distance =
                f64::sqrt(crate::js_math::pow(x_dist, 2.0) + crate::js_math::pow(y_dist, 2.0));

            if s.lin_log_mode {
                if s.outbound_attraction_distribution {
                    //-- LinLog Degree Distributed Attraction.
                    if distance > 0.0 {
                        factor = (-coefficient * ewc * crate::js_math::log(1.0 + distance))
                            / distance
                            / f64::from(nm[n1 + NODE_MASS]);
                    }
                } else {
                    //-- LinLog Attraction.
                    if distance > 0.0 {
                        factor =
                            (-coefficient * ewc * crate::js_math::log(1.0 + distance)) / distance;
                    }
                }
            } else if s.outbound_attraction_distribution {
                //-- Linear Attraction Mass Distributed.
                // NOTE: Distance is set to 1 to override next condition.
                distance = 1.0;
                factor = (-coefficient * ewc) / f64::from(nm[n1 + NODE_MASS]);
            } else {
                //-- Linear Attraction.
                // NOTE: Distance is set to 1 to override next condition.
                distance = 1.0;
                factor = -coefficient * ewc;
            }
        }

        // Updating nodes' dx and dy (iterate.js:679-688).
        if distance > 0.0 {
            add_f32(nm, n1 + NODE_DX, x_dist * factor);
            add_f32(nm, n1 + NODE_DY, y_dist * factor);

            add_f32(nm, n2 + NODE_DX, -(x_dist * factor));
            add_f32(nm, n2 + NODE_DY, -(y_dist * factor));
        }

        e += PPE;
    }
}

/// Integrates accumulated forces into node positions, a verbatim port of
/// `iterate.js:691-789`.
///
/// Branches on `s.adjust_sizes`: the anti-collision path caps the force
/// magnitude at [`MAX_FORCE`] before computing swinging/traction, and does
/// not touch `NODE_CONVERGENCE`; the standard path has no force cap but
/// folds the per-node `NODE_CONVERGENCE` value into `nodespeed` and updates
/// it afterwards. Both paths skip fixed nodes (`NODE_FIXED == 1`) and
/// finish by dividing the displacement by `s.slow_down`.
pub(crate) fn apply_forces_phase(s: &Settings, nm: &mut [f32]) {
    let order = nm.len();

    if s.adjust_sizes {
        // MATH: sqrt and square distances (iterate.js:696-741).
        let mut n = 0;
        while n < order {
            if nm[n + NODE_FIXED] != 1.0 {
                let mut dx = f64::from(nm[n + NODE_DX]);
                let mut dy = f64::from(nm[n + NODE_DY]);

                let force = f64::sqrt(crate::js_math::pow(dx, 2.0) + crate::js_math::pow(dy, 2.0));

                if force > MAX_FORCE {
                    nm[n + NODE_DX] = ((dx * MAX_FORCE) / force) as f32;
                    nm[n + NODE_DY] = ((dy * MAX_FORCE) / force) as f32;
                    // JS reads the capped, f32-rounded value back out of the
                    // typed array for the downstream computations
                    // (iterate.js:704-713); mirror that here rather than
                    // keeping the un-rounded f64 locals.
                    dx = f64::from(nm[n + NODE_DX]);
                    dy = f64::from(nm[n + NODE_DY]);
                }

                let old_dx = f64::from(nm[n + NODE_OLD_DX]);
                let old_dy = f64::from(nm[n + NODE_OLD_DY]);

                let swinging = f64::from(nm[n + NODE_MASS])
                    * f64::sqrt((old_dx - dx) * (old_dx - dx) + (old_dy - dy) * (old_dy - dy));

                let traction =
                    f64::sqrt((old_dx + dx) * (old_dx + dx) + (old_dy + dy) * (old_dy + dy)) / 2.0;

                let nodespeed =
                    (0.1 * crate::js_math::log(1.0 + traction)) / (1.0 + f64::sqrt(swinging));

                // Updating node's position.
                let new_x = f64::from(nm[n + NODE_X]) + dx * (nodespeed / s.slow_down);
                nm[n + NODE_X] = new_x as f32;

                let new_y = f64::from(nm[n + NODE_Y]) + dy * (nodespeed / s.slow_down);
                nm[n + NODE_Y] = new_y as f32;
            }

            n += PPN;
        }
    } else {
        let mut n = 0;
        while n < order {
            if nm[n + NODE_FIXED] != 1.0 {
                let dx = f64::from(nm[n + NODE_DX]);
                let dy = f64::from(nm[n + NODE_DY]);
                let old_dx = f64::from(nm[n + NODE_OLD_DX]);
                let old_dy = f64::from(nm[n + NODE_OLD_DY]);

                let swinging = f64::from(nm[n + NODE_MASS])
                    * f64::sqrt((old_dx - dx) * (old_dx - dx) + (old_dy - dy) * (old_dy - dy));

                let traction =
                    f64::sqrt((old_dx + dx) * (old_dx + dx) + (old_dy + dy) * (old_dy + dy)) / 2.0;

                let nodespeed = (f64::from(nm[n + NODE_CONVERGENCE])
                    * crate::js_math::log(1.0 + traction))
                    / (1.0 + f64::sqrt(swinging));

                // Updating node convergence.
                let convergence = f64::min(
                    1.0,
                    f64::sqrt(
                        (nodespeed * (crate::js_math::pow(dx, 2.0) + crate::js_math::pow(dy, 2.0)))
                            / (1.0 + f64::sqrt(swinging)),
                    ),
                );
                nm[n + NODE_CONVERGENCE] = convergence as f32;

                // Updating node's position.
                let new_x = f64::from(nm[n + NODE_X]) + dx * (nodespeed / s.slow_down);
                nm[n + NODE_X] = new_x as f32;

                let new_y = f64::from(nm[n + NODE_Y]) + dy * (nodespeed / s.slow_down);
                nm[n + NODE_Y] = new_y as f32;
            }

            n += PPN;
        }
    }
}

/// Runs one full ForceAtlas2 iteration over `nm`/`em` in place, a verbatim
/// composition of `iterate.js`'s phase order: init, (quadtree build when
/// `s.barnes_hut_optimize`), repulsion, gravity, attraction, apply forces.
pub fn iterate(s: &Settings, nm: &mut [f32], em: &[f32]) {
    let outbound_att_compensation = init_phase(s, nm);

    if s.barnes_hut_optimize {
        let rm = crate::quadtree::build(nm);
        repulsion_phase(s, nm, &rm);
    } else {
        repulsion_phase(s, nm, &[]);
    }

    gravity_phase(s, nm);
    attraction_phase(s, nm, em, outbound_att_compensation);
    apply_forces_phase(s, nm);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::*;
    use crate::settings::Settings;

    fn two_nodes() -> Vec<f32> {
        graph_to_matrices(2, &[], None, &[(0.0, 0.0), (1.0, 0.0)]).0
    }

    #[test]
    fn init_copies_dx_to_old_and_zeroes() {
        let mut nm = two_nodes();
        nm[NODE_DX] = 5.0;
        init_phase(&Settings::default(), &mut nm);
        assert_eq!(nm[NODE_OLD_DX], 5.0); // iterate.js:76-79
        assert_eq!(nm[NODE_DX], 0.0);
    }

    #[test]
    fn pairwise_repulsion_is_equal_and_opposite_on_x_axis() {
        let mut nm = two_nodes();
        let s = Settings {
            barnes_hut_optimize: false,
            ..Settings::default()
        };
        init_phase(&s, &mut nm);
        repulsion_phase(&s, &mut nm, &[]);
        assert!(nm[NODE_DX] < 0.0); // node 0 pushed -x
        assert!(nm[PPN + NODE_DX] > 0.0); // node 1 pushed +x
        assert_eq!(nm[NODE_DX], -nm[PPN + NODE_DX]); // symmetric masses
        assert_eq!(nm[NODE_DY], 0.0);
    }

    #[test]
    fn barnes_hut_approximates_pairwise_for_small_graph() {
        // With theta=0.5 and few nodes the tree devolves to near-exact pairs;
        // directions must agree even if magnitudes differ slightly.
        let coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (5.0, 5.0)];
        let (mut nm_bh, _) = graph_to_matrices(4, &[], None, &coords);
        let mut nm_pw = nm_bh.clone();
        let s_pw = Settings {
            barnes_hut_optimize: false,
            ..Settings::default()
        };
        let s_bh = Settings {
            barnes_hut_optimize: true,
            ..Settings::default()
        };
        init_phase(&s_pw, &mut nm_pw);
        repulsion_phase(&s_pw, &mut nm_pw, &[]);
        let rm = crate::quadtree::build(&nm_bh);
        init_phase(&s_bh, &mut nm_bh);
        repulsion_phase(&s_bh, &mut nm_bh, &rm);
        for i in 0..4 {
            assert_eq!(
                nm_bh[i * PPN + NODE_DX].signum(),
                nm_pw[i * PPN + NODE_DX].signum()
            );
            assert_eq!(
                nm_bh[i * PPN + NODE_DY].signum(),
                nm_pw[i * PPN + NODE_DY].signum()
            );
        }
    }

    #[test]
    fn gravity_pulls_toward_origin() {
        let (mut nm, _) = graph_to_matrices(1, &[], None, &[(3.0, 4.0)]);
        let s = Settings {
            gravity: 1.0,
            ..Settings::default()
        };
        init_phase(&s, &mut nm);
        gravity_phase(&s, &mut nm);
        assert!(nm[NODE_DX] < 0.0 && nm[NODE_DY] < 0.0);
    }

    #[test]
    fn attraction_pulls_edge_endpoints_together() {
        let (mut nm, em) = graph_to_matrices(2, &[(0, 1)], None, &[(0.0, 0.0), (10.0, 0.0)]);
        let s = Settings::default();
        init_phase(&s, &mut nm);
        attraction_phase(&s, &mut nm, &em, 1.0);
        assert!(nm[NODE_DX] > 0.0); // node 0 pulled +x
        assert!(nm[PPN + NODE_DX] < 0.0); // node 1 pulled -x
    }

    #[test]
    fn full_iteration_moves_connected_pair_closer_and_is_finite() {
        let (mut nm, em) = graph_to_matrices(2, &[(0, 1)], None, &[(0.0, 0.0), (10.0, 0.0)]);
        let before = (nm[PPN + NODE_X] - nm[NODE_X]).abs();
        iterate(&Settings::default(), &mut nm, &em);
        let after = (nm[PPN + NODE_X] - nm[NODE_X]).abs();
        assert!(after < before);
        assert!(nm.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn max_force_caps_displacement_inputs() {
        // adjustSizes path (iterate.js:699-708) caps dx/dy at MAX_FORCE
        let (mut nm, em) = graph_to_matrices(2, &[(0, 1)], None, &[(0.0, 0.0), (1e-6, 0.0)]);
        let s = Settings {
            adjust_sizes: true,
            ..Settings::default()
        };
        iterate(&s, &mut nm, &em);
        assert!(nm.iter().all(|v| v.is_finite()));
    }
}
