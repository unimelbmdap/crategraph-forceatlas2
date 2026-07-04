//! Barnes-Hut quadtree region build.
//!
//! Faithful, verbatim port of the region-building section of
//! graphology-layout-forceatlas2's single iteration function
//! (`iterate.js:92-357`): the bounds pass, root region initialisation,
//! per-node insertion (the `while (true)` descent with quadrant selection),
//! leaf subdivision, and the `SUBDIVISION_ATTEMPTS` give-up fallback for
//! coincident nodes.
//!
//! `RegionMatrix` is a plain JS array of doubles (`iterate.js:69`), not a
//! typed array, so this port stores it as `Vec<f64>`, not `Vec<f32>`. Node
//! coordinates/mass are read from the `f32` `NodeMatrix` and widened to
//! `f64` on every read, matching the JS's implicit `Float32Array` ->
//! number widening.
#![allow(clippy::needless_range_loop, clippy::too_many_lines, clippy::similar_names)]

use crate::matrices::{
    NODE_MASS, NODE_X, NODE_Y, PPN, PPR, REGION_CENTER_X, REGION_CENTER_Y, REGION_FIRST_CHILD,
    REGION_MASS, REGION_MASS_CENTER_X, REGION_MASS_CENTER_Y, REGION_NEXT_SIBLING, REGION_NODE,
    REGION_SIZE, SUBDIVISION_ATTEMPTS,
};

/// Builds the Barnes-Hut `RegionMatrix` for `node_matrix`, a verbatim port
/// of `iterate.js:92-357`.
///
/// `node_matrix` is the `NodeMatrix` produced by
/// [`crate::matrices::graph_to_matrices`]; `node_matrix.len()` plays the
/// role of the JS `order` variable (total slot count, i.e. `n_nodes * PPN`,
/// not a node count).
pub fn build(node_matrix: &[f32]) -> Vec<f64> {
    let order = node_matrix.len();

    let mut region_matrix: Vec<f64> = Vec::new();

    // Computing min and max values (iterate.js:105-111).
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let mut n = 0;
    while n < order {
        let x = f64::from(node_matrix[n + NODE_X]);
        let y = f64::from(node_matrix[n + NODE_Y]);
        min_x = f64::min(min_x, x);
        max_x = f64::max(max_x, x);
        min_y = f64::min(min_y, y);
        max_y = f64::max(max_y, y);
        n += PPN;
    }

    // Squarify bounds, it's a quadtree (iterate.js:114-122).
    let dx = max_x - min_x;
    let dy = max_y - min_y;
    if dx > dy {
        min_y -= (dx - dy) / 2.0;
        max_y = min_y + dx;
    } else {
        min_x -= (dy - dx) / 2.0;
        max_x = min_x + dy;
    }

    // Build the Barnes Hut root region (iterate.js:124-133).
    region_matrix.resize(PPR, 0.0);
    region_matrix[REGION_NODE] = -1.0;
    region_matrix[REGION_CENTER_X] = (min_x + max_x) / 2.0;
    region_matrix[REGION_CENTER_Y] = (min_y + max_y) / 2.0;
    region_matrix[REGION_SIZE] = f64::max(max_x - min_x, max_y - min_y);
    region_matrix[REGION_NEXT_SIBLING] = -1.0;
    region_matrix[REGION_FIRST_CHILD] = -1.0;
    region_matrix[REGION_MASS] = 0.0;
    region_matrix[REGION_MASS_CENTER_X] = 0.0;
    region_matrix[REGION_MASS_CENTER_Y] = 0.0;

    // Add each node in the tree (iterate.js:135-357).
    let mut l: usize = 1;
    let mut n = 0;
    while n < order {
        // Current region, starting with root.
        let mut r: usize = 0;
        let mut subdivision_attempts = SUBDIVISION_ATTEMPTS;

        'descend: loop {
            // Are there sub-regions?
            // We look at first child index.
            if region_matrix[r + REGION_FIRST_CHILD] >= 0.0 {
                // There are sub-regions.
                //
                // We just iterate to find a "leaf" of the tree that is an
                // empty region or a region with a single node (see next
                // case).

                // Find the quadrant of n.
                let q: usize = if f64::from(node_matrix[n + NODE_X]) < region_matrix[r + REGION_CENTER_X] {
                    if f64::from(node_matrix[n + NODE_Y]) < region_matrix[r + REGION_CENTER_Y] {
                        // Top Left quarter.
                        region_matrix[r + REGION_FIRST_CHILD] as usize
                    } else {
                        // Bottom Left quarter.
                        region_matrix[r + REGION_FIRST_CHILD] as usize + PPR
                    }
                } else if f64::from(node_matrix[n + NODE_Y]) < region_matrix[r + REGION_CENTER_Y] {
                    // Top Right quarter.
                    region_matrix[r + REGION_FIRST_CHILD] as usize + PPR * 2
                } else {
                    // Bottom Right quarter.
                    region_matrix[r + REGION_FIRST_CHILD] as usize + PPR * 3
                };

                // Update center of mass and mass (we only do it for
                // non-leaf regions).
                let node_x = f64::from(node_matrix[n + NODE_X]);
                let node_y = f64::from(node_matrix[n + NODE_Y]);
                let node_mass = f64::from(node_matrix[n + NODE_MASS]);

                region_matrix[r + REGION_MASS_CENTER_X] = (region_matrix[r + REGION_MASS_CENTER_X]
                    * region_matrix[r + REGION_MASS]
                    + node_x * node_mass)
                    / (region_matrix[r + REGION_MASS] + node_mass);

                region_matrix[r + REGION_MASS_CENTER_Y] = (region_matrix[r + REGION_MASS_CENTER_Y]
                    * region_matrix[r + REGION_MASS]
                    + node_y * node_mass)
                    / (region_matrix[r + REGION_MASS] + node_mass);

                region_matrix[r + REGION_MASS] += node_mass;

                // Iterate on the right quadrant.
                r = q;
                continue 'descend;
            }

            // There are no sub-regions: we are in a "leaf".

            // Is there a node in this leaf?
            if region_matrix[r + REGION_NODE] < 0.0 {
                // There is no node in region: we record node n and go on.
                region_matrix[r + REGION_NODE] = n as f64;
                break 'descend;
            }

            // There is a node in this region.
            //
            // We will need to create sub-regions, stick the two nodes (the
            // old one r[0] and the new one n) in two subregions. If they
            // fall in the same quadrant, we will iterate.

            // Create sub-regions.
            region_matrix[r + REGION_FIRST_CHILD] = (l * PPR) as f64;
            let w = region_matrix[r + REGION_SIZE] / 2.0; // new size (half)

            // NOTE: we use screen coordinates from Top Left to Bottom Right.

            let first_child = region_matrix[r + REGION_FIRST_CHILD] as usize;
            if region_matrix.len() < first_child + 4 * PPR {
                region_matrix.resize(first_child + 4 * PPR, 0.0);
            }

            // Top Left sub-region.
            let mut g = first_child;
            region_matrix[g + REGION_NODE] = -1.0;
            region_matrix[g + REGION_CENTER_X] = region_matrix[r + REGION_CENTER_X] - w;
            region_matrix[g + REGION_CENTER_Y] = region_matrix[r + REGION_CENTER_Y] - w;
            region_matrix[g + REGION_SIZE] = w;
            region_matrix[g + REGION_NEXT_SIBLING] = (g + PPR) as f64;
            region_matrix[g + REGION_FIRST_CHILD] = -1.0;
            region_matrix[g + REGION_MASS] = 0.0;
            region_matrix[g + REGION_MASS_CENTER_X] = 0.0;
            region_matrix[g + REGION_MASS_CENTER_Y] = 0.0;

            // Bottom Left sub-region.
            g += PPR;
            region_matrix[g + REGION_NODE] = -1.0;
            region_matrix[g + REGION_CENTER_X] = region_matrix[r + REGION_CENTER_X] - w;
            region_matrix[g + REGION_CENTER_Y] = region_matrix[r + REGION_CENTER_Y] + w;
            region_matrix[g + REGION_SIZE] = w;
            region_matrix[g + REGION_NEXT_SIBLING] = (g + PPR) as f64;
            region_matrix[g + REGION_FIRST_CHILD] = -1.0;
            region_matrix[g + REGION_MASS] = 0.0;
            region_matrix[g + REGION_MASS_CENTER_X] = 0.0;
            region_matrix[g + REGION_MASS_CENTER_Y] = 0.0;

            // Top Right sub-region.
            g += PPR;
            region_matrix[g + REGION_NODE] = -1.0;
            region_matrix[g + REGION_CENTER_X] = region_matrix[r + REGION_CENTER_X] + w;
            region_matrix[g + REGION_CENTER_Y] = region_matrix[r + REGION_CENTER_Y] - w;
            region_matrix[g + REGION_SIZE] = w;
            region_matrix[g + REGION_NEXT_SIBLING] = (g + PPR) as f64;
            region_matrix[g + REGION_FIRST_CHILD] = -1.0;
            region_matrix[g + REGION_MASS] = 0.0;
            region_matrix[g + REGION_MASS_CENTER_X] = 0.0;
            region_matrix[g + REGION_MASS_CENTER_Y] = 0.0;

            // Bottom Right sub-region.
            g += PPR;
            region_matrix[g + REGION_NODE] = -1.0;
            region_matrix[g + REGION_CENTER_X] = region_matrix[r + REGION_CENTER_X] + w;
            region_matrix[g + REGION_CENTER_Y] = region_matrix[r + REGION_CENTER_Y] + w;
            region_matrix[g + REGION_SIZE] = w;
            region_matrix[g + REGION_NEXT_SIBLING] = region_matrix[r + REGION_NEXT_SIBLING];
            region_matrix[g + REGION_FIRST_CHILD] = -1.0;
            region_matrix[g + REGION_MASS] = 0.0;
            region_matrix[g + REGION_MASS_CENTER_X] = 0.0;
            region_matrix[g + REGION_MASS_CENTER_Y] = 0.0;

            l += 4;

            // Now the goal is to find two different sub-regions for the
            // two nodes: the one previously recorded (r[0]) and the one we
            // want to add (n).

            // Find the quadrant of the old node.
            let old_node = region_matrix[r + REGION_NODE] as usize;
            let old_x = f64::from(node_matrix[old_node + NODE_X]);
            let old_y = f64::from(node_matrix[old_node + NODE_Y]);

            let q: usize = if old_x < region_matrix[r + REGION_CENTER_X] {
                if old_y < region_matrix[r + REGION_CENTER_Y] {
                    // Top Left quarter.
                    region_matrix[r + REGION_FIRST_CHILD] as usize
                } else {
                    // Bottom Left quarter.
                    region_matrix[r + REGION_FIRST_CHILD] as usize + PPR
                }
            } else if old_y < region_matrix[r + REGION_CENTER_Y] {
                // Top Right quarter.
                region_matrix[r + REGION_FIRST_CHILD] as usize + PPR * 2
            } else {
                // Bottom Right quarter.
                region_matrix[r + REGION_FIRST_CHILD] as usize + PPR * 3
            };

            // We remove r[0] from the region r, add its mass to r and
            // record it in q.
            region_matrix[r + REGION_MASS] = f64::from(node_matrix[old_node + NODE_MASS]);
            region_matrix[r + REGION_MASS_CENTER_X] = old_x;
            region_matrix[r + REGION_MASS_CENTER_Y] = old_y;

            region_matrix[q + REGION_NODE] = region_matrix[r + REGION_NODE];
            region_matrix[r + REGION_NODE] = -1.0;

            // Find the quadrant of n.
            let node_x = f64::from(node_matrix[n + NODE_X]);
            let node_y = f64::from(node_matrix[n + NODE_Y]);

            let q2: usize = if node_x < region_matrix[r + REGION_CENTER_X] {
                if node_y < region_matrix[r + REGION_CENTER_Y] {
                    // Top Left quarter.
                    region_matrix[r + REGION_FIRST_CHILD] as usize
                } else {
                    // Bottom Left quarter.
                    region_matrix[r + REGION_FIRST_CHILD] as usize + PPR
                }
            } else if node_y < region_matrix[r + REGION_CENTER_Y] {
                // Top Right quarter.
                region_matrix[r + REGION_FIRST_CHILD] as usize + PPR * 2
            } else {
                // Bottom Right quarter.
                region_matrix[r + REGION_FIRST_CHILD] as usize + PPR * 3
            };

            if q == q2 {
                // If both nodes are in the same quadrant, we have to try it
                // again on this quadrant.
                if subdivision_attempts != 0 {
                    subdivision_attempts -= 1;
                    r = q;
                    continue 'descend; // while
                }
                // We are out of precision here, and we cannot subdivide
                // anymore, but we have to break the loop anyway.
                #[allow(unused_assignments)]
                {
                    subdivision_attempts = SUBDIVISION_ATTEMPTS;
                }
                break 'descend; // while
            }

            // If both quadrants are different, we record n in its
            // quadrant.
            region_matrix[q2 + REGION_NODE] = n as f64;
            break 'descend;
        }

        n += PPN;
    }

    region_matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::{PPN, REGION_CENTER_X, REGION_CENTER_Y, REGION_NODE, REGION_SIZE};

    fn nm_from(coords: &[(f64, f64)]) -> Vec<f32> {
        crate::matrices::graph_to_matrices(coords.len(), &[], None, coords).0
    }

    #[test]
    fn root_region_spans_all_nodes() {
        let nm = nm_from(&[(0.0, 0.0), (4.0, 0.0), (0.0, 3.0), (4.0, 3.0)]);
        let rm = build(&nm);
        // iterate.js:112-135: root centre is midpoint of bounds, size = max span
        assert_eq!(rm[REGION_CENTER_X], 2.0);
        assert_eq!(rm[REGION_CENTER_Y], 1.5);
        assert_eq!(rm[REGION_SIZE], 4.0);
    }

    // NOTE: do NOT test mass conservation or exact barycentres -- the real
    // JS does not maintain them. When a leaf splits, the parent's mass is
    // set from the old node only (iterate.js:306) and a new node landing in
    // a different quadrant is recorded WITHOUT updating that parent's mass
    // centre (iterate.js:350). A "fixed" Rust version would break parity.
    // Test only behaviours the JS guarantees; the bit-parity fixtures are
    // the real correctness gate for the tree.

    #[test]
    fn coincident_nodes_trigger_subdivision_fallback_without_hanging() {
        // 4 nodes at the identical coordinate: subdivision can never separate
        // them; graphology gives up after SUBDIVISION_ATTEMPTS (iterate.js:336-347).
        let nm = nm_from(&[(1.0, 1.0); 4]);
        let rm = build(&nm); // must terminate
        assert!(rm.len() >= PPR);
    }

    #[test]
    fn two_node_tree_has_root_and_two_populated_children() {
        // Simplest split: a root plus subdivided children referencing both node
        // offsets. REGION_NODE semantics verified against iterate.js:137-357:
        // a region's REGION_NODE holds >= 0 (a NodeMatrix offset) only for a
        // *leaf* region that currently owns exactly one node; -1 marks an
        // empty leaf. Once a region gets sub-regions (REGION_FIRST_CHILD >= 0)
        // it stops being addressed via REGION_NODE at all -- iterate.js:315
        // explicitly resets the just-subdivided parent's REGION_NODE to -1,
        // so "internal" and "empty leaf" share the same -1 sentinel and are
        // distinguished only by REGION_FIRST_CHILD, not by REGION_NODE.
        let nm = nm_from(&[(0.0, 0.0), (2.0, 2.0)]);
        let rm = build(&nm);
        assert!(rm.len() > PPR); // subdivided beyond the root
        let referenced: Vec<f64> = (0..rm.len() / PPR)
            .map(|r| rm[r * PPR + REGION_NODE])
            .filter(|&v| v >= 0.0)
            .collect();
        assert!(referenced.contains(&0.0) && referenced.contains(&(PPN as f64)));
    }
}
