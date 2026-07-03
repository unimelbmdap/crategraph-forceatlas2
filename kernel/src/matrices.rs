//! Node/edge matrix layout and the graph-to-matrix builder.
//!
//! Faithful port of graphology-layout-forceatlas2's `helpers.js` (slot
//! layout from `iterate.js:12-46`, builder from `helpers.js:117-171`).
//! `NodeMatrix`/`EdgeMatrix` are plain `Vec<f32>`, mirroring the JS
//! `Float32Array`s: every read widens to `f64`, arithmetic happens in
//! `f64`, and every store narrows back `as f32`.

/// Properties-per-node: slot count in a `NodeMatrix` row.
pub const PPN: usize = 10;
/// Properties-per-edge: slot count in an `EdgeMatrix` row.
pub const PPE: usize = 3;
/// Properties-per-region: slot count in a Barnes-Hut `RegionMatrix` row.
pub const PPR: usize = 9;

/// Node matrix slot offsets (`iterate.js:12-21`).
pub const NODE_X: usize = 0;
pub const NODE_Y: usize = 1;
pub const NODE_DX: usize = 2;
pub const NODE_DY: usize = 3;
pub const NODE_OLD_DX: usize = 4;
pub const NODE_OLD_DY: usize = 5;
pub const NODE_MASS: usize = 6;
pub const NODE_CONVERGENCE: usize = 7;
pub const NODE_SIZE: usize = 8;
pub const NODE_FIXED: usize = 9;

/// Edge matrix slot offsets (`iterate.js:23-25`).
pub const EDGE_SOURCE: usize = 0;
pub const EDGE_TARGET: usize = 1;
pub const EDGE_WEIGHT: usize = 2;

/// Region matrix slot offsets (`iterate.js:27-35`).
pub const REGION_NODE: usize = 0;
pub const REGION_CENTER_X: usize = 1;
pub const REGION_CENTER_Y: usize = 2;
pub const REGION_SIZE: usize = 3;
pub const REGION_NEXT_SIBLING: usize = 4;
pub const REGION_FIRST_CHILD: usize = 5;
pub const REGION_MASS: usize = 6;
pub const REGION_MASS_CENTER_X: usize = 7;
pub const REGION_MASS_CENTER_Y: usize = 8;

/// Maximum force magnitude clamp (`iterate.js:46`).
pub const MAX_FORCE: f64 = 10.0;
/// Barnes-Hut quad-tree subdivision retry limit (`iterate.js:37`).
pub const SUBDIVISION_ATTEMPTS: usize = 3;

/// Builds the flat `NodeMatrix`/`EdgeMatrix` pair consumed by the
/// iteration kernel, faithfully porting `graphToByteArrays`
/// (`helpers.js:117-171`).
///
/// `edges` and `weights` are parallel slices (`weights[i]` is the weight
/// of `edges[i]`, defaulting to `1.0` when `weights` is `None`, matching
/// the JS `getEdgeWeight` default). `init` supplies the starting `(x, y)`
/// coordinates per node index. The edge matrix stores each endpoint as
/// its `NodeMatrix` offset (`index * PPN`), not the raw node id.
pub fn graph_to_matrices(
    n_nodes: usize,
    edges: &[(u32, u32)],
    weights: Option<&[f64]>,
    init: &[(f64, f64)],
) -> (Vec<f32>, Vec<f32>) {
    let mut node_matrix = vec![0f32; n_nodes * PPN];
    let mut edge_matrix = vec![0f32; edges.len() * PPE];

    // Iterate through nodes (helpers.js:129-146).
    for (i, &(x, y)) in init.iter().enumerate().take(n_nodes) {
        let j = i * PPN;
        node_matrix[j + NODE_X] = x as f32;
        node_matrix[j + NODE_Y] = y as f32;
        node_matrix[j + NODE_DX] = 0.0;
        node_matrix[j + NODE_DY] = 0.0;
        node_matrix[j + NODE_OLD_DX] = 0.0;
        node_matrix[j + NODE_OLD_DY] = 0.0;
        node_matrix[j + NODE_MASS] = 1.0;
        node_matrix[j + NODE_CONVERGENCE] = 1.0;
        node_matrix[j + NODE_SIZE] = 1.0;
        node_matrix[j + NODE_FIXED] = 0.0;
    }

    // Iterate through edges (helpers.js:149-165).
    for (e, &(u, v)) in edges.iter().enumerate() {
        let sj = u as usize * PPN;
        let tj = v as usize * PPN;
        let weight = weights.map_or(1.0, |w| w[e]);

        // Incrementing mass to be a node's weighted degree: JS does
        // `NodeMatrix[j] += weight` on a Float32Array, i.e.
        // read-widen-add-narrow, not accumulate in f64 then narrow once.
        let sj_mass = node_matrix[sj + NODE_MASS] as f64 + weight;
        node_matrix[sj + NODE_MASS] = sj_mass as f32;
        let tj_mass = node_matrix[tj + NODE_MASS] as f64 + weight;
        node_matrix[tj + NODE_MASS] = tj_mass as f32;

        let ej = e * PPE;
        edge_matrix[ej + EDGE_SOURCE] = sj as f32;
        edge_matrix[ej + EDGE_TARGET] = tj as f32;
        edge_matrix[ej + EDGE_WEIGHT] = weight as f32;
    }

    (node_matrix, edge_matrix)
}

#[cfg(test)]
#[allow(clippy::erasing_op, clippy::identity_op)]
mod tests {
    use super::*;

    // path graph 0-1-2, unit weights, fixed init coords
    fn build() -> (Vec<f32>, Vec<f32>) {
        graph_to_matrices(3, &[(0, 1), (1, 2)], None, &[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)])
    }

    #[test]
    fn node_matrix_layout_matches_graphology() {
        let (nm, _) = build();
        assert_eq!(nm.len(), 3 * PPN);
        // helpers.js:129-145 slot init: x, y, dx=0, dy=0, old_dx=0, old_dy=0,
        // mass=1 (then += weighted degree), convergence=1, size=1, fixed=0
        assert_eq!(nm[0 * PPN + NODE_X], 0.1f32);
        assert_eq!(nm[0 * PPN + NODE_Y], 0.2f32);
        assert_eq!(nm[0 * PPN + NODE_CONVERGENCE], 1.0);
        assert_eq!(nm[0 * PPN + NODE_SIZE], 1.0);
        assert_eq!(nm[0 * PPN + NODE_FIXED], 0.0);
        // mass = 1 + weighted degree (helpers.js:156-164): ends=1, middle=2
        assert_eq!(nm[0 * PPN + NODE_MASS], 2.0);
        assert_eq!(nm[1 * PPN + NODE_MASS], 3.0);
        assert_eq!(nm[2 * PPN + NODE_MASS], 2.0);
    }

    #[test]
    fn edge_matrix_stores_offsets_not_ids() {
        let (_, em) = build();
        assert_eq!(em.len(), 2 * PPE);
        // helpers.js:149-164: source/target are NodeMatrix offsets (index * PPN)
        assert_eq!(em[0 * PPE + EDGE_SOURCE], 0.0);
        assert_eq!(em[0 * PPE + EDGE_TARGET], (1 * PPN) as f32);
        assert_eq!(em[1 * PPE + EDGE_SOURCE], (1 * PPN) as f32);
        assert_eq!(em[1 * PPE + EDGE_TARGET], (2 * PPN) as f32);
        assert_eq!(em[0 * PPE + EDGE_WEIGHT], 1.0);
    }

    #[test]
    fn weights_feed_mass_and_edge_matrix() {
        let (nm, em) = graph_to_matrices(2, &[(0, 1)], Some(&[2.5]), &[(0.0, 0.0), (1.0, 1.0)]);
        assert_eq!(em[EDGE_WEIGHT], 2.5f32);
        assert_eq!(nm[0 * PPN + NODE_MASS], 3.5f32); // 1 + 2.5, f32-rounded
    }

    #[test]
    fn coords_round_to_f32_like_float32array_stores() {
        let (nm, _) = graph_to_matrices(1, &[], None, &[(0.1234567890123, 0.0)]);
        assert_eq!(nm[NODE_X], 0.1234567890123f64 as f32);
    }
}
