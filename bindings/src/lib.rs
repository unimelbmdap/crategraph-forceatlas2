//! PyO3 binding: `_native.layout_raw`, the low-level entry point behind the
//! public Python `crategraph_forceatlas2.layout()` wrapper.
//!
//! All validation lives in the Python wrapper (`crategraph_forceatlas2/__init__.py`);
//! this binding trusts its inputs and only does the mechanical work of:
//! copying numpy arrays into Rust-owned buffers *before* releasing the GIL
//! (so no Python object is ever touched off-thread), building a `Settings`
//! from the (already-validated) camelCase dict, running the kernel, and
//! widening the resulting f32 positions back into a freshly allocated
//! `(n, 2)` f64 ndarray.
//!
//! `#[allow(clippy::useless_conversion)]` at the crate level below works
//! around a known pyo3 0.22 `#[pyfunction]`/`#[pymodule]` macro-expansion
//! false positive: the generated wrapper code triggers this lint even
//! though nothing in this file performs a real no-op conversion.
#![allow(clippy::useless_conversion)]

use std::cell::RefCell;

use cfa2_kernel::layout::run_layout;
use cfa2_kernel::matrices::{graph_to_matrices, NODE_X, NODE_Y, PPN};
use cfa2_kernel::settings::Settings;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

/// Builds a `Settings` from the Python-side camelCase dict. The dict has
/// already been validated (keys checked against `VALID_SETTINGS`, values
/// type/range-checked) by the Python wrapper, so this just maps names and
/// falls back to `Settings::default()` for anything absent.
fn settings_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Settings> {
    let mut s = Settings::default();

    macro_rules! set {
        ($key:literal, $field:ident) => {
            if let Some(v) = dict.get_item($key)? {
                s.$field = v.extract()?;
            }
        };
    }

    set!("linLogMode", lin_log_mode);
    set!("outboundAttractionDistribution", outbound_attraction_distribution);
    set!("adjustSizes", adjust_sizes);
    set!("edgeWeightInfluence", edge_weight_influence);
    set!("scalingRatio", scaling_ratio);
    set!("strongGravityMode", strong_gravity_mode);
    set!("gravity", gravity);
    set!("slowDown", slow_down);
    set!("barnesHutOptimize", barnes_hut_optimize);
    set!("barnesHutTheta", barnes_hut_theta);

    Ok(s)
}

/// Runs a ForceAtlas2 layout and returns the final `(n, 2)` f64 positions.
///
/// Every array argument is copied into a Rust-owned buffer up front, before
/// any GIL release, so nothing borrowed from Python is ever touched from a
/// background context. Without `progress` the whole iteration loop runs
/// inside one `py.allow_threads` block; with `progress` each iteration gets
/// its own `allow_threads` block and the GIL is reacquired afterwards to
/// invoke the callback -- a raising callback aborts the loop and the
/// original Python exception (not a generic error) is re-raised.
#[pyfunction]
#[pyo3(signature = (n_nodes, edges, weights, init, iterations, threads, settings, progress))]
#[allow(clippy::too_many_arguments)]
fn layout_raw<'py>(
    py: Python<'py>,
    n_nodes: usize,
    edges: PyReadonlyArray2<'py, u32>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init: PyReadonlyArray2<'py, f64>,
    iterations: u32,
    threads: usize,
    settings: Bound<'py, PyDict>,
    progress: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Copy every input into Rust-owned buffers before any GIL release.
    let edges_vec: Vec<(u32, u32)> =
        edges.as_array().rows().into_iter().map(|r| (r[0], r[1])).collect();
    let weights_vec: Option<Vec<f64>> = weights.map(|w| w.as_array().to_vec());
    let init_vec: Vec<(f64, f64)> =
        init.as_array().rows().into_iter().map(|r| (r[0], r[1])).collect();
    let s = settings_from_dict(&settings)?;

    let (mut nm, em) = graph_to_matrices(n_nodes, &edges_vec, weights_vec.as_deref(), &init_vec);

    match progress {
        None => {
            py.allow_threads(|| run_layout(&s, &mut nm, &em, iterations, threads, |_, _| Ok(())))
                .expect("no-op callback never aborts");
        }
        Some(callback) => {
            // Stash a raised exception from the callback here so the
            // ORIGINAL Python exception (not a generic error) can be
            // re-raised once the abort has unwound back to Python code.
            let caught_err: RefCell<Option<PyErr>> = RefCell::new(None);

            'iterations: for i in 1..=iterations {
                // One iteration per `allow_threads` block: run it off-GIL,
                // then reacquire the GIL (implicitly, since we're back on
                // the calling thread) to invoke the progress callback.
                py.allow_threads(|| run_layout(&s, &mut nm, &em, 1, threads, |_, _| Ok(())))
                    .expect("no-op callback never aborts");

                if let Err(e) = callback.call1((i, iterations)) {
                    *caught_err.borrow_mut() = Some(e);
                    break 'iterations;
                }
            }

            if let Some(err) = caught_err.into_inner() {
                return Err(err);
            }
        }
    }

    let out = PyArray2::<f64>::zeros_bound(py, (n_nodes, 2), false);
    // Safety: `out` was just allocated above and no other reference to it
    // exists yet, so writing through a mutable view here is exclusive.
    unsafe {
        let mut view = out.as_array_mut();
        for i in 0..n_nodes {
            let j = i * PPN;
            view[[i, 0]] = f64::from(nm[j + NODE_X]);
            view[[i, 1]] = f64::from(nm[j + NODE_Y]);
        }
    }
    Ok(out)
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(layout_raw, m)?)?;
    Ok(())
}
