use pyo3::prelude::*;

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__kernel_ready__", false)?;
    Ok(())
}
