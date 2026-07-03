//! Settings for ForceAtlas2 layout algorithm, with defaults from graphology.

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub lin_log_mode: bool,
    pub outbound_attraction_distribution: bool,
    pub adjust_sizes: bool,
    pub edge_weight_influence: f64,
    pub scaling_ratio: f64,
    pub strong_gravity_mode: bool,
    pub gravity: f64,
    pub slow_down: f64,
    pub barnes_hut_optimize: bool,
    pub barnes_hut_theta: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            lin_log_mode: false,
            outbound_attraction_distribution: false,
            adjust_sizes: false,
            edge_weight_influence: 1.0,
            scaling_ratio: 1.0,
            strong_gravity_mode: false,
            gravity: 1.0,
            slow_down: 1.0,
            barnes_hut_optimize: false,
            barnes_hut_theta: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_default_graphology_values() {
        let defaults = Settings::default();

        // From graphology-layout-forceatlas2/defaults.js
        assert!(!defaults.lin_log_mode);
        assert!(!defaults.outbound_attraction_distribution);
        assert!(!defaults.adjust_sizes);
        assert_eq!(defaults.edge_weight_influence, 1.0);
        assert_eq!(defaults.scaling_ratio, 1.0);
        assert!(!defaults.strong_gravity_mode);
        assert_eq!(defaults.gravity, 1.0);
        assert_eq!(defaults.slow_down, 1.0);
        assert!(!defaults.barnes_hut_optimize);
        assert_eq!(defaults.barnes_hut_theta, 0.5);
    }
}
