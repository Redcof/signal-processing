#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

run() {
    local label="$1"
    local script="$2"
    echo ""
    echo "=========================================="
    echo "  $label"
    echo "=========================================="
    python "$script"
}

run "Percolation Test"              percolation_threshold/precollation_test.py
run "Percolation All Experiment Plots" percolation_threshold/percolation_all_experiment_plots.py
run "Multiscale Plots"              percolation_threshold/multiscale_plots.py
run "Weight Search"                 weight_search/weight_search.py
run "Weight Search (100 Seeds)"     weight_search/weight_search_100seeds.py
run "Synthetic Clutter Validation"  clutter_score_validation_ablation/sythentic_clutter_validation.py
run "Ablation Complete"             clutter_score_validation_ablation/ablation_complete.py

echo ""
echo "=========================================="
echo "  All scripts completed."
echo "=========================================="