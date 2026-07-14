# SNOCS: A Deterministic Scale-Normalised Occupancy Clutter Score

Pooling-based clutter scoring for binary spatial fields using four physical
scales [10, 50, 200, 500 mm].  All analysis scripts live under `snoc/` and
share a common library in `snoc/core/`.

---

## Environment setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install required packages
pip install -r requirements.txt

# 3. (Optional) GPU-accelerated batch processing
#    Uncomment the torch line in requirements.txt, then:
pip install torch                  # or follow https://pytorch.org for CUDA/MPS builds
```

Python 3.10+ is recommended.

---

## Running scripts

All scripts must be executed **from the `snoc/` directory** so that
`from core.xxx import ...` and `from utils import ...` resolve correctly.

```bash
cd snoc/
```

### Percolation threshold validation

| Script | One-line description | Run |
|---|---|---|
| `percolation_threshold/percolation_threshold.py` | Validates that the SNOCS clutter score detects the theoretical 2D-lattice percolation phase transition (p_c ≈ 0.5927) via inflection-point and derivative analysis. | `python percolation_threshold/percolation_threshold.py` |
| `percolation_threshold/multiscale_plots.py` | Generates visualisations of avg- and max-pooled maps at key occupancy values and for blob/fragmented synthetic scenes. | `python percolation_threshold/multiscale_plots.py` |
| `percolation_threshold/precollation_test.py` | GPU-accelerated (PyTorch) batch percolation sweep; benchmarks CPU vs device throughput and writes results to `percolation_summary.csv`. | `python percolation_threshold/precollation_test.py` |
| `percolation_threshold/percolation_all_experiment_plots.py` | Reads `percolation_summary.csv` and produces figure | `python percolation_threshold/percolation_all_experiment_plots.py` |

### Clutter score validation & ablation

| Script | One-line description | Run |
|---|---|---|
| `clutter_score_validation_ablation/sythentic_clutter_validation.py` | Three-stage empirical validation protocol (occupancy sweep, fragmentation series, archetype sanity check) of the SNOCS score on synthetic binary fields. | `python clutter_score_validation_ablation/sythentic_clutter_validation.py` |
| `clutter_score_validation_ablation/ablation_complete.py` | Full ablation study of scale selection, feature type, and image resolution on 1024×1024 synthetic fields using the 12-dim SNOCS feature set. | `python clutter_score_validation_ablation/ablation_complete.py` |

### Weight search

| Script | One-line description | Run |
|---|---|---|
| `weight_search/weight_search.py` | Finds optimal SNOCS feature weights (density / variance / fragmentation) via exhaustive grid search and Bayesian optimisation (scikit-optimize GP surrogate). | `python weight_search/weight_search.py` |
| `weight_search/weight_search_100seeds.py` | Repeats the full weight search across 100 random seeds to quantify weight stability and produces aggregate statistics and box-plot figures. | `python weight_search/weight_search_100seeds.py` |

---

## Core library (`core/`)

Not run directly — imported by all scripts above.

| Module | Purpose |
|---|---|
| `core/pooling.py` | `avg_pool`, `max_pool`, `extract_features` (16-dim), `extract_features_12` (12-dim), optional PyTorch batch variant. |
| `core/percolation.py` | `PC_THEORY` constant, largest-cluster fraction, spanning detection, Savitzky-Golay smoothing, derivative and inflection utilities. |
| `core/synthesis.py` | Bernoulli fields, blob fields, fragmented dot fields, ground-truth label `gt_label`. |
| `core/timing.py` | `timeit` decorator, `TimerContext` context manager, `get_device` (MPS → CUDA → CPU). |

---

## Output

Each script writes figures to a `results/` subdirectory alongside itself.
Scripts that generate thesis figures additionally copy select outputs to
the project root via `utils.copy_figure`.

## Citing SNOCS

If you find this repository useful, please consider giving a star :star: and citation:

```
@misc{sardar2026snocs,
  title={{SNOCS}},
  author={Sardar, Soumen},
  year={2026},
  url={https://github.com/Redcof/signal-processing/tree/main/snoc},
}
```