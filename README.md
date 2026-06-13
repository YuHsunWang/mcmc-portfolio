# Bayesian MCMC Preference Model for Browsing Sequences

This project demonstrates a Bayesian preference model for user browsing sequences. It estimates user-level category preference, diversity/elasticity, and switching-cost parameters from session-level browsing logs using a Metropolis-Hastings MCMC sampler.

The repository is designed as a portfolio project. It includes synthetic browsing data, a reproducible research notebook, a GPU-ready implementation, benchmark comparisons, and visualization notebooks for parameter dynamics.

## Project Highlights

- Builds a session-level sequence model from browsing events.
- Estimates user-specific parameters:
  - preference strength: `$a_u$`
  - diversity/elasticity: `$s_u = 1 + z_u,\ z_u > 0$`
  - switching cost: `$C^{switch}_u$`
- Uses an empirical transition matrix as a switching-friction proxy.
- Implements Metropolis-Hastings updates for user-level parameters.
- Provides both notebook and tensorized GPU-ready implementations.
- Compares the MCMC model with popularity, user-frequency, and Markov baselines on held-out test sessions.

## Repository Structure

```text
mcmc_portfolio/
|-- MCMC_L1_portfolio.ipynb
|-- MCMC_portfolio_demo.ipynb
|-- MCMC_iterative_flow.md
|-- RESULTS.md
|-- mcmc_gpu_parallel.py
|-- generate_realistic_synthetic_data.py
|-- requirements.txt
|-- requirements_gpu.txt
|-- data_summary.json
|-- robustness_summary_100u_10000iter.csv
|-- data/
|   |-- itemPV_202201to202204.parquet
|   |-- itemPV_202201to202204_balanced.parquet
|   |-- corr.csv
|   `-- sample_preview.csv
|-- outputs_100u_10000iter/
|   |-- benchmark_comparison.csv
|   |-- gpu_mcmc_results.npz
|   `-- gpu_mcmc_summary.json
|-- outputs_realistic_100u_10000iter/
|   |-- benchmark_comparison.csv
|   |-- gpu_mcmc_results.npz
|   `-- gpu_mcmc_summary.json
`-- README.md
```

## Data

This project uses synthetic browsing data for reproducibility and public sharing. The primary dataset is generated with a realistic sparse browsing pattern: each user has a small number of dominant interests, and sessions tend to stay within one intent cluster with occasional related-category switches.

Main data file:

```text
data/itemPV_202201to202204.parquet
```

The earlier balanced synthetic dataset is kept as a comparison scenario:

```text
data/itemPV_202201to202204_balanced.parquet
```

Required columns:

| Column | Description |
|---|---|
| `CID` | User/customer ID |
| `session_id` | Browsing session ID |
| `order` | Event order within a session |
| `time` | Event timestamp |
| `g_class4` | Original category ID |
| `succ_joined` | Event indicator used to rebuild order |
| `category_name` | Human-readable synthetic category label |
| `is_fake_data` | Marks the row as synthetic |

Plot labels use English category names to avoid font-rendering issues:

```text
Toys, Home, Vehicles, Electronics, Travel,
Appliances, Books, Mobile, Lifestyle, Gaming
```

Regenerate the realistic synthetic data:

```bash
python generate_realistic_synthetic_data.py
```

## Model Overview

For each user `$u$`, the model estimates:

- `$a_u$`: category preference vector
- `$s_u$`: diversity/elasticity parameter, constrained as `$s_u = 1 + z_u$`
- `$C^{switch}_u$`: user-level switching-cost scale

The marginal utility for each category is converted into choice probabilities using a multinomial logit model with an outside/stop option.

The MCMC sampler iteratively:

1. proposes user-level parameters by lognormal random walk,
2. computes likelihood and prior ratios,
3. applies Metropolis-Hastings accept/reject decisions,
4. updates population-level hyperparameters,
5. stores posterior samples and diagnostics.

See [MCMC_iterative_flow.md](MCMC_iterative_flow.md) for the iteration flowchart.

## Benchmarks

The demo notebook compares the MCMC model with simple held-out test benchmarks:

| Benchmark | Description |
|---|---|
| Global Popularity | Always predicts the most common training categories |
| User Most Frequent | Predicts each user's most frequent training categories |
| Markov Transition | Predicts next category using empirical transition matrix |
| MCMC Preference Model | Predicts using estimated user-level parameters |

Evaluation metrics:

- Top-1 next-category accuracy
- Top-3 next-category accuracy
- Sequence similarity

## Current Results

The included result files use `100 users` and `10,000 MCMC iterations` for each synthetic scenario.

| Scenario | Train Accuracy | MCMC Top-1 | MCMC Top-3 | Sequence Similarity | Top-1 vs Markov | Top-1 vs User Most Frequent | Top-1 vs Global Popularity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced synthetic | 50.05% | 48.13% | 64.73% | 0.873 | +0.72 pp | +31.47 pp | +35.27 pp |
| Realistic sparse synthetic | 70.97% | 67.30% | 85.34% | 0.814 | +4.11 pp | +25.63 pp | +54.01 pp |

The MCMC model consistently outperforms popularity-based and user-frequency baselines. In the realistic sparse-interest setting, it also outperforms the Markov transition benchmark across Top-1 accuracy, Top-3 accuracy, and sequence similarity.

Full result files:

```text
outputs_100u_10000iter/benchmark_comparison.csv
outputs_realistic_100u_10000iter/benchmark_comparison.csv
robustness_summary_100u_10000iter.csv
```

See [RESULTS.md](RESULTS.md) for the full benchmark comparison and interpretation.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the main research notebook:

```text
MCMC_L1_portfolio.ipynb
```

Open the presentation-oriented demo notebook:

```text
MCMC_portfolio_demo.ipynb
```

Recommended quick demo settings:

```python
N_USERS = 500
N_ITER = 50
```

For a larger experiment:

```python
N_USERS = 5000
N_ITER = 1000
```

## GPU-Ready Version

The script `mcmc_gpu_parallel.py` is a tensorized implementation designed for GPU acceleration with CuPy. If CuPy/CUDA is unavailable, it automatically falls back to NumPy.

Small demo:

```bash
python mcmc_gpu_parallel.py --n-users 100 --n-iter 10
```

Reproduce the included 100-user realistic sparse result:

```bash
python mcmc_gpu_parallel.py --n-users 100 --n-iter 10000 --output-dir outputs_realistic_100u_10000iter
```

Run the balanced scenario:

```bash
python mcmc_gpu_parallel.py --data data/itemPV_202201to202204_balanced.parquet --n-users 100 --n-iter 10000 --output-dir outputs_100u_10000iter
```

Optional GPU dependency:

```bash
pip install cupy-cuda12x
```

Choose the CuPy package that matches your CUDA version.

## Running on Google Colab

1. Upload this folder to Google Drive or clone the repository.
2. Select `Runtime > Change runtime type > GPU` if using the GPU script.
3. Install dependencies:

```python
!pip install -r requirements.txt
!pip install cupy-cuda12x  # optional for GPU runtime
```

4. Run a demo:

```python
!python mcmc_gpu_parallel.py --n-users 500 --n-iter 50
```

## Portfolio Demo Suggestions

For a clean portfolio presentation, use:

1. MCMC iteration flowchart
2. acceptance-rate trace plot
3. parameter trace plots for selected users
4. posterior parameter histograms
5. held-out benchmark comparison table
6. benchmark bar chart

Suggested project summary:

> This project estimates heterogeneous user preferences from browsing sequences using a Bayesian MCMC model. It captures category preference, diversity, and switching cost, then evaluates predictive performance against popularity, user-frequency, and Markov transition baselines.

Suggested robustness statement:

> The model is evaluated under multiple synthetic browsing scenarios, including balanced category exposure and realistic sparse-interest browsing behavior. Across these settings, the MCMC preference model remains competitive with transition-based benchmarks and consistently outperforms simpler popularity and user-frequency baselines.

## Notes and Limitations

- The dataset is synthetic and intended for reproducible public demonstration.
- The switching-cost matrix is an empirical proxy based on transition frequency, not a causal estimate of true switching cost.
- Short demo runs are useful for visualization but are not enough for final statistical inference.
- For serious inference, use longer chains, burn-in, posterior averaging, and convergence diagnostics.
- Benchmark results depend on the synthetic data-generating process and should be interpreted as pipeline validation rather than real business performance.

## GitHub Publishing

The included `.gitignore` excludes temporary files, pickle backups, and smoke-test outputs while keeping the final compact result folders.

From inside this folder:

```bash
git init
git add .
git commit -m "Add Bayesian MCMC browsing preference portfolio project"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## License

This project is provided for portfolio and educational purposes. Add a license file if you plan to distribute or reuse it publicly.
