# Demo Results and Benchmark Comparison

This page summarizes the demo results included in this repository. Both experiments use:

```text
100 users
10,000 MCMC iterations
CPU fallback runtime
```

The goal is to compare the Bayesian MCMC preference model against simple browsing-sequence benchmarks on held-out test sessions.

## Benchmark Models

| Model | Description |
|---|---|
| Global Popularity | Always predicts the most frequent training categories |
| User Most Frequent | Predicts each user's most frequent training categories |
| Markov Transition | Predicts next category from the previous category |
| MCMC Preference Model | Uses estimated user-level preference, diversity, and switching-cost parameters |

## Scenario 1: Balanced Synthetic Data

Balanced synthetic data has broader category exposure and more dispersed browsing behavior.

| Model | Top-1 Accuracy | Top-3 Accuracy | Sequence Similarity |
|---|---:|---:|---:|
| MCMC Preference Model | 48.13% | 64.73% | 0.873 |
| Markov Transition | 47.41% | 60.85% | 0.883 |
| User Most Frequent | 16.67% | 45.33% | 0.159 |
| Global Popularity | 12.86% | 35.13% | 0.120 |

### Improvement Over Benchmarks

| Compared With | Top-1 Improvement | Top-3 Improvement | Sequence Similarity Difference |
|---|---:|---:|---:|
| Markov Transition | +0.72 pp | +3.88 pp | -0.010 |
| User Most Frequent | +31.47 pp | +19.40 pp | +0.714 |
| Global Popularity | +35.27 pp | +29.60 pp | +0.753 |

**Interpretation:**  
In the more dispersed browsing scenario, the MCMC model slightly outperforms the Markov baseline on Top-1 and Top-3 accuracy, while dramatically outperforming popularity-based and user-frequency baselines. The Markov model has slightly higher sequence similarity, suggesting that transition rules remain strong when browsing is broad.

## Scenario 2: Realistic Sparse-Interest Data

Realistic sparse-interest data is designed to better mimic human browsing: users focus on a few dominant interests, and sessions usually stay within related categories.

| Model | Top-1 Accuracy | Top-3 Accuracy | Sequence Similarity |
|---|---:|---:|---:|
| MCMC Preference Model | 67.30% | 85.34% | 0.814 |
| Markov Transition | 63.19% | 82.17% | 0.794 |
| User Most Frequent | 41.67% | 72.26% | 0.443 |
| Global Popularity | 13.29% | 41.56% | 0.118 |

### Improvement Over Benchmarks

| Compared With | Top-1 Improvement | Top-3 Improvement | Sequence Similarity Difference |
|---|---:|---:|---:|
| Markov Transition | +4.11 pp | +3.16 pp | +0.020 |
| User Most Frequent | +25.63 pp | +13.08 pp | +0.371 |
| Global Popularity | +54.01 pp | +43.78 pp | +0.695 |

**Interpretation:**  
In the more realistic sparse-interest scenario, the MCMC model outperforms all benchmark models across all three metrics. This shows that the model can capture both short-term transition behavior and user-level preference heterogeneity.

## Robustness Summary

| Scenario | MCMC Top-1 | MCMC Top-3 | Top-1 vs Markov | Top-1 vs User Most Frequent | Top-1 vs Global Popularity |
|---|---:|---:|---:|---:|---:|
| Balanced synthetic | 48.13% | 64.73% | +0.72 pp | +31.47 pp | +35.27 pp |
| Realistic sparse synthetic | 67.30% | 85.34% | +4.11 pp | +25.63 pp | +54.01 pp |

## Key Takeaway

The MCMC preference model remains competitive across different synthetic browsing scenarios and consistently outperforms simple popularity and user-frequency baselines. In the realistic sparse-interest setting, it also beats the Markov transition benchmark, suggesting that user-level preference and switching-cost parameters add predictive value beyond transition frequency alone.

These results should be interpreted as a portfolio demonstration on synthetic data, not as proof of real-world production performance.

