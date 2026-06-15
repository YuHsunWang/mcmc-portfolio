"""
GPU-parallel MCMC demo for the L1 preference model.

This script is a portfolio-oriented, tensorized version of the notebook workflow.
It keeps the same modeling idea, but replaces per-user dictionaries and DataFrame
row-wise operations with batched array operations that can run on GPU via CuPy.

If CuPy/CUDA is unavailable, it automatically falls back to NumPy so the script
still runs as a reproducible CPU demo.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cupy as cp

    xp = cp
    GPU_AVAILABLE = True
except Exception:
    cp = None
    xp = np
    GPU_AVAILABLE = False


TOP10 = [5, 9, 19, 11, 20, 23, 6, 21, 25, 22]
N_CATE = 10
SCRIPT_DIR = Path(__file__).resolve().parent


def to_cpu(arr):
    if GPU_AVAILABLE:
        return cp.asnumpy(arr)
    return np.asarray(arr)


def softmax_with_outside_log_prob(mu, choice):
    """Return log P(choice) under multinomial logit with outside utility 0."""
    row_max = xp.maximum(xp.max(mu, axis=1), 0.0)
    exp_mu = xp.exp(mu - row_max[:, None])
    exp_outside = xp.exp(-row_max)
    log_denom = row_max + xp.log(exp_outside + xp.sum(exp_mu, axis=1))
    return mu[xp.arange(mu.shape[0]), choice] - log_denom


def outside_log_prob(mu):
    """Return log probability of the outside/stop option."""
    row_max = xp.maximum(xp.max(mu, axis=1), 0.0)
    exp_mu = xp.exp(mu - row_max[:, None])
    exp_outside = xp.exp(-row_max)
    log_denom = row_max + xp.log(exp_outside + xp.sum(exp_mu, axis=1))
    return -log_denom


def utility(log_a, log_s_raw, log_switch, user_idx, state_x, switch_cost_x):
    """Vectorized marginal utility for all observations."""
    a = xp.exp(log_a[user_idx])
    s = 1.0 + xp.exp(log_s_raw[user_idx])
    c_switch = xp.exp(log_switch[user_idx])[:, None]

    power1 = s / (s - 1.0)
    power2 = 1.0 / (s - 1.0)
    power3 = -1.0 / s

    term_sum = xp.sum(a * xp.power(state_x, power1[:, None]), axis=1)
    nonzero_state = xp.sum(state_x, axis=1) > 0

    mu_first = xp.power(a, power1[:, None])
    mu_later = (
        xp.power(xp.maximum(term_sum, 1e-12), power2)[:, None]
        * a
        * xp.power(state_x + 1.0, power3[:, None])
        - c_switch * switch_cost_x
    )
    return xp.where(nonzero_state[:, None], mu_later, mu_first)


def compute_log_likelihood(
    log_a,
    log_s_raw,
    log_switch,
    obs_user,
    obs_x,
    obs_switch_cost,
    obs_choice,
    stop_user,
    stop_x,
    stop_switch_cost,
    n_user,
):
    obs_mu = utility(log_a, log_s_raw, log_switch, obs_user, obs_x, obs_switch_cost)
    obs_logp = softmax_with_outside_log_prob(obs_mu, obs_choice)
    user_ll = xp.bincount(obs_user, weights=obs_logp, minlength=n_user)

    stop_mu = utility(log_a, log_s_raw, log_switch, stop_user, stop_x, stop_switch_cost)
    stop_logp = outside_log_prob(stop_mu)
    user_ll = user_ll + xp.bincount(stop_user, weights=stop_logp, minlength=n_user)
    return user_ll


def normal_log_prior_scalar(x, mean, std):
    var = std**2
    return -0.5 * ((x - mean) ** 2 / var + xp.log(2.0 * xp.pi * var))


def mvn_log_prior_rows(x, mean, cov_inv, log_det_cov):
    diff = x - mean[None, :]
    quad = xp.sum((diff @ cov_inv) * diff, axis=1)
    k = x.shape[1]
    return -0.5 * (k * xp.log(2.0 * xp.pi) + log_det_cov + quad)


def build_transition_matrix(sequences):
    trans = np.zeros((N_CATE, N_CATE), dtype=np.float64)
    for seq in sequences:
        for i, j in zip(seq[:-1], seq[1:]):
            trans[j, i] += 1.0
    denom = trans.sum(axis=0)
    trans = np.divide(trans, denom, out=np.zeros_like(trans), where=denom != 0)
    return trans.T


def prepare_training_tensors(data_path: Path, n_users: int, seed: int):
    rng = np.random.default_rng(seed)

    df = pd.read_parquet(data_path)
    df["cat"] = df["category"].astype(int)
    df = df[df["cat"].isin(TOP10)].copy()
    map_dict = {key: value for value, key in enumerate(TOP10)}
    df["c"] = df["cat"].map(map_dict).astype(int)

    train = df[df["time"] < "2022-02-01"].copy()
    test = df[df["time"] >= "2022-02-01"].copy()
    common_users = sorted(set(train["ID"]).intersection(test["ID"]))
    train = train[train["ID"].isin(common_users)]

    grouped = (
        train.sort_values(["ID", "session_id", "order"])
        .groupby(["ID", "session_id"])["c"]
        .apply(list)
        .reset_index()
    )
    grouped["len"] = grouped["c"].str.len()
    grouped = grouped[grouped["len"] <= 100].copy()

    eligible_users = np.array(sorted(grouped["ID"].unique()))
    if n_users > len(eligible_users):
        raise ValueError(f"n_users={n_users} exceeds available users={len(eligible_users)}")
    selected_users = np.sort(rng.choice(eligible_users, size=n_users, replace=False))
    user_to_idx = {cid: i for i, cid in enumerate(selected_users)}
    grouped = grouped[grouped["ID"].isin(selected_users)].copy()

    sequences = grouped["c"].tolist()
    transition = build_transition_matrix(sequences)
    switching_cost = 1.0 - transition
    np.fill_diagonal(switching_cost, 0.0)

    obs_user, obs_choice, obs_x, obs_switch = [], [], [], []
    stop_user, stop_x, stop_switch = [], [], []

    for cid, seq in zip(grouped["ID"].to_numpy(), grouped["c"]):
        u = user_to_idx[cid]
        state = np.zeros(N_CATE, dtype=np.float32)
        prev_choice = None

        for choice in seq:
            if prev_choice is None:
                switch_row = np.zeros(N_CATE, dtype=np.float32)
            else:
                switch_row = switching_cost[prev_choice].astype(np.float32)

            obs_user.append(u)
            obs_choice.append(choice)
            obs_x.append(state.copy())
            obs_switch.append(switch_row)

            state[choice] += 1.0
            prev_choice = choice

        final_switch = (
            np.zeros(N_CATE, dtype=np.float32)
            if prev_choice is None
            else switching_cost[prev_choice].astype(np.float32)
        )
        stop_user.append(u)
        stop_x.append(state.copy())
        stop_switch.append(final_switch)

    tensors = {
        "obs_user": xp.asarray(np.array(obs_user, dtype=np.int32)),
        "obs_choice": xp.asarray(np.array(obs_choice, dtype=np.int32)),
        "obs_x": xp.asarray(np.array(obs_x, dtype=np.float32)),
        "obs_switch": xp.asarray(np.array(obs_switch, dtype=np.float32)),
        "stop_user": xp.asarray(np.array(stop_user, dtype=np.int32)),
        "stop_x": xp.asarray(np.array(stop_x, dtype=np.float32)),
        "stop_switch": xp.asarray(np.array(stop_switch, dtype=np.float32)),
        "switching_cost": switching_cost,
        "selected_users": selected_users,
        "n_sessions": int(len(grouped)),
        "n_observations": int(len(obs_choice)),
    }
    return tensors


def run_mcmc(tensors, n_iter: int, seed: int, output_dir: Path):
    rng = xp.random.default_rng(seed)
    n_user = len(tensors["selected_users"])
    n_trace_users = min(5, n_user)

    log_a = rng.normal(0.0, 0.5, size=(n_user, N_CATE))
    log_s_raw = rng.normal(np.log(4.0), 0.2, size=n_user)
    log_switch = rng.normal(0.0, 0.4, size=n_user)

    theta_a = xp.mean(log_a, axis=0)
    cov_a = xp.eye(N_CATE) * 2.0
    cov_a_inv = xp.linalg.inv(cov_a)
    log_det_cov_a = xp.linalg.slogdet(cov_a)[1]

    theta_s = xp.mean(log_s_raw)
    sigma_s = xp.maximum(xp.std(log_s_raw), 0.2)
    theta_switch = xp.mean(log_switch)
    sigma_switch = xp.maximum(xp.std(log_switch), 0.2)

    current_ll = compute_log_likelihood(
        log_a,
        log_s_raw,
        log_switch,
        tensors["obs_user"],
        tensors["obs_x"],
        tensors["obs_switch"],
        tensors["obs_choice"],
        tensors["stop_user"],
        tensors["stop_x"],
        tensors["stop_switch"],
        n_user,
    )

    accept_trace = np.zeros((n_iter, 3), dtype=np.float64)
    accuracy_trace = np.zeros(n_iter, dtype=np.float64)
    theta_a_trace = np.zeros((n_iter, N_CATE), dtype=np.float64)
    theta_s_trace = np.zeros(n_iter, dtype=np.float64)
    theta_switch_trace = np.zeros(n_iter, dtype=np.float64)
    demo_a_trace = np.zeros((n_iter, n_trace_users, N_CATE), dtype=np.float64)
    demo_s_trace = np.zeros((n_iter, n_trace_users), dtype=np.float64)
    demo_switch_trace = np.zeros((n_iter, n_trace_users), dtype=np.float64)

    start = time.perf_counter()
    for i in range(n_iter):
        prior_a = mvn_log_prior_rows(log_a, theta_a, cov_a_inv, log_det_cov_a)
        proposal_a = log_a + rng.normal(0.0, 0.15, size=log_a.shape)
        ll_a = compute_log_likelihood(
            proposal_a,
            log_s_raw,
            log_switch,
            tensors["obs_user"],
            tensors["obs_x"],
            tensors["obs_switch"],
            tensors["obs_choice"],
            tensors["stop_user"],
            tensors["stop_x"],
            tensors["stop_switch"],
            n_user,
        )
        prior_a_star = mvn_log_prior_rows(proposal_a, theta_a, cov_a_inv, log_det_cov_a)
        accept_a = xp.log(rng.random(n_user)) < (ll_a + prior_a_star - current_ll - prior_a)
        log_a = xp.where(accept_a[:, None], proposal_a, log_a)
        current_ll = xp.where(accept_a, ll_a, current_ll)

        theta_a = xp.mean(log_a, axis=0)
        centered_a = log_a - theta_a[None, :]
        cov_a = (centered_a.T @ centered_a) / xp.maximum(n_user - 1, 1) + xp.eye(N_CATE) * 1e-3
        cov_a_inv = xp.linalg.inv(cov_a)
        log_det_cov_a = xp.linalg.slogdet(cov_a)[1]

        prior_s = normal_log_prior_scalar(log_s_raw, theta_s, sigma_s)
        proposal_s = log_s_raw + rng.normal(0.0, 0.12, size=log_s_raw.shape)
        ll_s = compute_log_likelihood(
            log_a,
            proposal_s,
            log_switch,
            tensors["obs_user"],
            tensors["obs_x"],
            tensors["obs_switch"],
            tensors["obs_choice"],
            tensors["stop_user"],
            tensors["stop_x"],
            tensors["stop_switch"],
            n_user,
        )
        prior_s_star = normal_log_prior_scalar(proposal_s, theta_s, sigma_s)
        accept_s = xp.log(rng.random(n_user)) < (ll_s + prior_s_star - current_ll - prior_s)
        log_s_raw = xp.where(accept_s, proposal_s, log_s_raw)
        current_ll = xp.where(accept_s, ll_s, current_ll)
        theta_s = xp.mean(log_s_raw)
        sigma_s = xp.maximum(xp.std(log_s_raw), 0.2)

        prior_switch = normal_log_prior_scalar(log_switch, theta_switch, sigma_switch)
        proposal_switch = log_switch + rng.normal(0.0, 0.12, size=log_switch.shape)
        ll_switch = compute_log_likelihood(
            log_a,
            log_s_raw,
            proposal_switch,
            tensors["obs_user"],
            tensors["obs_x"],
            tensors["obs_switch"],
            tensors["obs_choice"],
            tensors["stop_user"],
            tensors["stop_x"],
            tensors["stop_switch"],
            n_user,
        )
        prior_switch_star = normal_log_prior_scalar(proposal_switch, theta_switch, sigma_switch)
        accept_switch = xp.log(rng.random(n_user)) < (
            ll_switch + prior_switch_star - current_ll - prior_switch
        )
        log_switch = xp.where(accept_switch, proposal_switch, log_switch)
        current_ll = xp.where(accept_switch, ll_switch, current_ll)
        theta_switch = xp.mean(log_switch)
        sigma_switch = xp.maximum(xp.std(log_switch), 0.2)

        obs_mu = utility(
            log_a,
            log_s_raw,
            log_switch,
            tensors["obs_user"],
            tensors["obs_x"],
            tensors["obs_switch"],
        )
        pred = xp.argmax(obs_mu, axis=1)
        accuracy = xp.mean(pred == tensors["obs_choice"])

        accept_trace[i] = [
            float(to_cpu(xp.mean(accept_a))),
            float(to_cpu(xp.mean(accept_s))),
            float(to_cpu(xp.mean(accept_switch))),
        ]
        accuracy_trace[i] = float(to_cpu(accuracy))
        theta_a_trace[i] = to_cpu(theta_a)
        theta_s_trace[i] = float(to_cpu(theta_s))
        theta_switch_trace[i] = float(to_cpu(theta_switch))
        demo_a_trace[i] = np.exp(to_cpu(log_a[:n_trace_users]))
        demo_s_trace[i] = 1.0 + np.exp(to_cpu(log_s_raw[:n_trace_users]))
        demo_switch_trace[i] = np.exp(to_cpu(log_switch[:n_trace_users]))

        if (i + 1) % max(1, n_iter // 10) == 0 or i == 0:
            print(
                f"iter={i+1:>4}/{n_iter} "
                f"acc_a={accept_trace[i,0]:.3f} "
                f"acc_s={accept_trace[i,1]:.3f} "
                f"acc_switch={accept_trace[i,2]:.3f} "
                f"train_acc={accuracy_trace[i]:.3f}"
            )

    elapsed = time.perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "gpu_mcmc_results.npz",
        selected_users=tensors["selected_users"],
        final_a=np.exp(to_cpu(log_a)),
        final_s=1.0 + np.exp(to_cpu(log_s_raw)),
        final_switch=np.exp(to_cpu(log_switch)),
        accept_trace=accept_trace,
        accuracy_trace=accuracy_trace,
        theta_a_trace=theta_a_trace,
        theta_s_trace=theta_s_trace,
        theta_switch_trace=theta_switch_trace,
        demo_user_ids=tensors["selected_users"][:n_trace_users],
        demo_a_trace=demo_a_trace,
        demo_s_trace=demo_s_trace,
        demo_switch_trace=demo_switch_trace,
        switching_cost=tensors["switching_cost"],
    )

    summary = {
        "backend": "cupy-cuda" if GPU_AVAILABLE else "numpy-cpu-fallback",
        "n_users": int(n_user),
        "n_sessions": tensors["n_sessions"],
        "n_observations": tensors["n_observations"],
        "n_iter": int(n_iter),
        "elapsed_seconds": elapsed,
        "final_accuracy": float(accuracy_trace[-1]),
        "final_acceptance": {
            "a_u": float(accept_trace[-1, 0]),
            "s_u": float(accept_trace[-1, 1]),
            "C_switch_u": float(accept_trace[-1, 2]),
        },
    }
    (output_dir / "gpu_mcmc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="GPU-parallel MCMC for L1 preference model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=SCRIPT_DIR / "data" / "itemPV_202201to202204.parquet",
        help="Path to the synthetic browsing parquet file.",
    )
    parser.add_argument("--n-users", type=int, default=1000, help="Number of users to sample.")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of MCMC iterations.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Directory for result files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("backend:", "cupy-cuda" if GPU_AVAILABLE else "numpy-cpu-fallback")
    tensors = prepare_training_tensors(args.data, args.n_users, args.seed)
    summary = run_mcmc(tensors, args.n_iter, args.seed, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
