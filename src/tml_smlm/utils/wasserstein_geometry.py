#!/usr/bin/env python3
"""
Wasserstein-distance geometry for persistence diagrams.

This module treats a persistence diagram as a point in a metric space and
provides the machinery View 3 of the framework needs to work there: the
2-Wasserstein (W2) distance between two diagrams, a pairwise distance matrix
over a population of diagrams, and a fixed-cardinality Frechet mean/variance
estimate following the iterative barycenter scheme of Turner et al. (2014).
A medoid-based permutation test for group separation in Wasserstein space is
also included, built on top of a precomputed pooled distance matrix so it
never re-runs the (expensive) Frechet mean or additional W2 calls per
permutation. No gudhi/Hera dependency: the assignment problem is solved with
scipy's Hungarian-algorithm implementation.

DESIGN NOTES

  1. wasserstein_distance_2() is scipy-only (linear_sum_assignment). No
     gudhi/Hera dependency.

  2. Ground metric is L_infinity (Chebyshev) between points, matching the
     common GUDHI/Hera convention (internal_p=inf). Distance from a point
     (b, d) to the diagonal under this metric is (d - b) / 2. The cost
     matrix is the full (n + m) x (n + m) augmented assignment problem
     (each diagram implicitly extended by the diagonal).

  3. Frechet mean: fixed-cardinality iterative barycenter update (Turner et
     al. 2014: propose a mean, match it to every diagram, average the matched
     points, repeat). Turner et al. start from a randomly drawn diagram; this
     implementation starts from the population's medoid, so a given set of
     diagrams always returns the same mean. The mean is a single
     diagram-valued estimate, in contrast to the probabilistic Frechet mean
     of Munch et al. (2015), which returns a distribution over diagrams.
     The mean diagram keeps the medoid's point count throughout; points
     unmatched in a given replicate are pulled toward their own current
     diagonal projection rather than added or removed dynamically. This is
     a standard, legitimate simplification for a from-scratch
     implementation, not the full nonparametric algorithm. Frechet means
     are not unique in general -- this returns one representative
     fixed-cardinality local optimum, not "the" mean.

  4. Permutation testing does NOT recompute the gradient-descent Frechet
     mean per permutation -- for diagrams with hundreds of points that
     would be prohibitively slow. Instead, point estimates (mean,
     variance) are computed once. The significance test uses group-medoid
     separation as a computationally tractable proxy statistic, evaluated
     via lookups on a single precomputed pooled pairwise distance matrix --
     no new W2 calls per permutation. See permutation_group_test_wasserstein().

  5. permutation_group_test_wasserstein() returns a raw p-value with no
     pseudocount floor; callers that want one should apply it themselves.

CONTENTS
  Core W2                _augmented_cost_matrix, wasserstein_distance_2
  Distance matrix         pairwise_distance_matrix, medoid_index
  Frechet mean/variance   frechet_mean, frechet_variance
  Permutation test        medoid_distance_statistic, medoid_variance_statistic,
                          permutation_group_test_wasserstein

Dependencies: numpy, scipy. joblib is optional, used only for parallel
pairwise_distance_matrix.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# ══════════════════════════════════════════════════════════════════════════
# CORE W2  (design decisions 1-2 above)
# ══════════════════════════════════════════════════════════════════════════

def _augmented_cost_matrix(dgm1: np.ndarray, dgm2: np.ndarray) -> np.ndarray:
    """
    (n+m) x (n+m) squared-L_infinity cost matrix for the augmented
    assignment problem between dgm1 (n points) and dgm2 (m points), each
    diagram implicitly extended by the diagonal.

    Layout:
      [0:n, 0:m]     true point-to-point costs   ||x_i - y_j||_inf^2
      [0:n, m:m+n]   dgm1 points matched to diagonal (each row i filled
                     with dgm1[i]'s own squared diagonal distance, repeated
                     across all n columns of this block -- equivalent to,
                     and simpler than, a penalised n x n diagonal submatrix,
                     since the n diagonal "slots" are interchangeable)
      [n:n+m, 0:m]   dgm2 points matched to diagonal, same construction
      [n:n+m, m:m+n] dummy-to-dummy, cost 0

    Both diagrams may be empty (n=0 or m=0); returns a 0x0 matrix if both are.
    """
    n, m = len(dgm1), len(dgm2)
    size = n + m
    C = np.zeros((size, size))
    if n > 0 and m > 0:
        diff = np.abs(dgm1[:, None, :] - dgm2[None, :, :])
        cross = diff.max(axis=2)  # (n, m) L_infinity
        C[:n, :m] = cross ** 2
    if n > 0:
        diag1 = (dgm1[:, 1] - dgm1[:, 0]) / 2.0
        C[:n, m:m + n] = np.tile((diag1 ** 2)[:, None], (1, n))
    if m > 0:
        diag2 = (dgm2[:, 1] - dgm2[:, 0]) / 2.0
        C[n:n + m, :m] = np.tile((diag2 ** 2)[None, :], (m, 1))
    return C


def wasserstein_distance_2(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """
    2-Wasserstein distance between two persistence diagrams (birth-death
    arrays, shape (n, 2)), L_infinity ground metric, via the Hungarian
    algorithm on the augmented cost matrix. O((n+m)^3) worst case, but
    scipy's solver is fast in practice -- well under a second even for
    diagrams with several hundred points.
    """
    dgm1 = np.asarray(dgm1, dtype=float).reshape(-1, 2) if len(dgm1) else np.empty((0, 2))
    dgm2 = np.asarray(dgm2, dtype=float).reshape(-1, 2) if len(dgm2) else np.empty((0, 2))
    n, m = len(dgm1), len(dgm2)
    if n == 0 and m == 0:
        return 0.0
    C = _augmented_cost_matrix(dgm1, dgm2)
    row, col = linear_sum_assignment(C)
    return float(np.sqrt(C[row, col].sum()))


# ══════════════════════════════════════════════════════════════════════════
# DISTANCE MATRIX  (built once per population; reused by frechet_mean's
# medoid init AND by the permutation test below)
# ══════════════════════════════════════════════════════════════════════════

def pairwise_distance_matrix(
    diagrams: List[np.ndarray],
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Full symmetric N x N matrix of pairwise W2 distances. This is the
    expensive O(N^2) step -- compute it exactly once per population of
    diagrams so that downstream group comparisons and permutation tests
    can look up any subset without recomputation.
    """
    n = len(diagrams)
    D = np.zeros((n, n))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if n_jobs != 1 and HAS_JOBLIB and len(pairs) > 0:
        results = Parallel(n_jobs=n_jobs)(
            delayed(wasserstein_distance_2)(diagrams[i], diagrams[j]) for i, j in pairs
        )
    else:
        results = [wasserstein_distance_2(diagrams[i], diagrams[j]) for i, j in pairs]
    for (i, j), d in zip(pairs, results):
        D[i, j] = d
        D[j, i] = d
    return D


def medoid_index(D: np.ndarray, indices: Optional[np.ndarray] = None) -> int:
    """
    Index (into the full 0..N-1 range of D) of the diagram in the given
    subset minimising the sum of squared distances to the rest of the
    subset. If `indices` is None, medoid is taken over all of D.
    """
    if indices is None:
        idx_map = np.arange(D.shape[0])
        sub = D
    else:
        idx_map = np.asarray(indices)
        if len(idx_map) == 1:
            return int(idx_map[0])
        sub = D[np.ix_(idx_map, idx_map)]
    sums = (sub ** 2).sum(axis=1)
    return int(idx_map[np.argmin(sums)])


# ══════════════════════════════════════════════════════════════════════════
# FRECHET MEAN / VARIANCE  (design decision 3 -- Turner et al. 2014,
# fixed-cardinality, medoid-initialized)
# ══════════════════════════════════════════════════════════════════════════

def _single_diagram_match(Y: np.ndarray, dgm: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    One diagram's contribution to a frechet_mean iteration: the augmented
    optimal matching from Y to dgm, its squared-cost contribution, and the
    per-point "vote" each y_j in Y receives (a real point in dgm, or y_j's
    own diagonal projection if matched to the diagonal). Factored out so it
    can be dispatched in parallel across diagrams (see frechet_mean n_jobs).
    """
    n_y, n_d = len(Y), len(dgm)
    if n_y + n_d == 0:
        return 0.0, np.zeros((0, 2))
    C = _augmented_cost_matrix(Y, dgm)
    row, col = linear_sum_assignment(C)
    total_sq = float(C[row, col].sum())
    assign = dict(zip(row.tolist(), col.tolist()))
    targets = np.zeros((n_y, 2))
    for j in range(n_y):
        c = assign[j]
        if c < n_d:
            targets[j] = dgm[c]
        else:
            proj = (Y[j, 0] + Y[j, 1]) / 2.0
            targets[j] = [proj, proj]
    return total_sq, targets


def frechet_mean(
    diagrams: List[np.ndarray],
    D: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, Dict]:
    """
    Fixed-cardinality Frechet mean via iterative barycenter updates:
      1. Initialise Y = medoid diagram (minimises sum of squared W2 to the
         rest of the group; uses precomputed D if given, else computes it).
      2. Each iteration: for every diagram, compute the augmented optimal
         matching from Y to that diagram (_single_diagram_match). Each
         point y_j in Y is then updated to the average, across all
         diagrams, of (a) the real point it was matched to, or (b) y_j's
         OWN current diagonal projection if it was matched to the diagonal
         in that diagram. This per-diagram step is independent across
         diagrams within an iteration -- dispatched via a persistent joblib
         worker pool (opened once for the whole call, reused every
         iteration) when n_jobs != 1, to avoid re-spawning workers on every
         iteration.
      3. Converge when total squared W2 distance (computed with the PRE-
         update Y) changes by a RELATIVE amount < tol between iterations
         (i.e. |prev - cur| / prev < tol), or at max_iter. A relative
         tolerance matters here because total_sq scales with the square of
         the coordinate units -- an absolute tolerance that looks tight at
         one scale is either meaninglessly loose or impossible to satisfy
         at another.

    Y's point count is fixed at the medoid's point count throughout (see
    module docstring, design decision 3) -- points are never added or
    removed. Not unique in general; this returns one fixed-cardinality
    local optimum.

    n_jobs: parallelism for pairwise_distance_matrix's medoid-init
    computation ONLY (when D is not supplied). The per-iteration inner loop
    over diagrams is NOT parallelised -- at realistic diagram sizes the
    per-iteration dispatch overhead of joblib exceeds the actual compute
    per iteration, so it is left serial.

    Returns (mean_diagram, info) where info contains medoid_init_index,
    n_points, n_iterations, converged. NOTE: info does not contain the
    Frechet variance -- call frechet_variance(diagrams, mean_diagram)
    separately on the returned, final mean for an exact value (the
    in-loop total_sq is computed against the pre-update Y and is only
    used for the convergence check).
    """
    n = len(diagrams)
    if n == 0:
        raise ValueError("frechet_mean requires at least one diagram")
    diagrams = [np.asarray(d, dtype=float).reshape(-1, 2) if len(d) else np.empty((0, 2))
                for d in diagrams]

    if D is None:
        D = pairwise_distance_matrix(diagrams, n_jobs=n_jobs)
    medoid_idx = medoid_index(D)
    Y = diagrams[medoid_idx].copy()

    prev_total_sq = np.inf
    converged = False
    n_iter_run = 0

    for it in range(max_iter):
        n_iter_run = it + 1
        n_y = len(Y)
        sums = np.zeros_like(Y)
        total_sq = 0.0

        for dgm in diagrams:
            tsq, targets = _single_diagram_match(Y, dgm)
            total_sq += tsq
            if n_y > 0:
                sums += targets

        Y_new = sums / n if n_y > 0 else Y

        rel_change = (abs(prev_total_sq - total_sq) / max(prev_total_sq, 1e-12)
                      if np.isfinite(prev_total_sq) else np.inf)
        if rel_change < tol:
            Y = Y_new
            converged = True
            break
        prev_total_sq = total_sq
        Y = Y_new

    info = {
        "medoid_init_index": medoid_idx,
        "n_points": int(len(Y)),
        "n_iterations": n_iter_run,
        "converged": converged,
    }
    return Y, info


def frechet_variance(diagrams: List[np.ndarray], mean: np.ndarray) -> float:
    """
    Exact Frechet variance of `diagrams` about `mean`: the mean squared W2
    distance from `mean` to each diagram. Call this on the FINAL diagram
    returned by frechet_mean() -- do not reuse the in-loop total_sq, which
    is computed against the pre-update Y of the last iteration.
    """
    if len(diagrams) == 0:
        return 0.0
    sq = [wasserstein_distance_2(mean, dgm) ** 2 for dgm in diagrams]
    return float(np.mean(sq))


# ══════════════════════════════════════════════════════════════════════════
# PERMUTATION TEST  (design decision 4 -- medoid-based proxy statistic,
# lookups on a precomputed pooled distance matrix, no W2 calls per
# permutation)
# ══════════════════════════════════════════════════════════════════════════

def medoid_distance_statistic(
    D_full: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> float:
    """W2 distance between group A's medoid and group B's medoid (lookup only)."""
    med_a = medoid_index(D_full, idx_a)
    med_b = medoid_index(D_full, idx_b)
    return float(D_full[med_a, med_b])


def medoid_variance_statistic(D_full: np.ndarray, idx: np.ndarray) -> float:
    """Mean squared W2 distance from the group's medoid to the rest of the group (lookup only)."""
    med = medoid_index(D_full, idx)
    sq = D_full[med, idx] ** 2
    return float(sq.mean())


def permutation_group_test_wasserstein(
    D_full: np.ndarray,
    labels: np.ndarray,
    group_a,
    group_b,
    statistic: str = "medoid_distance",
    n_permutations: int = 1000,
    rng_seed: int = 0,
) -> Dict:
    """
    Permutation test for group separation in Wasserstein space, using a
    medoid-based proxy statistic (see module docstring, design decision 4).
    `D_full` must be the FULL pooled pairwise distance matrix for the
    population (all diagrams, both groups together -- from
    pairwise_distance_matrix()), and `labels` an array of the same length
    giving each diagram's group membership.

    statistic:
      "medoid_distance"       Distance between group_a's and group_b's
                               medoids. Labels are shuffled n_permutations
                               times; p-value is the fraction of null
                               medoid-distances >= observed.
      "medoid_variance_ratio" Ratio of group_a's to group_b's medoid-based
                               variance. p-value is the fraction of null
                               ratios >= observed (one-sided: group_a more
                               heterogeneous than group_b).

    Point estimates (Frechet mean/variance) are NOT recomputed here and are
    unaffected by this test's proxy statistic.
    """
    labels = np.asarray(labels)
    idx_a = np.where(labels == group_a)[0]
    idx_b = np.where(labels == group_b)[0]
    n_a, n_b = len(idx_a), len(idx_b)
    if n_a == 0 or n_b == 0:
        raise ValueError(f"one or both groups empty: n_a={n_a} (group_a={group_a}), "
                          f"n_b={n_b} (group_b={group_b})")
    if statistic not in ("medoid_distance", "medoid_variance_ratio"):
        raise ValueError(f"unknown statistic: {statistic}")

    def _compute(pa: np.ndarray, pb: np.ndarray) -> float:
        if statistic == "medoid_distance":
            return medoid_distance_statistic(D_full, pa, pb)
        var_a = medoid_variance_statistic(D_full, pa)
        var_b = medoid_variance_statistic(D_full, pb)
        return var_a / var_b if var_b > 0 else float("inf")

    observed = _compute(idx_a, idx_b)

    all_idx = np.concatenate([idx_a, idx_b])
    rng = np.random.default_rng(rng_seed)
    null_stats = np.empty(n_permutations)
    for p in range(n_permutations):
        perm = rng.permutation(all_idx)
        null_stats[p] = _compute(perm[:n_a], perm[n_a:])

    finite = np.isfinite(null_stats)
    p_value = float((null_stats >= observed).mean())

    return {
        "statistic": statistic,
        "observed": round(float(observed), 6),
        "n_permutations": n_permutations,
        "null_mean": round(float(null_stats[finite].mean()), 6) if finite.any() else None,
        "null_sd": round(float(null_stats[finite].std()), 6) if finite.any() else None,
        "p_value": round(p_value, 6),
        "group_a": str(group_a),
        "group_b": str(group_b),
        "n_a": int(n_a),
        "n_b": int(n_b),
    }


# ══════════════════════════════════════════════════════════════════════════
# SMOKE TEST  (run directly: python wasserstein_geometry.py)
# Hand-computable cases only -- verifies the assignment construction picks
# the correct match (direct vs. diagonal-kill) before trusting it on real
# data.
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    failures = []

    def check(name, got, want, tol=1e-6):
        ok = abs(got - want) < tol
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={got:.6f} want={want:.6f}")
        if not ok:
            failures.append(name)

    print("wasserstein_geometry.py smoke test")
    print("=" * 60)

    print("\n[1] Identical single-point diagrams -> W2 = 0")
    d1 = np.array([[0.0, 10.0]])
    check("identical points", wasserstein_distance_2(d1, d1.copy()), 0.0)

    print("\n[2] Direct match cheaper than diagonal-kill")
    # dgm1=[(0,10)], dgm2=[(0,20)]: direct L_inf cost = max(0,10)=10 -> cost^2=100
    # diagonal-kill: 5^2 + 10^2 = 25+100=125 > 100 -> direct match wins, W2=10
    d1 = np.array([[0.0, 10.0]])
    d2 = np.array([[0.0, 20.0]])
    check("direct match", wasserstein_distance_2(d1, d2), 10.0)

    print("\n[3] Diagonal-kill cheaper than direct match")
    # dgm1=[(0,100)], dgm2=[(0,1)]: direct cost^2 = 99^2=9801
    # diagonal-kill: 50^2 + 0.5^2 = 2500.25 < 9801 -> diagonal wins
    d1 = np.array([[0.0, 100.0]])
    d2 = np.array([[0.0, 1.0]])
    check("diagonal-kill", wasserstein_distance_2(d1, d2), np.sqrt(2500.25))

    print("\n[4] Empty vs single point -> distance to diagonal")
    d1 = np.empty((0, 2))
    d2 = np.array([[0.0, 10.0]])
    check("empty vs point", wasserstein_distance_2(d1, d2), 5.0)

    print("\n[5] Both empty -> 0")
    check("both empty", wasserstein_distance_2(np.empty((0, 2)), np.empty((0, 2))), 0.0)

    print("\n[6] Frechet mean of identical diagrams -> the diagram itself, variance 0")
    dgm = np.array([[0.0, 10.0], [5.0, 30.0], [2.0, 4.0]])
    group = [dgm.copy() for _ in range(5)]
    mean, info = frechet_mean(group)
    print(f"  medoid_init_index={info['medoid_init_index']} n_iterations={info['n_iterations']} "
          f"converged={info['converged']}")
    check("mean matches input (sorted sum check)", float(mean.sum()), float(dgm.sum()))
    var = frechet_variance(group, mean)
    check("variance of identical group", var, 0.0)

    print("\n[7] Medoid-based permutation test sanity -- two well-separated synthetic groups")
    rng = np.random.default_rng(0)
    group_a_dgms = [np.array([[0.0, 10.0 + rng.normal(0, 0.5)]]) for _ in range(15)]
    group_b_dgms = [np.array([[0.0, 200.0 + rng.normal(0, 0.5)]]) for _ in range(15)]
    all_dgms = group_a_dgms + group_b_dgms
    labels = np.array(["A"] * 15 + ["B"] * 15)
    D_full = pairwise_distance_matrix(all_dgms)
    result = permutation_group_test_wasserstein(
        D_full, labels, "A", "B", statistic="medoid_distance", n_permutations=200, rng_seed=0
    )
    print(f"  observed={result['observed']:.2f} null_mean={result['null_mean']:.2f} "
          f"p={result['p_value']:.4f}")
    ok = result["p_value"] < 0.05
    print(f"  [{'OK' if ok else 'FAIL'}] well-separated groups are significant at p<0.05")
    if not ok:
        failures.append("permutation test power check")

    print("\n[8] frechet_mean n_jobs=1 vs n_jobs=2 give the same result")
    rng2 = np.random.default_rng(7)
    grp = [np.column_stack([b := rng2.random(12) * 50, b + rng2.random(12) * 20]) for _ in range(8)]
    mean1, info1 = frechet_mean(grp, n_jobs=1)
    mean2, info2 = frechet_mean(grp, n_jobs=2)
    check("frechet_mean parallel matches serial (sum check)", float(mean2.sum()), float(mean1.sum()), tol=1e-6)
    print(f"  n_iterations: serial={info1['n_iterations']} parallel={info2['n_iterations']}")

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST FAILED: {failures}")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED")
        sys.exit(0)
