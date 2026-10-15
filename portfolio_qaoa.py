"""Sustainable portfolio selection with QAOA (Qompute 2026, Track 2: Optimizing for People).

Shared code for the two project notebooks. Run it directly for a text
summary of both solutions:

    python portfolio_qaoa.py

Problem. Hold exactly k of n assets, each with weight 1/k. With x_i = 1 meaning "hold asset i",
minimise

    f(x) = (lam / k^2) x^T Sigma x  -  (1/k) mu^T x  -  (eta/k) s^T x      subject to  sum_i x_i = k,

where mu is the expected return, Sigma the covariance (risk), s a sustainability (ESG) score,
lam the risk aversion and eta the weight on sustainability.

Bit order follows Qiskit: in basis state |b>, qubit i holds bit (b >> i) & 1, and in a measured
key such as '000111' qubit 0 is the rightmost character.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import SparseEfficiencyWarning
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector

# Qiskit's StatePreparation synthesis (used for the Dicke state) emits harmless SciPy performance warnings.
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


# --------------------------------------------------------------------------- problem data

@dataclass(frozen=True, eq=False)
class Portfolio:
    names: tuple[str, ...]
    mu: np.ndarray      # expected return of each asset
    sigma: np.ndarray   # covariance matrix (risk)
    esg: np.ndarray     # sustainability score of each asset
    k: int              # number of assets to hold
    lam: float = 0.15   # risk aversion
    eta: float = 0.0    # weight on sustainability

    @property
    def n(self) -> int:
        return len(self.names)

    def with_(self, **changes) -> Portfolio:
        return replace(self, **changes)


def example_portfolio(lam: float = 0.15, eta: float = 0.0, n: int = 6) -> Portfolio:
    """Illustrative assets (invented numbers): six funds A-F, or eight with G and H; hold k = n/2.

    Among A-F, the higher the return, the lower the ESG score. G and H mix both risk factors.
    """
    if n not in (6, 8):
        raise ValueError("the example has 6 or 8 assets")
    exposures = np.array([[3, 0], [2.5, 0.5], [2, 1], [1, 2], [0.5, 2.5], [0, 3], [1.75, 1.5], [0.75, 2.75]])[:n]
    return Portfolio(
        names=tuple("ABCDEFGH"[:n]),
        mu=np.array([9.0, 8, 7, 6, 5, 4, 7.5, 5.5])[:n],
        sigma=exposures @ exposures.T + np.eye(n),
        esg=np.array([1.0, 2, 5, 8, 9, 10, 4, 8.5])[:n],
        k=n // 2, lam=lam, eta=eta,
    )


# --------------------------------------------------------------------------- classical side

def all_bitstrings(n: int) -> np.ndarray:
    """Row b is basis state |b>; column i is the bit of qubit i."""
    return (np.arange(2**n)[:, None] >> np.arange(n)) & 1


def objective(pf: Portfolio, x) -> np.ndarray:
    """f(x) for one bitstring (shape n) or many (shape m x n)."""
    x = np.asarray(x, dtype=float)
    risk = np.einsum("...i,ij,...j->...", x, pf.sigma, x) / pf.k**2
    return pf.lam * risk - x @ pf.mu / pf.k - pf.eta * (x @ pf.esg) / pf.k


def assets(pf: Portfolio, x) -> str:
    return "".join(name for name, bit in zip(pf.names, x) if bit) or "(none)"


def summarize(pf: Portfolio, x) -> dict:
    """Return, risk (variance) and sustainability of the equal-weight portfolio x."""
    w = np.asarray(x, dtype=float) / pf.k
    return {"assets": assets(pf, x), "return": float(w @ pf.mu), "risk": float(w @ pf.sigma @ w),
            "ESG": float(w @ pf.esg), "f": float(objective(pf, x))}


@dataclass(eq=False)
class Exact:
    """Brute-force reference: the objective of every one of the 2^n bitstrings."""
    pf: Portfolio
    X: np.ndarray
    f: np.ndarray
    feasible: np.ndarray
    optimal: np.ndarray
    f_star: float
    f_worst: float

    def ranking(self, top: int | None = None) -> list[dict]:
        idx = np.flatnonzero(self.feasible)
        idx = idx[np.argsort(self.f[idx], kind="stable")][:top]
        return [summarize(self.pf, self.X[i]) for i in idx]

    @property
    def best(self) -> str:
        return assets(self.pf, self.X[np.flatnonzero(self.optimal)[0]])


def brute_force(pf: Portfolio) -> Exact:
    X = all_bitstrings(pf.n)
    f = objective(pf, X)
    feasible = X.sum(axis=1) == pf.k
    f_star, f_worst = f[feasible].min(), f[feasible].max()
    optimal = feasible & np.isclose(f, f_star, rtol=0, atol=1e-9)
    return Exact(pf, X, f, feasible, optimal, float(f_star), float(f_worst))


def critical_penalty(ex: Exact) -> float:
    """Smallest A for which every minimiser of f + A (sum x - k)^2 is feasible (ties at equality)."""
    v = (ex.X.sum(axis=1) - ex.pf.k) ** 2
    bad = ~ex.feasible
    return float(max(0.0, np.max((ex.f_star - ex.f[bad]) / v[bad])))


def safe_penalty(pf: Portfolio) -> dict:
    """A penalty that is provably large enough without knowing the optimum: any A > U - L works.

    U = f of a valid portfolio found by greedy_swap (so f* <= U). L = c + sum of the negative QUBO
    coefficients of f, a lower bound on f over all bitstrings. An invalid x has (sum x - k)^2 >= 1, so
    F_A(x) >= L + A > U >= f*.
    """
    a, b, c = qubo(pf, 0.0)
    lower = float(c + np.minimum(a, 0).sum() + np.minimum(b, 0).sum())
    upper = greedy_swap(pf)[1]
    return {"U": upper, "L": lower, "A_safe": upper - lower}


def greedy_swap(pf: Portfolio):
    """Classical heuristic: add the best asset k times, then do 1-for-1 swaps while f improves.

    Returns (x, f(x), number of objective evaluations).
    """
    calls = 0

    def f(x):
        nonlocal calls
        calls += 1
        return float(objective(pf, x))

    x = np.zeros(pf.n, dtype=int)
    for _ in range(pf.k):
        x = min((np.where(np.arange(pf.n) == j, 1, x) for j in np.flatnonzero(x == 0)), key=f)
    best, improved = f(x), True
    while improved:
        improved = False
        for i in np.flatnonzero(x == 1):
            for j in np.flatnonzero(x == 0):
                y = x.copy()
                y[i], y[j] = 0, 1
                if (fy := f(y)) < best - 1e-12:
                    x, best, improved = y, fy, True
                    break
            if improved:
                break
    return x, best, calls


# --------------------------------------------------------------------------- encoding

def qubo(pf: Portfolio, penalty: float):
    """Coefficients (a, b, c) of F(x) = c + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j = f(x) + A (sum x - k)^2."""
    k, A = pf.k, penalty
    a = pf.lam * np.diag(pf.sigma) / k**2 - pf.mu / k - pf.eta * pf.esg / k + A * (1 - 2 * k)
    b = np.triu(2 * pf.lam * pf.sigma / k**2 + 2 * A, 1)
    return a, b, A * k**2


def qubo_value(coeffs, X) -> np.ndarray:
    a, b, c = coeffs
    X = np.asarray(X, dtype=float)
    return c + X @ a + np.einsum("...i,ij,...j->...", X, b, X)


def to_ising(coeffs):
    """Substitute x_i = (1 - Z_i)/2. Returns (H, offset) with F(x) = <x|H|x> + offset."""
    a, b, c = coeffs
    n = len(a)
    h = -a / 2 - (b + b.T).sum(axis=1) / 4
    terms = [("Z", [i], h[i]) for i in range(n)]
    terms += [("ZZ", [i, j], b[i, j] / 4) for i in range(n) for j in range(i + 1, n) if b[i, j] != 0]
    offset = c + a.sum() / 2 + b.sum() / 4
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n), float(offset)


def cost_hamiltonian(pf: Portfolio, penalty: float = 0.0):
    """(H, offset) for f + A (sum x - k)^2. Use penalty=0 with the XY mixer, which never leaves sum x = k."""
    return to_ising(qubo(pf, penalty))


def diagonal(H: SparsePauliOp) -> np.ndarray:
    """<b|H|b> for every basis state b. H holds only I and Z terms, so this is its whole spectrum."""
    return np.real(H.to_matrix(sparse=True).diagonal())


# --------------------------------------------------------------------------- QAOA circuits

def dicke_state(n: int, k: int) -> QuantumCircuit:
    """Equal superposition of the C(n, k) bitstrings with exactly k ones.

    StatePreparation is exact but needs O(2^n) gates; Bärtschi and Eidenbenz (2019) give an O(nk)
    circuit for larger n.
    """
    amplitudes = (all_bitstrings(n).sum(axis=1) == k).astype(float)
    qc = QuantumCircuit(n, name=f"Dicke({n},{k})")
    qc.prepare_state(amplitudes / np.linalg.norm(amplitudes), range(n))
    return qc


def ring_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


class QAOA:
    """Depth-p QAOA for a diagonal cost Hamiltonian, simulated exactly with Qiskit's Statevector.

    mixer="x":  start in |+>^n; mixer exp(-i beta sum_i X_i). Explores all 2^n bitstrings.
    mixer="xy": start in the Dicke state; mixer = product over ring pairs (i, j) of
                exp(-i beta (X_i X_j + Y_i Y_j)/2) = RXX(beta) RYY(beta). Keeps exactly k assets.

    H is divided by its largest coefficient before the circuit is built, so gamma is measured in
    units where the strongest term is 1. This does not change which bitstring is optimal.
    Angles are passed as theta = [gamma_1 .. gamma_p, beta_1 .. beta_p].
    """

    def __init__(self, H: SparsePauliOp, p: int, mixer: str = "x", k: int | None = None):
        if mixer not in ("x", "xy"):
            raise ValueError("mixer must be 'x' or 'xy'")
        if np.any(H.paulis.x):
            raise ValueError("the cost Hamiltonian must contain only I and Z terms")
        self.p, self.mixer, self.n, self.k = p, mixer, H.num_qubits, k
        self.scale = float(np.max(np.abs(H.coeffs)))
        self.H = H / self.scale
        self.energies = diagonal(self.H)
        self.gamma = ParameterVector("γ", p)
        self.beta = ParameterVector("β", p)
        self.initial = QuantumCircuit(self.n)
        if mixer == "x":
            self.initial.h(range(self.n))
        else:
            self.initial.compose(dicke_state(self.n, k), inplace=True)
        self.layers = self._build_layers()
        self._psi0 = Statevector(self.initial)

    def _build_layers(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n)
        for gamma, beta in zip(self.gamma, self.beta):
            for pauli, coeff in zip(self.H.paulis, self.H.coeffs.real):
                q = [int(i) for i in np.flatnonzero(pauli.z)]
                if len(q) == 1:
                    qc.rz(2 * float(coeff) * gamma, q[0])
                elif len(q) == 2:
                    qc.rzz(2 * float(coeff) * gamma, q[0], q[1])
            if self.mixer == "x":
                qc.rx(2 * beta, range(self.n))
            else:
                for i, j in ring_pairs(self.n):
                    qc.rxx(beta, i, j)
                    qc.ryy(beta, i, j)
        return qc

    def _bind(self, theta) -> dict:
        theta = np.asarray(theta, dtype=float)
        return {self.gamma: theta[: self.p], self.beta: theta[self.p:]}

    def circuit(self, theta=None, measure: bool = False) -> QuantumCircuit:
        qc = self.initial.compose(self.layers)
        if theta is not None:
            qc = qc.assign_parameters(self._bind(theta))
        if measure:
            qc.measure_all()
        return qc

    def probabilities(self, theta) -> np.ndarray:
        return self._psi0.evolve(self.layers.assign_parameters(self._bind(theta))).probabilities()

    def expectation(self, theta) -> float:
        """<H> in scaled units (the quantity the optimizer minimises)."""
        return float(self.probabilities(theta) @ self.energies)

    def sample(self, theta, shots: int = 4096, seed: int | None = None) -> np.ndarray:
        """Measure `shots` times; returns the count of every basis index b (key '...' -> int(key, 2))."""
        job = StatevectorSampler(seed=seed).run([self.circuit(theta, measure=True)], shots=shots)
        counts = np.zeros(2**self.n)
        for key, c in job.result()[0].data.meas.get_counts().items():
            counts[int(key, 2)] += c
        return counts


def overlap_with_qaoa_ansatz(q: QAOA, theta) -> float:
    """|<ours|Qiskit's QAOAAnsatz>| for the X mixer. QAOAAnsatz orders its parameters [beta..., gamma...]."""
    ansatz = QAOAAnsatz(q.H, reps=q.p)
    theta = np.asarray(theta, dtype=float)
    reference = Statevector(ansatz.assign_parameters(np.concatenate([theta[q.p:], theta[: q.p]])))
    ours = q._psi0.evolve(q.layers.assign_parameters(q._bind(theta)))
    return float(abs(ours.inner(reference)))


def gate_counts(q: QAOA, seed: int = 0) -> dict:
    """Two-qubit gate count and depth after compiling to CX + single-qubit gates (all-to-all wiring)."""
    theta = np.random.default_rng(seed).uniform(0.1, 1.0, 2 * q.p)
    compiled = transpile(q.circuit(theta), basis_gates=["cx", "rz", "sx", "x"], optimization_level=1,
                         seed_transpiler=seed)
    return {"CX gates": int(compiled.count_ops().get("cx", 0)), "depth": int(compiled.depth())}


# --------------------------------------------------------------------------- training

def ramp_start(p: int, dt: float = 0.6, beta_sign: float = -1.0) -> np.ndarray:
    """Linear 'slow morph' schedule: gamma grows from 0 to dt while |beta| shrinks from dt to 0."""
    s = (np.arange(p) + 0.5) / p
    return np.concatenate([dt * s, beta_sign * dt * (1 - s)])


def interp_start(theta, p: int) -> np.ndarray:
    """INTERP (Zhou et al. 2020): stretch optimised depth-p angles into a depth-(p+1) starting point.

    New angle i (i = 1..p+1) = (i-1)/p * old_{i-1} + (p-i+1)/p * old_i, with old_0 = old_{p+1} = 0.
    """
    theta = np.asarray(theta, dtype=float)

    def stretch(v):
        v = np.concatenate([[0.0], v, [0.0]])
        i = np.arange(1, p + 2)
        return (i - 1) / p * v[i - 1] + (p - i + 1) / p * v[i]

    return np.concatenate([stretch(theta[:p]), stretch(theta[p:])])


def random_start(p: int, rng: np.random.Generator, gamma_max: float = np.pi,
                 beta_max: float = np.pi / 2) -> np.ndarray:
    return np.concatenate([rng.uniform(0, gamma_max, p), rng.uniform(-beta_max, beta_max, p)])


def metrics(probs, ex: Exact) -> dict:
    """P_feas: chance a shot is a valid portfolio. P_opt: chance it is the best portfolio.
    quality: (f_worst - E[f | valid]) / (f_worst - f*); 1 = always the best, 0 = always the worst valid one.
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    p_feas = float(probs[ex.feasible].sum())
    mean_f = float(probs[ex.feasible] @ ex.f[ex.feasible] / p_feas) if p_feas > 0 else np.nan
    return {"P_feas": p_feas, "P_opt": float(probs[ex.optimal].sum()),
            "quality": (ex.f_worst - mean_f) / (ex.f_worst - ex.f_star)}


@dataclass(eq=False)
class Run:
    mixer: str
    p: int
    penalty: float
    start: str
    theta: np.ndarray
    energy: float       # optimised <H> in the original units (without the constant offset)
    nfev: int
    probs: np.ndarray
    P_feas: float
    P_opt: float
    quality: float

    def row(self) -> dict:
        return {"mixer": self.mixer, "p": self.p, "A": self.penalty, "start": self.start,
                "nfev": self.nfev, "P_feas": self.P_feas, "P_opt": self.P_opt, "quality": self.quality}


def train(q: QAOA, ex: Exact, theta0, start: str = "", penalty: float = 0.0,
          maxiter: int = 400, rhobeg: float = 0.3) -> Run:
    res = minimize(q.expectation, np.asarray(theta0, dtype=float), method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": rhobeg})
    probs = q.probabilities(res.x)
    return Run(q.mixer, q.p, penalty, start, res.x, float(res.fun) * q.scale, int(res.nfev), probs,
               **metrics(probs, ex))


def depth_scan(ex: Exact, H: SparsePauliOp, depths, mixer: str = "x", penalty: float = 0.0,
               n_random: int = 5, seed: int = 0, maxiter: int = 400) -> list[Run]:
    """At every depth: n_random random starts plus one layerwise start.

    The layerwise start is a linear ramp at the first depth. After that it is INTERP of the best
    angles (lowest energy) found at the previous depth, so good angles are grown instead of re-guessed.
    """
    rng = np.random.default_rng(seed)
    beta_max = np.pi / 2 if mixer == "x" else np.pi
    runs, best_previous = [], None
    for p in depths:
        q = QAOA(H, p, mixer, k=ex.pf.k)
        here = [train(q, ex, random_start(p, rng, beta_max=beta_max), f"random {r + 1}", penalty, maxiter)
                for r in range(n_random)]
        if best_previous is None or best_previous.p != p - 1:
            theta0, label = ramp_start(p), "ramp"
        else:
            theta0, label = interp_start(best_previous.theta, best_previous.p), "layerwise"
        here.append(train(q, ex, theta0, label, penalty, maxiter, rhobeg=0.1))
        best_previous = min(here, key=lambda r: r.energy)
        runs += here
    return runs


def best_by_energy(runs: list[Run]) -> dict[int, Run]:
    """The run you would actually use at each depth: lowest energy (P_opt needs the answer, energy does not)."""
    return {p: min((r for r in runs if r.p == p), key=lambda r: r.energy) for p in sorted({r.p for r in runs})}


def summary_by_depth(runs: list[Run], metric: str = "P_opt") -> list[dict]:
    """Median, quartiles and best of the random starts, next to the layerwise run, at every depth."""
    rows = []
    for p in sorted({r.p for r in runs}):
        rand = np.array([getattr(r, metric) for r in runs if r.p == p and r.start.startswith("random")])
        chain = [getattr(r, metric) for r in runs if r.p == p and not r.start.startswith("random")]
        q1, med, q3 = np.percentile(rand, [25, 50, 75])
        rows.append({"p": p, f"random median {metric}": med, "IQR": f"{q1:.3f}-{q3:.3f}",
                     "random best": rand.max(), "layerwise / ramp": chain[0] if chain else np.nan})
    return rows


# --------------------------------------------------------------------------- reporting

def table(rows: list[dict], columns: list[str] | None = None, digits: int = 3) -> str:
    """Plain-text table (works in notebooks and terminals without pandas)."""
    if not rows:
        return "(no rows)"
    columns = columns or list(rows[0])

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    cells = [[fmt(r.get(c, "")) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    line = lambda vals: "  ".join(v.rjust(w) for v, w in zip(vals, widths))
    return "\n".join([line(columns), line(["-" * w for w in widths])] + [line(row) for row in cells])


def decode_counts(counts: np.ndarray, ex: Exact, top: int = 8) -> list[dict]:
    """Most frequent measured bitstrings with their meaning."""
    n = ex.pf.n
    rows = []
    for b in np.argsort(counts)[::-1][:top]:
        if counts[b] == 0:
            break
        rows.append({"Qiskit key": format(int(b), f"0{n}b"), "assets": assets(ex.pf, ex.X[b]),
                     "count": int(counts[b]), "valid": bool(ex.feasible[b]), "f": float(ex.f[b]),
                     "optimal": bool(ex.optimal[b])})
    return rows


def report_part(ex: Exact, H: SparsePauliOp, mixer: str, penalty: float, depths) -> None:
    runs = depth_scan(ex, H, depths, mixer, penalty=penalty)
    print(table(summary_by_depth(runs)))
    best = best_by_energy(runs)[max(depths)]
    counts = QAOA(H, best.p, mixer, k=ex.pf.k).sample(best.theta, shots=4096, seed=7)
    print(f"\n4096 shots at p = {best.p} (run with the lowest energy, start = {best.start}):")
    print(table(decode_counts(counts, ex, top=5)))
    m = metrics(counts, ex)
    print(f"P_feas = {m['P_feas']:.3f}, P_opt = {m['P_opt']:.3f}, quality = {m['quality']:.3f}")


def main(part: str = "all") -> None:
    pf = example_portfolio()
    ex = brute_force(pf)
    depths = [1, 2, 3, 4, 5]
    print("Exact reference (brute force over all 64 bitstrings; 20 hold exactly 3 assets):")
    print(table(ex.ranking(top=5)))

    if part in ("1", "all"):
        A = 2.0
        print(f"\nPART 1: penalty encoding (A = {A}; smallest safe A = {critical_penalty(ex):.3f}) + X mixer")
        H, offset = cost_hamiltonian(pf, A)
        assert np.allclose(diagonal(H) + offset, ex.f + A * (ex.X.sum(axis=1) - pf.k) ** 2)
        report_part(ex, H, "x", A, depths)

    if part in ("2", "all"):
        print("\nPART 2: Dicke start + XY mixer (no penalty needed)")
        report_part(ex, cost_hamiltonian(pf, 0.0)[0], "xy", 0.0, depths)
        x, fx, calls = greedy_swap(pf)
        print(f"\nClassical greedy + swap: {assets(pf, x)} with f = {fx:.3f} after {calls} evaluations")
        print("\nHow the sustainability weight eta changes the exact optimum:")
        print(table([{"eta": eta, **brute_force(pf.with_(eta=eta)).ranking(top=1)[0]}
                     for eta in [0, 0.25, 0.5, 0.75, 1.0]]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=["1", "2", "all"], default="all")
    main(parser.parse_args().part)
