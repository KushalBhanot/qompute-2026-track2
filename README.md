# Sustainable Portfolio Selection with QAOA

Qompute 2026 · Track 2: Optimizing for People

## Overview

A community pension fund must invest its members' savings in exactly 3 of 6 sector funds, in equal amounts. The trustees want to balance expected return, risk and sustainability (ESG). We formulate this as a constrained binary optimization problem, encode it as a cost Hamiltonian, solve it with QAOA in Qiskit, and verify every quantum result against exact brute force. We then examine how two modelling choices, the constraint penalty and the sustainability weight, determine which portfolio is called "optimal" and who bears the consequences.

## Approach

**Model.** One bit per fund, $x_i = 1$ meaning the fund is held. We minimize

$$f(x) = \frac{\lambda}{K^2}\,x^T\Sigma x - \frac{1}{K}\,\mu^T x - \frac{\eta}{K}\,s^T x \qquad \text{subject to} \qquad \sum_i x_i = K,$$

with expected returns $\mu$, covariance $\Sigma$, ESG scores $s$, risk aversion $\lambda = 0.15$ and sustainability weight $\eta$.

**Part 1: penalty formulation.** The cardinality constraint is added as a penalty $A(\sum_i x_i - K)^2$, expanded into a QUBO and mapped to an Ising Hamiltonian. QAOA uses the standard X mixer. We derive the critical penalty $A_{crit}$ from brute force, and a bound $A > U - L$ that does not require the optimum, and study how $A$ affects validity and solution quality.

**Part 2: constraint-preserving mixer.** QAOA starts from the Dicke state (an equal superposition of all valid portfolios) and uses an XY ring mixer, which preserves the number of funds held. No penalty is required. Training uses multiple random starts and layerwise INTERP initialization. We compare against Part 1 and against classical baselines, scale to 8 funds, and sweep $\eta$ to map the return–sustainability trade-off.

## Results

All results are from exact statevector simulation with fixed seeds. $P_\text{opt}$ is the probability that one shot returns the optimal portfolio.

| Method (6 funds, choose 3, depth $p = 5$) | Valid shots | $P_\text{opt}$ |
|---|---|---|
| Random valid portfolio | 1.00 | 0.05 |
| X mixer + penalty, $A = 2$ | 0.49 | 0.31 |
| X mixer + penalty, $A = 8$ | 0.99 | 0.06 |
| XY mixer, random starts (median) | 1.00 | 0.35 |
| XY mixer, layerwise initialization | 1.00 | **0.84** |

- **The penalty weight is a modelling choice.** Below $A_{crit} = 1.55$ the lowest-energy state holds 4 funds and violates the rule. Large penalties give valid samples but barely outperform random guessing. The bound that does not require the optimum ($A > 5.24$) falls in this poor regime.
- **The XY mixer enforces the rule by construction**, at the cost of a deeper circuit (317 against 150 CX gates at $p = 5$). With 8 funds (70 valid portfolios) it reaches $P_\text{opt} = 0.59$, against 0.014 for random guessing.
- **The sustainability weight changes the decision.** Six different portfolios are optimal as $\eta$ increases from 0 to 0.77, from ABC (return 8.0, ESG 2.7) to DEF (return 5.0, ESG 9.0). QAOA identifies the exact optimum at all five weights tested. Three of these cases are near-ties, where we argue the final choice should rest with the trustees rather than the algorithm.
- **Classical baseline.** A greedy search finds the optimum in 25 evaluations. At this size the project demonstrates a verified, constraint-preserving formulation rather than a quantum speedup.

All asset data are synthetic and chosen for illustration.

## Repository structure

| File | Contents |
|---|---|
| [`01_penalty_formulation.ipynb`](01_penalty_formulation.ipynb) | Scenario and stakeholders, model, penalty analysis, QUBO and Ising derivation, QAOA circuit, training, decoding, penalty and depth sweep, reflection. |
| [`02_xy_mixer_and_tradeoffs.ipynb`](02_xy_mixer_and_tradeoffs.ipynb) | Classical baselines, Dicke state and XY mixer, training protocol, comparison with Part 1, 8-fund instance, sustainability trade-off, reflection, references. |
| [`portfolio_qaoa.py`](portfolio_qaoa.py) | Shared implementation: problem data, brute force, Hamiltonian construction, QAOA circuits, training and metrics. |
| [`requirements.txt`](requirements.txt) | Pinned package versions. |

## Reproducing the results

```bash
python -m pip install -r requirements.txt jupyter
jupyter notebook 01_penalty_formulation.ipynb
jupyter notebook 02_xy_mixer_and_tradeoffs.ipynb
python portfolio_qaoa.py --part all      # text summary of both parts
```

Each notebook runs in about 1.5 minutes on a laptop. No quantum hardware or IBM account is required.

## Acknowledgements

The QAOA pipeline builds on B. Tirado, "Combinatorial Optimization with QAOA: Max-Cut", BasQ Qiskit Fall Fest 2026 (CC BY 4.0), the Track 2 starter notebook. Full references are listed at the end of Part 2.
