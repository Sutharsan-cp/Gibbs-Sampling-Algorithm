# LinkedIn Random Walker 🔗
## Homophily-Based Label Propagation via Gibbs Sampling

A Streamlit application demonstrating **Algorithm 2: GS (Gibbs Sampling)** for
semi-supervised label propagation on graph-structured data — inspired by LinkedIn's
professional network topology.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📁 Project Structure

```
linkedin_gibbs/
├── app.py              ← Streamlit frontend (all UI + visualisations)
├── gibbs_engine.py     ← Core Gibbs Sampling algorithm (pure Python)
├── requirements.txt    ← Dependencies
└── README.md
```

---

## 🔬 Algorithm Overview (Algorithm 2: GS)

```
BOOTSTRAP
  for each node Y_i in V:
    compute a_i using only observed neighbours
    Y_i ← sample(a_i)

BURN-IN  (n = 1 to B)
  generate random ordering O over nodes
  for each node Y_i in O:
    compute a_i using current neighbour assignments
    Y_i ← sample(a_i)

INITIALISE COUNTS
  c[i, l] = 0  for all labels l

COLLECT SAMPLES  (n = 1 to S)
  generate random ordering O over nodes
  for each node Y_i in O:
    compute a_i; Y_i ← sample(a_i)
    c[i, y_i] += 1

FINAL LABELS
  Y_i ← argmax_l  c[i, l]
```

---

## ⚙️ Configurable Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Nodes** | Graph size | 60 |
| **Sectors** | Number of label classes | 3 |
| **Homophily** | Edge bias toward same sector | 0.78 |
| **Seed nodes %** | Fraction with known labels | 15% |
| **Burn-in B** | Discarded MCMC iterations | 100 |
| **Samples S** | Aggregated MCMC iterations | 200 |
| **Alpha α** | Dirichlet smoothing prior | 1.0 |

---

## 📊 App Tabs

| Tab | Content |
|-----|---------|
| **Graph** | Interactive graph coloured by true sectors |
| **Algorithm** | Pseudocode walkthrough + key intuitions |
| **Results** | Run sampler → accuracy metrics, confidence histograms, convergence trace, per-sector accuracy, full prediction table |
| **About** | Theory, homophily, parameter guide, tech stack |

---

## 🧠 Key Concepts

**Homophily**: "Birds of a feather flock together" — professionals in the same sector
tend to connect on LinkedIn. This structural bias is the signal we exploit.

**Gibbs Sampling**: A Markov Chain Monte Carlo method that samples each variable
conditioned on all others, cycling until the chain mixes and samples approximate
the true posterior.

**Burn-in**: Early samples from an unmixed chain are discarded; only stable samples
contribute to the final label vote.

---

## Tech Stack
- Python 3.10+
- Streamlit (frontend)
- NetworkX (graphs)
- Plotly (visualisations)
- NumPy / Pandas
