"""
LinkedIn Random Walker — Gibbs Sampling Label Propagation
Streamlit App (app.py)

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import random
import math
from collections import defaultdict
from gibbs_engine import (
    GibbsSampler,
    generate_linkedin_graph,
    evaluate,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Random Walker · Gibbs Sampling",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark editorial aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg: #0d0f14;
    --card: #161922;
    --border: #252a35;
    --accent: #4fffb0;
    --accent2: #ff6b6b;
    --accent3: #6eb5ff;
    --text: #e8eaf0;
    --muted: #6b7280;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}

/* Buttons */
.stButton button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 4px !important;
    letter-spacing: 0.05em !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Sliders */
.stSlider [data-testid="stThumbValue"] { color: var(--accent) !important; }

/* Progress bar */
.stProgress > div > div { background-color: var(--accent) !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Hero header */
.hero {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.hero span { color: var(--accent); font-style: italic; }
.subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.tag {
    display: inline-block;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 2px 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 0.05em;
    margin-right: 6px;
}
.algo-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 16px 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    line-height: 1.8;
    color: #a8b4c8;
    margin: 1rem 0;
}
.algo-box b { color: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette per sector
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_COLORS = {
    "Tech":       "#4fffb0",
    "Finance":    "#ff6b6b",
    "Healthcare": "#6eb5ff",
    "Marketing":  "#ffd166",
    "Legal":      "#c77dff",
    "Education":  "#ff9f43",
}

def sector_color(lbl):
    return SECTOR_COLORS.get(lbl, "#aaaaaa")

# ─────────────────────────────────────────────────────────────────────────────
# Graph layout (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_layout(n_nodes, n_communities, homophily, seed):
    G, true_labels, label_names = generate_linkedin_graph(
        n_nodes, n_communities, homophily, seed
    )
    pos = nx.spring_layout(G, seed=seed, k=1.5 / math.sqrt(n_nodes))
    return G, true_labels, label_names, pos

# ─────────────────────────────────────────────────────────────────────────────
# Plotly graph renderer
# ─────────────────────────────────────────────────────────────────────────────
def make_graph_figure(
    G: nx.Graph,
    pos: dict,
    node_labels: dict,
    observed: set,
    confidence: dict = None,
    title: str = "",
) -> go.Figure:
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.6, color="#252a35"),
        hoverinfo="none",
    )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_colors = [sector_color(node_labels.get(n, "")) for n in G.nodes()]
    node_sizes = [14 if n in observed else 9 for n in G.nodes()]
    node_symbols = ["diamond" if n in observed else "circle" for n in G.nodes()]
    conf_vals = [confidence.get(n, 1.0) if confidence else 1.0 for n in G.nodes()]

    hover_texts = []
    for n in G.nodes():
        lbl = node_labels.get(n, "?")
        obs = "✦ Seed" if n in observed else "◦ Inferred"
        conf_str = f"{conf_vals[list(G.nodes()).index(n)]:.0%}" if confidence else "—"
        hover_texts.append(
            f"<b>Node {n}</b><br>Sector: {lbl}<br>Status: {obs}<br>Confidence: {conf_str}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            symbol=node_symbols,
            line=dict(width=1.5, color="#0d0f14"),
            opacity=[0.6 + 0.4 * c for c in conf_vals],
        ),
        hoverinfo="text",
        hovertext=hover_texts,
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(family="DM Serif Display", size=16, color="#e8eaf0")),
            showlegend=False,
            paper_bgcolor="#0d0f14",
            plot_bgcolor="#0d0f14",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hoverlabel=dict(
                bgcolor="#161922",
                font_family="Space Mono",
                font_size=11,
            ),
        )
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**Graph**")
    n_nodes = st.slider("Nodes", 20, 150, 60, 5)
    n_communities = st.slider("Sectors (labels)", 2, 6, 3)
    homophily = st.slider("Homophily", 0.0, 1.0, 0.78, 0.05,
                           help="Probability that edges connect same-sector nodes")
    seed = st.number_input("Random seed", 0, 9999, 42)

    st.markdown("---")
    st.markdown("**Gibbs Sampler**")
    pct_observed = st.slider("% Seed nodes (observed)", 5, 50, 15,
                              help="Fraction of nodes with known labels")
    burn_in = st.slider("Burn-in iterations (B)", 10, 500, 100, 10)
    num_samples = st.slider("Sample iterations (S)", 10, 500, 200, 10)
    alpha = st.slider("Dirichlet smoothing α", 0.01, 5.0, 1.0, 0.05)

    st.markdown("---")
    run_btn = st.button("▶  Run Gibbs Sampler", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">LinkedIn <span>Random Walker</span></div>
<div class="subtitle">Homophily-Based Label Propagation · Gibbs Sampling</div>
<span class="tag">Algorithm 2: GS</span>
<span class="tag">Graph ML</span>
<span class="tag">Semi-Supervised</span>
""", unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# Load graph
# ─────────────────────────────────────────────────────────────────────────────
G, true_labels, label_names, pos = get_layout(n_nodes, n_communities, homophily, int(seed))

# Choose seed (observed) nodes
rng = np.random.default_rng(int(seed) + 7)
all_nodes = list(G.nodes())
n_obs = max(1, int(len(all_nodes) * pct_observed / 100))
observed_nodes = set(rng.choice(all_nodes, size=n_obs, replace=False).tolist())
observed_dict = {n: true_labels[n] for n in observed_nodes}

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["🌐  GRAPH", "🔬  ALGORITHM", "📊  RESULTS", "📖  ABOUT"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Graph
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([3, 1])

    with col_a:
        # Initial graph: true labels shown
        init_fig = make_graph_figure(
            G, pos, true_labels, observed_nodes,
            title="Ground Truth Graph — ◆ = seed nodes"
        )
        st.plotly_chart(init_fig, use_container_width=True, key="init_graph")

    with col_b:
        st.markdown("**Graph Stats**")
        st.metric("Nodes", G.number_of_nodes())
        st.metric("Edges", G.number_of_edges())
        st.metric("Avg degree", f"{2*G.number_of_edges()/max(G.number_of_nodes(),1):.1f}")
        st.metric("Seed nodes", n_obs)
        st.metric("Unknown nodes", len(all_nodes) - n_obs)
        st.metric("Homophily", f"{homophily:.0%}")

        st.markdown("**Sector Legend**")
        for lbl in label_names:
            col = sector_color(lbl)
            st.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{col};border-radius:2px;margin-right:6px"></span>{lbl}',
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Algorithm
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Algorithm 2 — Gibbs Sampling (GS)")
    st.markdown("""
<div class="algo-box">
<b>Bootstrap phase</b><br>
for each node Y_i ∈ 𝒱 do<br>
&nbsp;&nbsp;compute a<sub>i</sub> using only observed nodes in 𝒩_i<br>
&nbsp;&nbsp;Y_i ← f̂(a<sub>i</sub>)<br>
end for<br><br>
<b>Burn-in phase</b> (n = 1 to B)<br>
for each node Y_i ∈ O do<br>
&nbsp;&nbsp;compute a<sub>i</sub> using current assignments to 𝒩_i<br>
&nbsp;&nbsp;Y_i ← f̂(a<sub>i</sub>)<br>
end for<br><br>
<b>Initialize sample counts</b><br>
for each label l ∈ ℒ do<br>
&nbsp;&nbsp;c[i, l] = 0<br>
end for<br><br>
<b>Collect samples</b> (n = 1 to S)<br>
for each node Y_i ∈ O do<br>
&nbsp;&nbsp;Y_i ← f̂(a<sub>i</sub>)<br>
&nbsp;&nbsp;c[i, y_i] ← c[i, y_i] + 1<br>
end for<br><br>
<b>Final labels</b><br>
Y_i ← argmax<sub>l∈ℒ</sub> c[i, l]
</div>
""", unsafe_allow_html=True)

    st.markdown("### Key Intuitions")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Homophily**\nNodes in the same professional sector tend to connect — Finance pros know Finance pros.")
    with c2:
        st.info("**Gibbs Sampling**\nIteratively resample each node's label conditioned on its current neighbors, building a Markov chain.")
    with c3:
        st.info("**Burn-in**\nDiscard early samples (chain hasn't mixed). Then aggregate stable samples for the final label vote.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Results (runs only after button click)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not run_btn:
        st.info("👈 Configure parameters in the sidebar, then click **Run Gibbs Sampler**.")
    else:
        # ── Run algorithm ──────────────────────────────────────────────
        progress_bar = st.progress(0, text="Initialising…")
        status_text  = st.empty()
        total_iters  = burn_in + num_samples

        def on_progress(step, total, phase=""):
            pct = int(step / total * 100)
            progress_bar.progress(pct, text=f"{phase.title()} — iteration {step}/{total}")

        sampler = GibbsSampler(
            graph=G,
            labels=label_names,
            observed=observed_dict,
            burn_in=burn_in,
            num_samples=num_samples,
            alpha=alpha,
        )
        predicted = sampler.run(progress_callback=on_progress)
        confidence = sampler.get_confidence()

        progress_bar.progress(100, text="Done ✓")
        status_text.empty()

        # ── Metrics ─────────────────────────────────────────────────────
        eval_results = evaluate(true_labels, predicted, observed_nodes)
        acc = eval_results["accuracy"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy (unlabeled)", f"{acc:.1%}")
        m2.metric("Correct predictions", eval_results["correct"])
        m3.metric("Total unlabeled", eval_results["total_unlabeled"])
        m4.metric("Seed nodes used", n_obs)

        # ── Side-by-side graph comparison ───────────────────────────────
        st.markdown("### Label Propagation Result")
        g1, g2 = st.columns(2)

        with g1:
            # Before: hide non-seed labels
            masked = {n: (true_labels[n] if n in observed_nodes else "?") for n in G.nodes()}
            masked_colors = {n: (sector_color(true_labels[n]) if n in observed_nodes else "#333344")
                             for n in G.nodes()}
            fig_before = make_graph_figure(
                G, pos, masked, observed_nodes,
                title="Before — only seed nodes labelled"
            )
            st.plotly_chart(fig_before, use_container_width=True, key="before")

        with g2:
            fig_after = make_graph_figure(
                G, pos, predicted, observed_nodes,
                confidence=confidence,
                title="After Gibbs Sampling — all nodes labelled"
            )
            st.plotly_chart(fig_after, use_container_width=True, key="after")

        # ── Confidence distribution ──────────────────────────────────────
        st.markdown("### Confidence Distribution")
        conf_data = [
            {"node": n, "confidence": confidence[n], "label": predicted[n], "seed": n in observed_nodes}
            for n in G.nodes()
        ]
        df_conf = pd.DataFrame(conf_data)
        unlabeled_df = df_conf[~df_conf["seed"]]

        fig_hist = px.histogram(
            unlabeled_df,
            x="confidence",
            color="label",
            color_discrete_map=SECTOR_COLORS,
            nbins=20,
            title="Prediction Confidence for Inferred Nodes",
            labels={"confidence": "Confidence (argmax count / S)", "label": "Sector"},
        )
        fig_hist.update_layout(
            paper_bgcolor="#0d0f14",
            plot_bgcolor="#161922",
            font=dict(family="Space Mono", color="#e8eaf0"),
            bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Per-sector accuracy ──────────────────────────────────────────
        st.markdown("### Per-Sector Accuracy")
        sector_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for n, pred in predicted.items():
            if n not in observed_nodes:
                true = true_labels[n]
                sector_stats[true]["total"] += 1
                if pred == true:
                    sector_stats[true]["correct"] += 1

        sector_df = pd.DataFrame([
            {"sector": s, "accuracy": v["correct"] / max(v["total"], 1), "n": v["total"]}
            for s, v in sector_stats.items()
        ]).sort_values("accuracy", ascending=True)

        fig_bar = px.bar(
            sector_df,
            x="accuracy",
            y="sector",
            orientation="h",
            color="sector",
            color_discrete_map=SECTOR_COLORS,
            text=sector_df["accuracy"].map(lambda x: f"{x:.0%}"),
            title="Accuracy by Sector",
        )
        fig_bar.update_layout(
            paper_bgcolor="#0d0f14",
            plot_bgcolor="#161922",
            font=dict(family="Space Mono", color="#e8eaf0"),
            showlegend=False,
            xaxis=dict(tickformat=".0%", range=[0, 1.05]),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Sample convergence trace ─────────────────────────────────────
        st.markdown("### Label Stability Over Iterations")
        # Track how many nodes change label each iteration
        changes = []
        for i in range(1, len(sampler.history)):
            prev = sampler.history[i - 1]
            curr = sampler.history[i]
            n_changes = sum(1 for n in G.nodes() if n not in observed_nodes
                            and prev.get(n) != curr.get(n))
            changes.append({"iteration": i, "label_changes": n_changes,
                             "phase": "burn-in" if i <= burn_in else "sampling"})

        df_conv = pd.DataFrame(changes)
        fig_conv = px.line(
            df_conv, x="iteration", y="label_changes", color="phase",
            color_discrete_map={"burn-in": "#ff6b6b", "sampling": "#4fffb0"},
            title="Label Changes per Iteration (lower = more stable)",
        )
        fig_conv.update_layout(
            paper_bgcolor="#0d0f14",
            plot_bgcolor="#161922",
            font=dict(family="Space Mono", color="#e8eaf0"),
        )
        fig_conv.add_vline(x=burn_in, line_dash="dash", line_color="#ffd166",
                           annotation_text="burn-in end", annotation_font_color="#ffd166")
        st.plotly_chart(fig_conv, use_container_width=True)

        # ── Node table ───────────────────────────────────────────────────
        with st.expander("📋 Full Node Prediction Table"):
            rows = []
            for n in sorted(G.nodes()):
                rows.append({
                    "Node": n,
                    "True Label": true_labels[n],
                    "Predicted": predicted[n],
                    "Correct": "✓" if predicted[n] == true_labels[n] else "✗",
                    "Confidence": f"{confidence[n]:.1%}",
                    "Status": "Seed" if n in observed_nodes else "Inferred",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – About
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### About This App")
    st.markdown("""
This application demonstrates **Gibbs Sampling for semi-supervised label propagation** 
on graph-structured data, inspired by LinkedIn's professional network topology.

#### Problem Statement
Given a graph where some nodes have known labels (professional sectors) and most 
do not, infer the labels of unlabeled nodes using the graph structure.

#### Why Gibbs Sampling?
The joint posterior P(Y_unobserved | Y_observed, Graph) is intractable to compute 
directly. Gibbs Sampling constructs a Markov Chain that converges to this posterior, 
allowing us to approximate it via empirical counts.

#### Homophily Assumption
On professional networks like LinkedIn, people in the same sector tend to connect — 
a Software Engineer is more likely to connect with other engineers than with lawyers. 
This structural bias is the key signal we exploit.

#### Parameters
| Parameter | Role |
|-----------|------|
| **B** (burn-in) | Discarded iterations for Markov chain to mix |
| **S** (samples) | Iterations used to accumulate label counts |
| **α** (smoothing) | Dirichlet prior — prevents zero probabilities |
| **Homophily** | Controls how clustered the synthetic graph is |

#### Tech Stack
- **Algorithm**: Custom Python implementation of GS (Algorithm 2)  
- **Graphs**: NetworkX  
- **Frontend**: Streamlit  
- **Visualisation**: Plotly  
""")

    st.markdown("---")
    st.markdown(
        "<div style='font-family:Space Mono;font-size:0.65rem;color:#4a5568'>"
        "LinkedIn Random Walker · Gibbs Sampling Label Propagation · Built with Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )
