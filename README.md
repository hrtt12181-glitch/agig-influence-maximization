# AGIG Influence Maximization with Interest Lifecycle

**AGIG (Awareness-Gap Influence Gap)** — A deep learning framework for personalized seed selection in social influence maximization. This prototype combines knowledge graph-enhanced content encoding, multi-interest extraction, interest lifecycle modeling, and dynamic social graph propagation to find the most influential seed users for a target content item.

## Overview

Influence maximization is the problem of selecting a small set of seed users who, through social influence propagation, will maximize awareness or adoption of a target piece of content. This framework extends the classic formulation with:

- **Interest Lifecycle** — models how each user's interest in a topic evolves over time (growth, peak, decline), rather than treating interest as static.
- **KG-Enhanced Content Understanding** — uses knowledge graph triples to enrich both user history and target content representations.
- **Multi-Interest Extraction** — captures multiple latent interest dimensions from each user's historical interactions.
- **Dynamic Social Graph** — processes a temporal sequence of social graphs (weighted edges) through GraphSAGE layers + GRU to capture evolving social relationships.
- **Target-Aware Propagation** — computes diffusion probabilities conditioned on both the target content and the user's current interest.

## Architecture

The pipeline consists of seven components:

`
User History + Content  ──►  KG-Enhanced Content Encoder  ──►  Query + History Repr
                                     │
                                     ▼
                            Multi-Interest Extractor
                                     │
                                     ▼
                            Interest Lifecycle Module
                                     │
                                     ▼
Dynamic Social Graph ──►  Dynamic Graph Encoder
                                     │
                                     ▼
                            Potential Interest Scorer
                                     │
                                     ▼
                            Target-Aware Propagation
                                     │
                                     ▼
                            Greedy Seed Selection
`

### 1. KG-Enhanced Content Encoder
Encodes user history and target content text via a knowledge graph. Uses relation-aware GCN layers to propagate information across entities, producing a content query vector **q(c)** and user history representations **H_u**.

### 2. Multi-Interest Extractor
Projects the user history representation into **multiple latent interest vectors** using a self-attention mechanism, allowing each user to have distinct interest facets.

### 3. Interest Lifecycle Module
Takes multi-interest vectors plus lifecycle features (recent activity, frequency trend, recency, interaction interval, growth slope) and models each interest's current phase. Outputs refined interest vectors weighted by lifecycle stage.

### 4. Dynamic Social Graph Encoder
Processes a sequence of temporal social graphs through stacked GraphSAGE layers followed by a GRU. Each graph snapshot produces node embeddings that capture evolving structural roles and relationship strengths.

### 5. Potential Interest User Scorer
Combines the content query, lifecycle-weighted interests, KG similarity, raw activity, and dynamic graph context to compute **I_u(c, t)** — the potential interest score of user *u* for content *c* at time *t*.

### 6. Target-Aware Propagation
Extends the independent cascade model: **p_target(u, v, c, t)** captures the probability that user *v* is influenced by *u* specifically for content *c* at time *t*, factoring in both users' interest scores.

### 7. Greedy Seed Selection
Given the dense propagation matrix and interest scores, selects the top-*k* seeds using a greedy marginal gain approximation (2-hop influence estimate).

## Getting Started

### Requirements

- Python 3.10+
- PyTorch 2.x
- numpy

### Installation

`ash
# Create and activate virtual environment (optional)
python -m venv .venv
.venv\\Scripts\\activate

# Install dependencies
pip install torch numpy
`

### Run

`ash
python main.py
`

Output includes:
- Per-user potential interest scores
- Target-aware edge propagation probabilities
- Selected seed set and its approximate influence value

## Project Structure

`
├── main.py                         # Entry point: assembles pipeline and runs inference
├── data/
│   └── toy_data.py                 # Toy dataset generator (8 users, 3 time steps)
├── models/
│   ├── kg_encoder.py               # KG-enhanced content encoder
│   ├── multi_interest.py           # Multi-interest extractor (self-attention)
│   ├── lifecycle.py                # Interest lifecycle module
│   ├── dynamic_graph_encoder.py    # GraphSAGE + GRU for temporal graphs
│   ├── interest_scorer.py          # Potential interest scorer I_u(c,t)
│   └── propagation.py             # Target-aware propagation model
├── optimization/
│   └── seed_selection.py           # Greedy seed selection
├── utils/
│   └── metrics.py                  # Helper utilities
└── requirements.txt
`

## Roadmap (Possible Extensions)

- Replace toy data with real social network and interaction datasets.
- Add temporal attention to the lifecycle module.
- Incorporate explicit time embeddings into the dynamic graph encoder.
- Support for continuous-time dynamic graphs.
- Alternative seed selection algorithms (CELF, RIS-based).
