# GaN-DefectML

### Physics-Informed Machine Learning and Graph Neural Network Framework for Defect Analysis in Gallium Nitride

GaN-DefectML is a computational materials science framework for generating, characterizing, representing, and eventually predicting the properties of point defects and dopants in **wurtzite gallium nitride (GaN)**.

The project combines crystal-structure analysis, automated defect generation, physics-informed feature engineering, classical machine learning, edge-aware graph neural networks (GNNs), and explainable AI (XAI) within a single modular pipeline.

The current repository establishes and validates the complete computational framework. Supervised defect-property prediction is intentionally gated until a sufficiently large set of validated target labels is available.

---

## Overview

Point defects and impurities strongly influence the electronic, optical, and transport properties of semiconductor materials. In GaN, vacancies, antisites, interstitials, and substitutional dopants can alter carrier concentrations, defect levels, band structure, and material stability.

Conventional first-principles approaches such as density functional theory (DFT) provide accurate defect calculations but can become computationally expensive when exploring large defect spaces.

GaN-DefectML is designed as a physics-informed machine-learning framework capable of representing these defect configurations using both:

- engineered physical descriptors, and
- periodic crystal graphs.

The framework currently supports:

- GaN structure retrieval and analysis
- wurtzite host selection
- supercell construction
- intrinsic point-defect generation
- substitutional dopant generation
- interstitial-site generation
- defect-library validation
- structural and chemical descriptor extraction
- compact physics-informed feature construction
- classical ML pipelines
- periodic crystal graph construction
- edge-aware graph neural networks
- model-readiness validation
- diagnostic explainability
- supervised explainability infrastructure

---

## Project Workflow

```text
Materials Project
       │
       ▼
Ga-N Structure Retrieval
       │
       ▼
Host Screening and Structural Analysis
       │
       ▼
Primary Wurtzite GaN Host
       │
       ▼
3 × 3 × 2 Supercell
       │
       ▼
Defect and Dopant Generation
       │
       ▼
Validated Structure Library
       │
       ├───────────────────────────────┐
       ▼                               ▼
Physics-Informed Features       Periodic Crystal Graphs
       │                               │
       ▼                               ▼
Compact Descriptor Matrix       Node / Edge / Global Features
       │                               │
       ▼                               ▼
Classical ML                   Edge-Aware GNN
       │                               │
       └───────────────┬───────────────┘
                       ▼
                   Evaluation
                       │
                       ▼
                Explainable AI
```

---

## Primary GaN Host

The primary defect host is the wurtzite GaN structure:

| Property | Value |
|---|---|
| Materials Project ID | `mp-804` |
| Formula | GaN |
| Crystal system | Hexagonal |
| Space group | P6₃mc |
| Space-group number | 186 |
| Primitive-cell sites | 4 |
| a | 3.1889 Å |
| c | 5.1924 Å |
| Density | 6.0810 g/cm³ |

A `3 × 3 × 2` replication of the host structure is used for defect generation.

### Defect Supercell

| Property | Value |
|---|---:|
| Total atoms | 72 |
| Ga atoms | 36 |
| N atoms | 36 |
| Supercell volume | 823.1098 Å³ |
| Density | 6.0810 g/cm³ |
| Single-defect atomic fraction | 1.389% |

The supercell provides a common structural basis for generating and comparing point-defect configurations.

---

## Defect Structure Library

The framework constructs a validated library containing **15 configurations**.

### Pristine

- `Pristine`

### Intrinsic Defects

- `V_Ga` — gallium vacancy
- `V_N` — nitrogen vacancy
- `Ga_N` — Ga antisite on an N site
- `N_Ga` — N antisite on a Ga site

### Substitutional Dopants

- `Mg_Ga`
- `Si_Ga`
- `O_N`
- `C_N`

### Interstitial Defects

Geometrically generated interstitial configurations include Ga and N interstitials at candidate interstitial sites.

The final master structure library is automatically checked for:

- expected configuration count
- duplicate configuration IDs
- missing configurations
- unexpected configurations
- structural consistency

The current library passes all validation checks.

---

## Interstitial-Site Search

Interstitial positions are generated geometrically rather than manually placing atoms at arbitrary coordinates.

Candidate positions are evaluated according to their periodic distances from existing host atoms.

One representative primary geometric site identified during development was:

```text
Candidate ID          : I_site_1
Fractional coordinates: [0.0, 0.0, 0.1]
Minimum host distance : 2.0553 Å
```

The framework can then insert Ga or N atoms at selected candidate positions to construct interstitial structures.

---

## Physics-Informed Feature Engineering

Each defect structure is transformed into numerical descriptors describing its composition, geometry, local environment, defect chemistry, and structural perturbation.

Feature families include:

### Structural Descriptors

- lattice parameters
- cell volume
- density
- volume per atom
- number of atomic sites
- local coordination information

### Local Atomic Environment

- periodic neighbor distances
- nearest-neighbor statistics
- coordination-shell statistics
- Ga/N neighbor populations
- distance spread and local distortion descriptors

### Chemical Descriptors

Elemental information includes quantities such as:

- atomic number
- atomic mass
- atomic radius
- electronegativity
- periodic-table group
- periodic-table row
- first ionization energy
- valence-electron information

### Defect-Chemistry Descriptors

The framework explicitly encodes the chemical difference introduced by a defect or dopant, including normalized and absolute changes in elemental properties.

This is important because configurations such as:

```text
Mg_Ga
Si_Ga
O_N
C_N
```

may have similar structural topology while representing very different chemical perturbations.

---

## Feature Reduction

The initial engineered representation is intentionally broad and subsequently reduced to avoid redundant information.

During development:

```text
Enhanced numerical features : 146

Requested physics features  : 67
Available                   : 67

Constant features removed   : 2
Duplicate features removed  : 3
Remaining features          : 62

Correlated features removed : 34

Final compact features      : 28
```

The resulting compact representation therefore contains:

```text
15 configurations × 28 physics-informed features
```

with:

```text
Remaining NaN values: 0
```

This 28-dimensional representation is also supplied to the GNN as a graph-level descriptor vector.

---

## Classical Machine Learning

The repository includes regression and classification pipelines for tabular defect descriptors.

Available baseline models include:

### Regression

- Ridge Regression
- Random Forest
- Extra Trees
- Histogram Gradient Boosting

### Classification

- Logistic Regression
- Random Forest
- Extra Trees
- Histogram Gradient Boosting

Potential supervised targets include:

### Regression Targets

- defect formation energy
- relaxed band gap
- band-gap change
- transition level
- carrier concentration

### Classification Targets

- donor/acceptor behavior
- carrier type

---

## Scientific Training Gate

A central design principle of this repository is that **generated structures are not treated as property labels**.

The current defect library contains 15 configurations, but no validated defect-property dataset has yet been attached to them.

The supervised-learning framework therefore performs target validation before training.

Current status:

```text
Validated formation-energy labels: 0
Minimum required samples         : 30

Supervised classical training: inactive
Supervised GNN training      : inactive
```

This behavior is intentional.

The repository does **not** generate artificial formation energies simply to demonstrate model training. Supervised models become active only after validated computational or experimental labels are supplied.

---

# Graph Neural Network Framework

Each crystal structure is converted into a periodic graph.

## Graph Representation

Atoms are represented as nodes and neighboring atoms within a periodic cutoff are represented as directed edges.

The current graph representation contains:

```text
Node-feature dimension   : 16
Edge-feature dimension   : 17
Global-feature dimension : 28
```

### Node Features

Each atom is represented by a 16-dimensional elemental/atomic feature vector.

### Edge Features

Edges encode periodic interatomic geometry.

The 17-dimensional edge representation contains:

```text
1 raw interatomic distance
+
16 radial-basis distance features
```

This allows the network to distinguish different local bond environments instead of using connectivity alone.

### Global Features

Each graph also receives the 28-dimensional compact physics-informed descriptor vector.

The model therefore combines:

```text
Atomic identity
      +
Local periodic geometry
      +
Global defect physics
```

---

## Graph Dataset

The current graph dataset contains 15 structures.

For the pristine 72-atom GaN supercell, a representative graph generated during development contained:

```text
Nodes             : 72
Edges             : 1152
Node matrix       : (72, 16)
Edge attributes   : (1152, 17)
Global attributes : (1, 28)
```

Graph batching was also validated successfully.

---

# GNN Architectures

Two architectures are retained for comparison.

## Hybrid GNN Baseline

`GaNDefectGNN`

The baseline network combines:

- atomic node encoding
- GATv2 message passing
- global mean pooling
- global max pooling
- physics-informed global descriptors
- graph-level regression head

The baseline verifies the feasibility of combining learned graph embeddings with manually engineered defect descriptors.

---

## Edge-Aware GaN Defect GNN

`EdgeAwareGaNDefectGNN`

The final architecture extends the baseline by explicitly incorporating the 17-dimensional edge descriptors during message passing.

Conceptually:

```text
Node Features
     │
     ▼
Node Encoder
     │
     ▼
Edge-Aware GAT Blocks ◄──── Edge Features
     │
     ▼
Mean + Max Pooling
     │
     ▼
Graph Embedding
     │
     ├──────────────────────────┐
     │                          │
     │                    Global Features
     │                          │
     │                          ▼
     │                  Global Projection
     │                          │
     └──────────────┬───────────┘
                    ▼
             Hybrid Embedding
                    │
                    ▼
              Prediction Head
```

The edge-aware architecture successfully passed forward-propagation tests.

Most importantly, an edge-attribute sensitivity experiment confirmed that changing edge descriptors changes network predictions:

```text
Mean absolute output change : 0.00260814
Maximum output change       : 0.00546675
Edge attributes influence output: True
```

This verifies that the edge channel is functionally connected to the model output.

These values are **diagnostic results from an untrained model**, not scientific measures of bond importance.

---

# Explainable AI

GaN-DefectML contains separate explainability pipelines for tabular and graph-based models.

## Tabular XAI

Implemented methods include:

- permutation importance
- grouped physics-feature importance
- descriptor-group ablation
- SHAP support

Descriptor importance can be grouped into physically meaningful categories such as:

- composition
- structure
- defect identity
- chemical differences
- local environment

---

## Graph Explainability

The GNN explanation framework supports:

- node masking
- edge-feature masking
- graph-descriptor masking
- individual node importance
- chemical edge-pair importance
- graph-descriptor importance
- defect-site ranking
- defect-centered spatial analysis

For interstitial structures, the framework can explicitly identify the inserted defect atom.

For example:

```text
Configuration ID : Ga_i_Geo_I_1
Defect atom index: 72
Element          : Ga
Fractional coords: [0.0, 0.0, 0.1]
Cartesian coords : [0.0, 0.0, 1.038471]
```

This enables future analysis of whether a trained GNN actually localizes learned importance around the physical defect site.

---

## Diagnostic XAI vs Scientific XAI

The repository explicitly distinguishes between two modes.

### Diagnostic Mode

Before supervised training, perturbation-based XAI is used to verify that:

- node features affect predictions
- edge features affect predictions
- graph-level descriptors affect predictions
- defect atoms can be located and analyzed
- explanation algorithms execute correctly

These results test the **architecture and information flow**.

They do not demonstrate learned defect physics.

### Scientific Mode

Once a model has been trained and validated using reliable defect-property labels, the same tools can investigate:

- which atoms control predicted properties
- which local chemical interactions matter
- which descriptors dominate predictions
- whether importance is localized around defects
- how different defect families alter learned representations

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd GaN-DefectML
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

or Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Materials Project Access

Initial GaN structures are retrieved from the Materials Project database.

To reproduce the data-collection stage, a valid Materials Project API key is required.

The API key should be stored outside the repository, for example as an environment variable:

```bash
export MP_API_KEY="your_api_key"
```

Do not commit API credentials to GitHub.

---

# Basic Usage

Initialize the project:

```python
from src.utils import initialize_project

config = initialize_project()
```

Build or load the defect structure library, generate physics-informed descriptors, and construct graph representations using the corresponding modules under `src/`.

The default GNN can be created with:

```python
from models import build_edge_aware_gnn

model = build_edge_aware_gnn()
```

The model should only be trained once a validated target dataset satisfies the configured scientific-readiness requirements.

---

# Current Project Status

| Component | Status |
|---|---|
| Materials Project retrieval | Complete |
| GaN host analysis | Complete |
| Wurtzite supercell generation | Complete |
| Intrinsic defect generation | Complete |
| Dopant generation | Complete |
| Interstitial generation | Complete |
| Structure-library validation | Complete |
| Physics-informed descriptors | Complete |
| Chemical-difference encoding | Complete |
| Feature reduction | Complete |
| 28-feature compact matrix | Complete |
| Classical ML framework | Complete |
| Periodic graph construction | Complete |
| GNN batching | Complete |
| Hybrid GNN architecture | Complete |
| Edge-aware GNN architecture | Complete |
| Edge-channel sensitivity testing | Complete |
| Tabular XAI framework | Complete |
| GNN diagnostic XAI | Complete |
| Validated supervised target dataset | Pending |
| Scientific supervised training | Pending |
| Scientific XAI interpretation | Pending |

---

# Planned Extensions

The next scientific stage is the acquisition or computation of validated defect-property labels.

Priority targets include:

1. defect formation energies
2. charge-state-dependent formation energies
3. thermodynamic transition levels
4. relaxed electronic band gaps
5. defect-induced band-gap changes
6. donor/acceptor classification
7. carrier-type behavior

With sufficient labeled configurations, the framework can then perform:

- classical ML benchmarking
- GNN cross-validation
- hyperparameter optimization
- prediction uncertainty analysis
- model comparison
- scientifically interpretable XAI
- defect-property screening

A longer-term extension is integration with automated first-principles calculations so that new DFT results can continuously expand the supervised defect dataset.

---

# Scientific Scope

GaN-DefectML should currently be regarded as a **validated computational and machine-learning framework**, not a completed defect-property predictor.

The repository demonstrates that:

- physically meaningful GaN defect structures can be generated systematically,
- defect chemistry can be encoded numerically,
- periodic structures can be represented as crystal graphs,
- edge geometry can participate directly in GNN message passing,
- classical and graph-learning pipelines can be activated once sufficient labels exist,
- and explainability tools can probe both engineered descriptors and graph representations.

Predictive claims about GaN defect energetics or electronic properties require validated supervised labels and subsequent model evaluation.

---

# Technologies

- Python
- NumPy
- pandas
- SciPy
- Matplotlib
- scikit-learn
- PyTorch
- PyTorch Geometric
- Pymatgen
- Materials Project API
- SHAP

---

# Reproducibility

The default random seed is:

```text
42
```

The repository centralizes major learning, graph, and XAI parameters through the project configuration utilities.

Key defaults include:

```text
Graph cutoff              : 3.0 Å
Node feature dimension    : 16
Edge feature dimension    : 17
Global feature dimension  : 28

Classical CV folds        : 5
GNN CV folds              : 5
GNN hidden dimension      : 64
GNN dropout               : 0.2
GNN learning rate         : 1e-3
GNN weight decay          : 1e-5
Maximum GNN epochs        : 300
Early-stopping patience   : 30

Minimum regression labels : 30
Minimum classification    : 30
Permutation repeats       : 30
```

---

# Disclaimer

This repository is intended for computational materials-science research and methodological development.

Diagnostic outputs produced by randomly initialized or untrained models must not be interpreted as physical predictions or evidence of learned defect mechanisms.

Reliable scientific conclusions require validated labels, proper model training, independent evaluation, and comparison against established computational or experimental results.

---

## Author

**Vishal Chowdhary**

GaN-DefectML  
Physics-Informed Machine Learning and Graph Neural Networks for Defect Analysis in Gallium Nitride
