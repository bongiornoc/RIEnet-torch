# RIEnet Torch: A Rotational Invariant Estimator Network for GMV Optimization

**This library implements the neural estimators introduced in:**
- **Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2026). End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning. The Journal of Finance and Data Science: 100179. [10.1016/j.jfds.2026.100179](https://doi.org/10.1016/j.jfds.2026.100179)**
- **Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025). Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage. In *Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF ’25)*. [10.1145/3768292.3770370](https://doi.org/10.1145/3768292.3770370)**

**RIEnet Torch** is a PyTorch implementation for end-to-end global minimum-variance portfolio construction.

Given a tensor of asset returns, the model estimates a structured covariance / precision representation and produces analytic GMV portfolio weights in a single forward pass.

This repository is intended for:
- research and methodological replication,
- experimentation on large equity universes,
- integration into quantitative portfolio construction workflows,
- deployment as a standalone PyTorch package.

## What this package provides

- End-to-end training on a realized-variance objective for GMV portfolios
- Access to portfolio weights, cleaned covariance matrices, and precision matrices
- A dimension-agnostic architecture suitable for large cross-sectional universes
- A PyTorch implementation aligned with the published methodology

## Evidence in published experiments

The empirical properties of the method are documented in the associated papers.

In particular, the published experiments evaluate the model on large equity universes under a global minimum-variance objective and compare it against standard covariance-based benchmarks.

For details on datasets, training protocol, benchmark definitions, and evaluation metrics, please refer to the papers listed above.

## Module Organization

- `rienet_torch.trainable_layers`: modules with trainable parameters (`RIEnetLayer`, `LagTransformLayer`, `DeepLayer`, `DeepRecurrentLayer`, `CorrelationEigenTransformLayer`).
- `rienet_torch.ops_layers`: deterministic operation modules (statistics, normalization, eigensystem algebra, weight post-processing).
- `rienet_torch.losses`: the GMV variance objective.
- `rienet_torch.serialization`: Torch-native save/load helpers.

## Installation

Install from PyPI:

```bash
pip install rienet-torch
```

Or install from source:

```bash
cd /path/to/RIEnet-torch
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import torch
from rienet_torch import RIEnetLayer, variance_loss_function

# Defaults reproduce the compact GMV architecture
# (bidirectional GRU with 16 units, 8-unit volatility head)
rienet_layer = RIEnetLayer(output_type=["weights", "precision"])

# Sample data: (batch_size, n_stocks, n_days)
returns = torch.randn(32, 10, 60) * 0.02

# Retrieve GMV weights and cleaned precision in one pass
outputs = rienet_layer(returns)
weights = outputs["weights"]      # (32, 10, 1)
precision = outputs["precision"]  # (32, 10, 10)

# GMV training objective
covariance = torch.randn(32, 10, 10)
covariance = covariance @ covariance.transpose(-1, -2)
loss = variance_loss_function(covariance, weights)
print(loss.shape)  # torch.Size([32, 1, 1])
```

### Training with the GMV Variance Loss

```python
import torch
from rienet_torch import RIEnetLayer, variance_loss_function

model = RIEnetLayer(output_type="weights")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Synthetic training data
X_train = torch.randn(256, 10, 60) * 0.02
Sigma_train = torch.randn(256, 10, 10)
Sigma_train = Sigma_train @ Sigma_train.transpose(-1, -2)

model.train()
for step in range(100):
    optimizer.zero_grad()
    weights = model(X_train)
    loss = variance_loss_function(Sigma_train, weights).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

> **Tip:** When you intend to deploy RIEnet on portfolios of varying size, train on batches that span different asset universes. The RIE-based architecture is dimension agnostic and benefits from heterogeneous training shapes.

### Using Different Output Types

```python
import torch
from rienet_torch import RIEnetLayer

returns = torch.randn(32, 10, 60) * 0.02

# GMV weights only
weights = RIEnetLayer(output_type="weights")(returns)

# Precision matrix only
precision_matrix = RIEnetLayer(output_type="precision")(returns)

# Precision, covariance, and lag-transformed inputs in one pass
outputs = RIEnetLayer(
    output_type=["precision", "covariance", "input_transformed"]
)(returns)
precision_matrix = outputs["precision"]
covariance_matrix = outputs["covariance"]
lagged_inputs = outputs["input_transformed"]

# Spectral components (non-inverse)
spectral = RIEnetLayer(
    output_type=["eigenvalues", "eigenvectors", "transformed_std"]
)(returns)
cleaned_eigenvalues = spectral["eigenvalues"]   # (batch, n_stocks, 1)
eigenvectors = spectral["eigenvectors"]         # (batch, n_stocks, n_stocks)
transformed_std = spectral["transformed_std"]   # (batch, n_stocks, 1)

# Optional: disable variance normalisation
raw_covariance = RIEnetLayer(
    output_type="covariance",
    normalize_transformed_variance=False
)(returns)
```

> When RIEnet is trained end-to-end on the GMV variance loss, leave `normalize_transformed_variance=True` (the default). The loss is invariant to global covariance rescalings and the layer keeps the implied variance scale centred around one. Disable the normalization only when using alternative objectives where the absolute volatility scale must be preserved.

### Using `LagTransformLayer` Directly

`LagTransformLayer` is exposed both at package root and in the dedicated module:

```python
import torch
from rienet_torch import LagTransformLayer
# or: from rienet_torch.lag_transform import LagTransformLayer

# Dynamic lookback (T can change call-by-call)
compact = LagTransformLayer(variant="compact")
y1 = compact(torch.randn(4, 12, 20))
y2 = compact(torch.randn(4, 12, 40))

# Fixed lookback inferred at first build/call (requires static T)
per_lag = LagTransformLayer(variant="per_lag")
z1 = per_lag(torch.randn(4, 12, 20))
z2 = per_lag(torch.randn(4, 8, 20))  # n_assets can change
```

### Using `EigenWeightsLayer` Directly

`EigenWeightsLayer` is part of the public API and can be imported directly:

```python
import torch
from rienet_torch import EigenWeightsLayer

layer = EigenWeightsLayer(name="gmv_weights")

eigenvectors = torch.randn(8, 20, 20)
inverse_eigenvalues = torch.rand(8, 20, 1)
inverse_std = torch.rand(8, 20, 1)

# Full GMV-like branch (includes inverse_std scaling)
weights = layer(eigenvectors, inverse_eigenvalues, inverse_std)

# Covariance-eigensystem branch (inverse_std omitted)
weights_cov = layer(eigenvectors, inverse_eigenvalues)
```

Notes:
- `inverse_std` is optional by design.
- Output shape is always `(..., n_assets, 1)`, normalized to sum to one along assets.

### Using `CorrelationEigenTransformLayer` Directly

```python
import torch
from rienet_torch import CorrelationEigenTransformLayer

layer = CorrelationEigenTransformLayer(name="corr_cleaner")

# Correlation matrix: (batch, n_assets, n_assets)
corr = torch.eye(6).expand(4, 6, 6).clone()

# Optional attributes: (batch, k) e.g. q, lookback, regime flags, etc.
attrs = torch.tensor([
    [0.5, 60.0],
    [0.7, 60.0],
    [1.2, 30.0],
    [0.9, 90.0],
], dtype=torch.float32)

# With attributes (default output_type='correlation')
cleaned_corr = layer(corr, attributes=attrs)

# Request multiple outputs
details = layer(
    corr,
    attributes=attrs,
    output_type=[
        "correlation",
        "inverse_correlation",
        "eigenvalues",
        "eigenvectors",
        "inverse_eigenvalues",
    ],
)
cleaned_eigvals = details["eigenvalues"]              # (batch, n_assets, 1)
cleaned_inv_eigvals = details["inverse_eigenvalues"]  # (batch, n_assets, 1)
inv_corr = details["inverse_correlation"]             # (batch, n_assets, n_assets)

# Without attributes
cleaned_corr_no_attr = CorrelationEigenTransformLayer(name="corr_cleaner_no_attr")(corr)
```

Notes:
- `attributes` is optional and can have shape `(batch, k)` or `(batch, n_assets, k)`.
- The output is a cleaned correlation matrix `(batch, n_assets, n_assets)`.
- If you change attribute width `k`, use a new layer instance.

## Loss Function

### Variance Loss Function

```python
from rienet_torch import variance_loss_function

loss = variance_loss_function(
    covariance_true=true_covariance,    # (batch_size, n_assets, n_assets)
    weights_predicted=predicted_weights # (batch_size, n_assets, 1)
)
```

**Mathematical Formula:**

```text
Loss = n_assets x w^T Sigma w
```

Where `w` are the portfolio weights and `Sigma` is the realised covariance matrix.

## Serialization

The package uses PyTorch-native serialization:

```python
from rienet_torch import RIEnetLayer
from rienet_torch.serialization import save_module, load_module

model = RIEnetLayer(output_type="weights")
save_module(model, "rienet.pt")

restored = load_module(RIEnetLayer, "rienet.pt")
```

This stores:
- the `state_dict`,
- the module config when `get_config()` is available,
- the lazy-build metadata needed to materialize shape-dependent parameters.

## Architecture Details

The RIEnet pipeline consists of:

1. **Input Scaling**: annualise returns by 252
2. **Lag Transformation**: five-parameter memory kernel for temporal weighting
3. **Covariance Estimation**: sample covariance across assets
4. **Eigenvalue Decomposition**: spectral analysis of the covariance matrix
5. **Recurrent Cleaning**: bidirectional GRU/LSTM processing of eigen spectra
6. **Marginal Volatility Head**: dense network forecasting inverse standard deviations
7. **Matrix Reconstruction**: RIE-based synthesis of `Sigma^{-1}` and GMV weight normalisation

Paper defaults use a single bidirectional GRU layer with 16 units per direction and a marginal-volatility head with 8 hidden units, matching the compact network described in Bongiorno et al. (2025).

## Requirements

- Python >= 3.10
- PyTorch >= 2.5.0
- NumPy >= 1.26.0

## Development

```bash
cd /path/to/RIEnet-torch
pip install -e ".[dev]"
python -m pytest tests/
```

## Release Automation

This repository is ready for automated publishing with GitHub Actions:

- CI workflow: `.github/workflows/ci.yml`
- PyPI publish workflow: `.github/workflows/publish.yml`
- TestPyPI publish workflow: `.github/workflows/publish-testpypi.yml`

The release checklist is documented in `RELEASING.md`.

## Citation

Please cite the following references when using RIEnet:

```bibtex
@article{bongiorno2026end,
  title={End-to-end large portfolio optimization for variance minimization with neural networks through covariance cleaning},
  author={Bongiorno, Christian and Manolakis, Efstratios and Mantegna, Rosario Nunzio},
  journal={The Journal of Finance and Data Science},
  pages={100179},
  year={2026},
  publisher={Elsevier}
}

@inproceedings{bongiorno2025Neural,
  author = {Bongiorno, Christian and Manolakis, Efstratios and Mantegna, Rosario Nunzio},
  title = {Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage},
  year = {2025},
  isbn = {9798400722202},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3768292.3770370},
  doi = {10.1145/3768292.3770370},
  booktitle = {Proceedings of the 6th ACM International Conference on AI in Finance},
  pages = {449–455},
  numpages = {7},
  location = {},
  series = {ICAIF '25}
}
```

For software citation:

```bibtex
@software{rienet_torch2026,
  title={RIEnet Torch: A Rotational Invariant Estimator Network for Global Minimum-Variance Optimisation},
  author={Bongiorno, Christian},
  year={2026},
  version={0.1.0},
  url={https://github.com/your-user/rienet-torch}
}
```

You can print citation information programmatically:

```python
import rienet_torch
rienet_torch.print_citation()
```


## Support

For questions, issues, or contributions, please:

- Open an issue on [GitHub](https://github.com/bongiornoc/RIEnet-torch/issues)
- Check the documentation
- Contact Prof. Christian Bongiorno (<christian.bongiorno@centralesupelec.fr>) for calibrated model weights or collaboration requests
