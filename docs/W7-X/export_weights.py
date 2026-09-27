"""Export the trained pressure-parametric W7-X PINN (equinox .eqx) to JSON for the web viewer.

Usage (from Scimba_dev_project/Wx7/):
    python PINN_WebViewer/export_weights.py [path/to/trained_space.eqx] [M N]

Defaults: the SLURM W7x/10k_epochs_P_Parametric checkpoint, with M, N from
inference_only.py (M = N = 12 -> K = 313, last layer 3*K = 939).
Pass M N explicitly for a checkpoint trained with another mode set.

Writes PINN_WebViewer/model.json (and the identical model.js, loaded by index.html) containing:
  * the MLP layers (weight [out, in], bias [out]) -- tanh on hidden layers, linear last layer
  * the Fourier modes (m, n), nfp, boundary coefficients Rb/Zb aligned with the modes
  * the VMEC magnetic-axis coefficients Ra/Za used by initialize_axis_bias (Thun eq. 37),
    so the page can draw the initial-guess surfaces as a dashed reference
  * the mu training box and a few reference (R, Z) values computed here in JAX,
    so the browser can check its own forward pass against them at load time
  * reference <beta>_vol values from Postprocess_Beta.compute_beta (the real scimba
    |B| / det_g chain) on the 24^3 grid the browser uses, for the same check on beta.
    This is slow (~11 min per mu on the scimba chain); set SKIP_BETA=1 to skip it.

Only mu[0] (pressure amplitude) was varied in training; mu[1] (iota amplitude) is
still a network input but was held at 1.0 (domain_mu = [(0.1, 5.0), (1.0, 1.0)]),
so the viewer pins it to 1.0.

Run configuration (nfp, Rb/Zb, architecture) is imported from
inference_only.py, which MUST match ParametricMHD_Pressure_Routine.py's make_space().
"""
import json
import os
import sys

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import inference_only
from inference_only import MU_DIM, Rb_coeffs, Zb_coeffs, nfp
from Postprocess_Beta import compute_beta, pressure_stellarator_dudt
from scimba_jax.nonlinear_approximation.approximation_spaces.approximation_spaces import (
    ApproximationSpace,
)
from scimba_jax.nonlinear_approximation.networks.mlp import MLP

DOMAIN_MU = [(0.1, 5.0), (1.0, 1.0)]  # [pressure amplitude, iota amplitude (fixed)]
# magnetic axis (VMEC raxis_c / zaxis_s), copy of ParametricMHD_Pressure_Routine.py
Ra_coeffs = {(0, 0): 5.6343, (0, 1): 0.35209}
Za_coeffs = {(0, 1): -0.29578}

DEFAULT_PATH = "/mnt/c/Users/matte/OneDrive/Desktop/results/SLURM/W7x/10k_epochs_P_Parametric/trained_space.eqx"
path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
M, N = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (inference_only.M, inference_only.N)
modes = [(m, n) for m in range(0, M + 1) for n in range(-N, N + 1) if not (m == 0 and n < 0)]
K = len(modes)

nn = MLP(in_size=1 + MU_DIM, out_size=3 * K, hidden_sizes=[24, 24], key=jax.random.PRNGKey(0))
template = ApproximationSpace({"x": 3, "mu": MU_DIM}, [(nn, "vec", 3 * K)], model_type="x_mu")
space = eqx.tree_deserialise_leaves(path, template)
mlp = space.models[0]

m_arr = jnp.array([m for m, _ in modes], dtype=jnp.float64)
n_arr = jnp.array([n for _, n in modes], dtype=jnp.float64)
Rb = jnp.array([Rb_coeffs.get(k, 0.0) for k in modes])
Zb = jnp.array([Zb_coeffs.get(k, 0.0) for k in modes])


def RZ(rho, theta, zeta, mu):
    # same as StationaryMHDForceBalance._make_RlZ: network sees [2 rho^2 - 1, mu0, mu1]
    out = mlp(jnp.concatenate([jnp.atleast_1d(2.0 * rho * rho - 1.0), mu]))
    cR, cZ = out[0:K], out[2 * K : 3 * K]
    phase = m_arr * theta - n_arr * nfp * zeta
    rho_m = rho**m_arr
    dist = 1.0 - rho * rho
    R = jnp.dot(rho_m * (Rb + dist * cR), jnp.cos(phase))
    Z = jnp.dot(rho_m * (Zb + dist * cZ), jnp.sin(phase))
    return float(R), float(Z)


refs = []
for rho, theta, zeta, mu in [
    (0.0, 0.0, 0.0, [1.0, 1.0]),
    (0.5, 1.0, np.pi / nfp, [1.0, 1.0]),
    (1.0, 2.0, 0.1, [1.0, 1.0]),
    (0.8, 4.0, np.pi / (2 * nfp), [0.1, 1.0]),
    (0.3, 5.5, 0.0, [5.0, 1.0]),
]:
    R, Z = RZ(rho, theta, zeta, jnp.array(mu))
    refs.append(dict(rho=rho, theta=theta, zeta=zeta, mu=mu, R=R, Z=Z))

# reference <beta>_vol from the scimba chain; build_model() reads the module-level
# modes, so point it at this checkpoint's mode set first
BETA_N = 24  # must match BETA_N in index.html
inference_only.modes, inference_only.K = modes, K
eq_model = inference_only.build_model()
beta_refs = []
for mu in ([] if os.environ.get("SKIP_BETA") else [[5.0, 1.0]]):
    r = compute_beta(eq_model, space, nfp, pressure_stellarator_dudt, jnp.array(mu),
                     n_rho=BETA_N, n_theta=BETA_N, n_zeta=BETA_N, batch_size=4096)
    beta_refs.append(dict(mu=mu, n=BETA_N, beta_SI=r["beta_SI"], p_avg=r["p_avg"], B2_avg=r["B2_avg"]))

model = dict(
    name="Pressure-parametric W7-X PINN (NFP=5)",
    source=os.path.relpath(path, HERE),
    nfp=nfp,
    modes=modes,
    Rb=np.asarray(Rb).tolist(),
    Zb=np.asarray(Zb).tolist(),
    Ra=[Ra_coeffs.get(k, 0.0) for k in modes],
    Za=[Za_coeffs.get(k, 0.0) for k in modes],
    domain_mu=DOMAIN_MU,
    layers=[
        dict(W=np.asarray(l.weight).tolist(), b=np.asarray(l.bias).tolist())
        for l in mlp.layers
    ],
    reference=refs,
    reference_beta=beta_refs,
)
out_path = os.path.join(HERE, "model.json")
with open(out_path, "w") as f:
    json.dump(model, f)
# same content as a <script>-loadable file: works from file:// (no fetch/CORS) and on GitHub Pages
js_path = os.path.join(HERE, "model.js")
with open(js_path, "w") as f:
    f.write("window.PINN_MODEL = ")
    json.dump(model, f)
    f.write(";\n")
print(f"loaded {path}")
print(f"layers: {[np.asarray(l.weight).shape for l in mlp.layers]}  M={M} N={N} K={K}  MU_DIM={MU_DIM}")
print(f"wrote {out_path} and {js_path} ({os.path.getsize(out_path)/1024:.0f} KiB each)")
for r in refs + beta_refs:
    print(r)
