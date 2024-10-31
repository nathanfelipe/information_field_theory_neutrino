# required packages
import nifty8.re as jft
import jax
from jax import random, numpy as jnp
import healpy as hp
import sys

import matplotlib.pyplot as plt

cfm_zero_mode = dict(offset_mean=-3,
                     offset_std=(2.0, 1.0))

cfm_fluctuations = dict(fluctuations=(1.0, 0.5),
                        loglogavgslope=(-3.0, 1.5),
                        flexibility=(2.0, 1.0),
                        asperity=(2.0, 1.0))

# create a seed
seed = 934857243958
key = random.PRNGKey(seed)


# space info
dims = 8 # (128,128)
distances = 1.0/dims

fixed_cfm_zero_mode = dict(offset_mean=-3,
                           offset_std=lambda x: 2.5)

fixed_cfm_fluctuations = dict(fluctuations=lambda x: 2.0,
                              loglogavgslope=lambda x: -4.0,
                              flexibility=lambda x: 1.5,
                              asperity=lambda x: 1.5)

# create the cfm
fcfm = jft.CorrelatedFieldMaker('fixed_params')
fcfm.add_fluctuations(dims,
                     distances=distances,
                     **fixed_cfm_fluctuations,
                     prefix="",
                     non_parametric_kind="power",
                     harmonic_type='spherical')
fcfm.set_amplitude_total_offset(**cfm_zero_mode)

fixed_ps = fcfm.power_spectrum
fixed_correlated_field = fcfm.finalize()

key, subkey = random.split(key)
rand_pos = jft.random_like(subkey, fixed_correlated_field.domain)

ground_truth = fixed_correlated_field(rand_pos)
fixed_pow_spec = fixed_ps(rand_pos)

noise_cov = 1.0
noise_cov_inv = lambda x: noise_cov**-1 * x

key, subkey = random.split(key)
noisy_data = ground_truth + noise_cov*random.normal(subkey, ground_truth.shape)

# fig, axs = plt.subplots(1,2)

fig = plt.figure(figsize=(8, 10))

# First subplot: Truth
ax1 = fig.add_subplot(2, 1, 1)
hp.mollview(ground_truth,
            title="Truth", cmap='viridis', sub=(2, 1, 1),
            unit='fluctuation',
            )
hp.graticule()

# Second subplot: Data
ax2 = fig.add_subplot(2, 1, 2)
hp.mollview(noisy_data,
            title="Data = Truth + Noise", cmap='viridis', sub=(2, 1, 2),
            unit='fluctuation',
            )
hp.graticule()

# Save the figure
plt.savefig("CFM_synthetic")
plt.show()

# recreate the cfm
cfm = jft.CorrelatedFieldMaker('tests')
cfm.add_fluctuations(dims,
                     distances=distances,
                     **cfm_fluctuations,
                     prefix="",
                     non_parametric_kind="power",
                     harmonic_type='spherical'
                     )
cfm.set_amplitude_total_offset(**cfm_zero_mode)

cfm_ps = cfm.power_spectrum
correlated_field = cfm.finalize()

likelihood = jft.Gaussian(noisy_data, noise_cov_inv=noise_cov_inv).amend(correlated_field)

n_vi_iterations = 10
delta = 1e-4
n_samples = 4

key, k_i, k_o = random.split(key, 3)
# NOTE, changing the number of samples always triggers a resampling even if
# `resamples=False`, as more samples have to be drawn that did not exist before.
samples, state = jft.optimize_kl(
    likelihood,
    jft.Vector(likelihood.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 6 else n_samples,
    # Source for the stochasticity for sampling
    key=k_o,
    # Names of parameters that should not be sampled but still optimized
    # can be specified as point_estimates (effectively we are doing MAP for
    # these degrees of freedom).
    # point_estimates=("cfax1flexibility", "cfax1asperity"),
    # Arguments for the conjugate gradient method used to drawing samples from
    # an implicit covariance matrix
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(likelihood.domain) / 10.0, maxiter=100),
    ),
    # Arguments for the minimizer in the nonlinear updating of the samples
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="nonlinear_resample",
    odir="results_intro",
    resume=False,
)

post_sr_mean = jft.mean(tuple(correlated_field(s) for s in samples))
post_ps_mean = jft.mean(tuple(cfm_ps(s) for s in samples))
grid = correlated_field.target_grids[0]
to_plot = [
    ("Signal", ground_truth, "im"),
    ("Data", noisy_data, "im"),
    ("Reconstruction", post_sr_mean, "im"),
    (
        "Amplitude spectrum",
        (
            grid.harmonic_grid.mode_lengths,
            fixed_pow_spec,
            post_ps_mean,
        ),
        "loglog",
    ),
]


fig, axs = plt.subplots(2, 2)
for ax, v in zip(axs.flat, to_plot):
    title, field, tp, *labels = v
    ax.set_title(title)
    if tp == "im":
        end = tuple(n * d for n, d in zip(grid.shape, grid.distances))
        im = ax.imshow(field, cmap="inferno")
        plt.colorbar(im, ax=ax, orientation="horizontal")
    else:
        ax_plot = ax.loglog if tp == "loglog" else ax.plot
        x = field[0]
        for f in field[1:]:
            ax_plot(x, f, alpha=0.7)
for ax in axs.flat[len(to_plot) :]:
    ax.set_axis_off()
fig.tight_layout()
fig.savefig("results_intro_full_reconstruction.png", dpi=400)
plt.show()
