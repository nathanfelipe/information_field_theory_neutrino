import nifty8
import jax
from jax import numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import nifty8.re as jft
jax.config.update("jax_enable_x64", True)


# Signal Model (for synthetic and neutrino signal)
class SignalModel(jft.Model):
    def __init__(self, grid, amplitude_spectrum):
        self.grid = grid
        self.amplitude_spectrum = amplitude_spectrum
        super().__init__(domain=jax.ShapeDtypeStruct(shape=grid.shape, dtype=jnp.float64))

    def __call__(self, x):
        harmonic_transform = jft.correlated_field.hartley()
        harmonic_dvol = 1 / self.grid.total_volume

        # Ensure x is transformed to harmonic space
        harmonic_x = harmonic_transform(x)

        # Now apply the amplitude spectrum in harmonic space
        result_in_harmonic_space = self.amplitude_spectrum * harmonic_x

        # Return the inverse harmonic transform (back to spatial domain)
        return harmonic_dvol * harmonic_transform(result_in_harmonic_space)

class SignalResponse(jft.Model):
    def __init__(self, signal, sensitivity):
        self.signal = signal
        self.sensitivity = sensitivity
        super().__init__(domain=signal.domain)

    def __call__(self, x):
        return self.signal(x) * self.sensitivity


def amplitude_spectrum(grid):
    k = grid.harmonic_grid.mode_lengths  # Get the harmonic mode lengths
    a = 0.02 / (1 + k**2)  # Define the amplitude spectrum based on mode lengths
    a = a[grid.harmonic_grid.power_distributor]  # Apply power distributor to the spectrum
    return a


class FixedPowerCorrelatedField(jft.Model):
    def __init__(self, grid, a):
        self.grid = grid
        self.a = a
        super().__init__(
            domain=jax.ShapeDtypeStruct(shape=grid.shape, dtype=jnp.float64)
            )

    def __call__(self, x):

        # ht = jft.correlated_field.hartley
        harmonic_dvol = 1 / self.grid.total_volume
        return harmonic_dvol * ht(self.a * x)


# Wiener Filtering Procedure (for synthetic and neutrino data)
class WienerFilter:
    def __init__(self, signal_model, noise_std=0.1):
        self.signal_model = signal_model
        self.noise_std = noise_std

    def apply(self, data, key, n_samples=30):
        # Define noise covariance
        noise_cov = lambda x: self.noise_std ** 2 * x
        noise_cov_inv = lambda x: self.noise_std ** -2 * x

        # Likelihood model
        lh = jft.Gaussian(data, noise_cov_inv).amend(self.signal_model)

        # Perform Wiener filtering
        delta = 1e-6
        samples, info = jft.wiener_filter_posterior(
            lh,
            key=key,
            n_samples=n_samples,
            draw_linear_kwargs=dict(
                cg_name="W",
                cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
            ),
        )

        post_mean, post_std = jft.mean_and_std(tuple(self.signal_model(s) for s in samples))
        return post_mean, post_std


# Visualization function
def visualize_results(signal_truth, data, post_mean, post_std):
    fig, axs = plt.subplots(2, 2, figsize=(8, 10), constrained_layout=True)

    # Plot ground truth
    axs[0, 0].set_title('Ground Truth')
    im = axs[0, 0].imshow(signal_truth.T, origin='lower')
    plt.colorbar(im, ax=axs[0, 0])

    # Plot noisy data
    axs[0, 1].set_title('Noisy Data')
    im = axs[0, 1].imshow(data.T, origin='lower')
    plt.colorbar(im, ax=axs[0, 1])

    # Plot posterior mean
    axs[1, 0].set_title('Posterior Mean')
    im = axs[1, 0].imshow(post_mean.T, origin='lower')
    plt.colorbar(im, ax=axs[1, 0])

    # Plot posterior std
    axs[1, 1].set_title('Posterior Std')
    im = axs[1, 1].imshow(post_std.T, origin='lower')
    plt.colorbar(im, ax=axs[1, 1])

    plt.show()