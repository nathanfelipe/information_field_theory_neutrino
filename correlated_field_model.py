import nifty8.re as jft
from jax import random
import healpy as hp
import matplotlib.pyplot as plt


class GenerateSyntheticData:
    def __init__(self, seed):
        self.seed = seed
        self.key = random.PRNGKey(seed)

    def generate_key(self):
        self.key, subkey = random.split(self.key)
        return subkey

    def generate_noisy_data(self, ground_truth, noise_cov):
        self.key, subkey = random.split(self.key)
        noisy_data = ground_truth + noise_cov * random.normal(subkey, ground_truth.shape)
        return noisy_data


class CorrelatedFieldModel:
    def __init__(self, dims):
        self.dims = dims
        self.distances = 1.0 / dims
        self.cfm_zero_mode = dict(offset_mean=-3, offset_std=(2.0, 1.0))
        self.cfm_fluctuations = dict(fluctuations=(1.0, 0.5),
                                     loglogavgslope=(-3.0, 1.5),
                                     flexibility=(2.0, 1.0),
                                     asperity=(2.0, 1.0))
        self.fixed_cfm_zero_mode = dict(offset_mean=-3, offset_std=lambda x: 2.5)
        self.fixed_cfm_fluctuations = dict(fluctuations=lambda x: 2.0,
                                           loglogavgslope=lambda x: -4.0,
                                           flexibility=lambda x: 1.5,
                                           asperity=lambda x: 1.5)


class CorrelatedFieldModel:
    def __init__(self, dims):
        self.dims = dims
        self.distances = 1.0 / dims
        self.cfm_zero_mode = dict(offset_mean=-3, offset_std=(2.0, 1.0))
        self.cfm_fluctuations = dict(fluctuations=(1.0, 0.5),
                                     loglogavgslope=(-3.0, 1.5),
                                     flexibility=(2.0, 1.0),
                                     asperity=(2.0, 1.0))
        self.fixed_cfm_zero_mode = dict(offset_mean=-3, offset_std=lambda x: 2.5)
        self.fixed_cfm_fluctuations = dict(fluctuations=lambda x: 2.0,
                                           loglogavgslope=lambda x: -4.0,
                                           flexibility=lambda x: 1.5,
                                           asperity=lambda x: 1.5)

    def create_field_model(self):
        cfm = jft.CorrelatedFieldMaker('neutrino_model')
        cfm.add_fluctuations(
                    self.dims,
                    distances=self.distances,
                    **self.cfm_fluctuations,
                    prefix="",
                    non_parametric_kind="power",
                    harmonic_type='spherical'
                )
        cfm.set_amplitude_total_offset(**self.cfm_zero_mode)
        power_spectrum = cfm.power_spectrum
        correlated_field = cfm.finalize()
        return correlated_field, power_spectrum


    def create_field_model(self):
        fcfm = jft.CorrelatedFieldMaker('fixed_params')
        fcfm.add_fluctuations(self.dims,
                              distances=self.distances,
                              **self.fixed_cfm_fluctuations,
                              prefix="", non_parametric_kind="power",
                              harmonic_type='spherical')
        fcfm.set_amplitude_total_offset(**self.cfm_zero_mode)
        return fcfm.finalize(), fcfm.power_spectrum


    def plot_field(self, ground_truth, noisy_data):
        fig = plt.figure(figsize=(8, 10))

        # First subplot: Truth
        ax1 = fig.add_subplot(2, 1, 1)
        hp.mollview(ground_truth, title="Truth", cmap='viridis', sub=(2, 1, 1), unit='fluctuation')
        hp.graticule()

        # Second subplot: Data
        ax2 = fig.add_subplot(2, 1, 2)
        hp.mollview(noisy_data, title="Data = Truth + Noise", cmap='viridis', sub=(2, 1, 2), unit='fluctuation')
        hp.graticule()

        plt.savefig("CFM_synthetic")
        plt.show()


class CorrelatedFieldModelNeutrino:
    def __init__(self, dims):
        self.dims = dims
        self.distances = 1.0 / dims
        self.cfm_zero_mode = dict(offset_mean=-3, offset_std=(2.0, 1.0))
        self.cfm_fluctuations = dict(
            fluctuations=(1.0, 0.5),
            loglogavgslope=(-3.0, 1.5),
            flexibility=(2.0, 1.0),
            asperity=(2.0, 1.0)
        )


class PlotResults:
    def __init__(self, grid):
        self.grid = grid

    def plot(self, to_plot, filename="results_reconstruction.png"):
        fig, axs = plt.subplots(2, 2)
        for ax, v in zip(axs.flat, to_plot):
            title, field, tp, *labels = v
            ax.set_title(title)
            if tp == "im":
                end = tuple(n * d for n, d in zip(self.grid.shape, self.grid.distances))
                im = ax.imshow(field, cmap="inferno")
                plt.colorbar(im, ax=ax, orientation="horizontal")

            elif tp == "spherical":
                hp.mollview(field, title=title, cmap='viridis')
                hp.graticule()
                plt.savefig(f"{title}_spherical.png")

            else:  # loglog plot for the amplitude spectrum
                ax_plot = ax.loglog if tp == "loglog" else ax.plot
                x = field[0]
                for f in field[1:]:
                    ax_plot(x, f, alpha=0.7)

        # Hide unused subplots if any
        for ax in axs.flat[len(to_plot):]:
            ax.set_axis_off()

        # Final layout and save
        fig.tight_layout()
        fig.savefig(filename, dpi=400)
        plt.show()