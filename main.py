from data_loader import DataLoader
from neutrino_energy_histogram import NeutrinoEnergyHistogram
from healpix_energy_map import HealpixEnergyMap
from jax import random
import nifty8.re as jft
from wiener_filter import amplitude_spectrum, SignalModel, WienerFilter, visualize_results, SignalResponse
from correlated_field_model import (GenerateSyntheticData,
                                    CorrelatedFieldModel, PlotResults, CorrelatedFieldModelNeutrino)
import numpy as np

def main():
    # Load the data
    file_path_pattern = 'neutrino_dataset/dataverse_files/events/IC86_*.csv'
    loader = DataLoader(file_path_pattern)
    data = loader.load_multiple_files()  # or loader.load_single_file('specific_file.csv')

    # Ask user if they want to plot the histogram
    plot_histogram = input("Do you want to plot the neutrino energy histogram? (yes/no): ").strip().lower()

    if plot_histogram == 'yes':
        # Plot neutrino energy histogram
        histogram_plotter = NeutrinoEnergyHistogram(data)
        histogram_plotter.plot_histogram('neutrino_counts_by_energy.png')

    # Ask user if they want to plot the HEALPix energy map
    plot_healpix_map = input("Do you want to plot the HEALPix energy map? (yes/no): ").strip().lower()

    if plot_healpix_map == 'yes':
        # Plot HEALPix energy map
        healpix_plotter = HealpixEnergyMap(data)
        healpix_plotter.plot_healpix_map('healpy_neutrino_energy_maps.png')

    valid_options = ['synthetic', 'neutrino']
    data_type = None

    while True:
        data_type = get_valid_data_type(valid_options)
        if data_type == 'synthetic':
            synthetic_analysis()
        elif data_type == 'neutrino':
            neutrino_analysis()
        else:
            print("Only two options are valid: 'synthetic' or 'neutrino'. Please try again.")

        # After the analysis, ask if the user wants to perform another one
        while True:
            again = input("Would you like to perform another analysis? Type 'yes' or 'no': ").strip().lower()
            if again in ['yes', 'no']:
                break
            else:
                print("Please type 'yes' or 'no'.")
        if again == 'no':
            print("Exiting the program.")
            break  # Exit the loop and end the program


def get_valid_data_type(valid_options):
    while True:
        data_type = input("Do you want to perform a synthetic analysis or neutrino analysis?"
                          " Type 'synthetic' or 'neutrino': ").strip().lower()
        if data_type in valid_options:
            return data_type
        else:
            print("Only two options are valid: 'synthetic' or 'neutrino'. Please try again.")


def synthetic_analysis():
    seed = 934857243958
    synthetic_data_gen = GenerateSyntheticData(seed=seed)

    # Correlated field model setup
    dims = 64
    correlated_field_model = CorrelatedFieldModel(dims=dims)
    fixed_correlated_field, fixed_ps = correlated_field_model.create_field_model()

    # Generate random positions and ground truth
    subkey = synthetic_data_gen.generate_key()
    rand_pos = jft.random_like(subkey, fixed_correlated_field.domain)
    ground_truth = fixed_correlated_field(rand_pos)
    fixed_pow_spec = fixed_ps(rand_pos)

    # Generate noisy data
    noise_cov = 1.0
    noisy_data = synthetic_data_gen.generate_noisy_data(ground_truth, noise_cov)

    # Plot the results
    correlated_field_model.plot_field(ground_truth, noisy_data)

    # Recreate the CFM for further analysis
    cfm = jft.CorrelatedFieldMaker('cfm_synthetic')
    cfm.add_fluctuations(dims, distances=1.0 / dims, **correlated_field_model.cfm_fluctuations,
                         prefix="", non_parametric_kind="power", harmonic_type='spherical')
    cfm.set_amplitude_total_offset(**correlated_field_model.cfm_zero_mode)
    cfm_ps = cfm.power_spectrum
    correlated_field = cfm.finalize()

    # Set up likelihood
    noise_cov_inv = lambda x: noise_cov ** -1 * x
    likelihood = jft.Gaussian(noisy_data, noise_cov_inv=noise_cov_inv).amend(correlated_field)

    n_vi_iterations = 10
    delta = 1e-4
    n_samples = 4

    key, k_i, k_o = random.split(synthetic_data_gen.key, 3)
    samples, state = jft.optimize_kl(
        likelihood,
        jft.Vector(likelihood.init(k_i)),
        n_total_iterations=n_vi_iterations,
        n_samples=lambda i: n_samples // 2 if i < 6 else n_samples,
        key=k_o,
        draw_linear_kwargs=dict(
            cg_name="SL",
            cg_kwargs=dict(absdelta=delta * jft.size(likelihood.domain) / 10.0, maxiter=100),
        ),
        nonlinearly_update_kwargs=dict(
            minimize_kwargs=dict(
                name="SN", xtol=delta, cg_kwargs=dict(name=None), maxiter=5,
            )
        ),
        kl_kwargs=dict(
            minimize_kwargs=dict(
                name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35,
            )
        ),
        sample_mode="nonlinear_resample",
        odir="results_intro",
        resume=False,
    )

    # Post-processing results
    post_sr_mean = jft.mean(tuple(correlated_field(s) for s in samples))
    post_ps_mean = jft.mean(tuple(cfm_ps(s) for s in samples))
    grid = correlated_field.target_grids[0]
    to_plot = [
        ("Signal", ground_truth, "spherical"),
        ("Data", noisy_data, "spherical"),
        ("Reconstruction", post_sr_mean, "spherical"),
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

    # Create an instance of the PlotResults class and call the plot method
    plotter = PlotResults(grid)
    plotter.plot(to_plot)
    return post_ps_mean, post_sr_mean


def neutrino_analysis():
    # Load the Neutrino Data
    file_path_pattern = 'neutrino_dataset/dataverse_files/events/IC86_*.csv'
    loader = DataLoader(file_path_pattern)
    data = loader.load_multiple_files()  # Or use loader.load_single_file('specific_file.csv') if needed

    # Set Up Random Keys
    seed = 934857243958
    key = random.PRNGKey(seed)
    key, k_i, k_o = random.split(key, 3)

    # Define HEALPix Parameters and Create the Energy Map
    nside = 64  # Adjust based on your data resolution
    healpix_map = HealpixEnergyMap(data, nside=nside)
    energy_map = healpix_map.create_healpix_map(coord_type='azimuthal')
    # 'energy_map' is now a numpy array of size hp.nside2npix(nside)

    # Ensure energy_map is a regular array without masked elements
    if isinstance(energy_map, np.ma.MaskedArray):
        energy_map = energy_map.filled(fill_value=0.0)

    # Replace NaNs with zeros or a suitable value
    energy_map = np.nan_to_num(energy_map, nan=0.0, posinf=0.0, neginf=0.0)

    # Define the Dimensions
    lmax = 3 * nside - 1  # Maximum multipole moment for HEALPix
    dims = lmax + 1  # Include l=0

    # Initialize the Correlated Field Model
    # Ensure that CorrelatedFieldModel is defined as per our data needs
    correlated_field_model = CorrelatedFieldModel(dims=dims)
    # Another option would be:
    # correlated_field_model = CorrelatedFieldModelNeutrino(dims=dims)

    print("Is energy_map a masked array?", isinstance(energy_map, np.ma.MaskedArray))
    print("Number of NaNs in energy_map:", np.isnan(energy_map).sum())

    # Recreate the CorrelatedFieldMaker
    cfm = jft.CorrelatedFieldMaker('neutrino_model_cfm')
    cfm.add_fluctuations(
        shape=nside,
        distances=1.0 / dims,
        **correlated_field_model.cfm_fluctuations,
        prefix="",
        non_parametric_kind="power",
        harmonic_type='spherical'
    )
    cfm.set_amplitude_total_offset(**correlated_field_model.cfm_zero_mode)
    cfm_ps = cfm.power_spectrum
    correlated_field = cfm.finalize()

    # Use our Neutrino Data as 'noisy_data'
    noisy_data = energy_map  # This is our actual observed data

    # Define the Noise Covariance Inverse
    noise_cov = 0.1  # Adjust if you have an estimate of the noise
    noise_cov_inv = lambda x: (1.0 / noise_cov) * x

    # Set Up the Likelihood Function
    likelihood = jft.Gaussian(noisy_data, noise_cov_inv=noise_cov_inv).amend(correlated_field)

    # Define Optimization Parameters
    n_vi_iterations = 2
    delta = 1e-4
    n_samples = 4

    # Run the Optimization
    samples, state = jft.optimize_kl(
        likelihood,
        jft.Vector(likelihood.init(k_i)),
        n_total_iterations=n_vi_iterations,
        n_samples=lambda i: n_samples // 2 if i < 6 else n_samples,
        key=k_o,
        draw_linear_kwargs=dict(
            cg_name="SL",
            cg_kwargs=dict(absdelta=delta * jft.size(likelihood.domain) / 10.0, maxiter=100),
        ),
        nonlinearly_update_kwargs=dict(
            minimize_kwargs=dict(
                name="SN", xtol=delta, cg_kwargs=dict(name=None), maxiter=5,
            )
        ),
        kl_kwargs=dict(
            minimize_kwargs=dict(
                name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35,
            )
        ),
        sample_mode="nonlinear_resample",
        odir="results_neutrino",
        resume=False,
    )

    # Post-Processing Results
    # Compute the posterior mean of the signal reconstruction
    post_sr_mean = jft.mean(tuple(correlated_field(s) for s in samples))

    # Compute the posterior mean of the power spectrum
    post_ps_mean = jft.mean(tuple(cfm_ps(s) for s in samples))

    # Get the grid for plotting
    grid = correlated_field.target_grids[0]

    # Another possibility for plotting the grid is given by:
    # position_space = HPSpace(nside)
    # grid = Field.from_raw(position_space, energy_map)

    # Prepare data for plotting
    to_plot = [
        ("Data", noisy_data, "spherical"),
        ("Reconstruction", post_sr_mean, "spherical"),
        (
            "Amplitude Spectrum",
            (
                grid.harmonic_grid.mode_lengths,
                post_ps_mean,
            ),
            "loglog",
        ),
    ]

    # Plot the Results
    plotter = PlotResults(grid)
    plotter.plot(to_plot)

    # Obsolete parameters (need to investigate why)
    # npix = hp.nside2npix(nside)
    return post_ps_mean, post_sr_mean


if __name__ == "__main__":
    main()