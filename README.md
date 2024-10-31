
# Neutrino Observation and Information Field Theory with JAX

## Project Overview

This repository contains a set of tools and scripts for analyzing and modeling astrophysical data, specifically focusing on the observation and analysis of neutrino energies and fields. Leveraging Information Field Theory (IFT) and the JAX framework, this project aims to implement computational methods to interpret high-dimensional data fields derived from neutrino observations.

This project uses **JAX** for high-performance computations, especially on GPU/TPU, **NIFTy** for managing field-based data, and **HEALPix** for pixelizing data on spherical surfaces, which is crucial for astrophysical maps. 

## Features

- **Field Correlation Modeling**: Includes code for building correlated field models that facilitate understanding of large-scale structures in astrophysical data.
- **Neutrino Energy Histogram Analysis**: Provides scripts for generating histograms from neutrino energy data to analyze distribution and occurrences.
- **Wiener Filtering for Field Reconstruction**: Uses Wiener filtering techniques to enhance and reconstruct noisy data fields.
- **HEALPix-Based Energy Mapping**: Provides tools for visualizing energy distributions on a spherical (sky) map.
- **Data Loading and Transformation**: Utility functions for loading astrophysical data and transforming it into formats suitable for further analysis.
- **JAX Acceleration**: Optimizes computations using JAX, facilitating efficient GPU/TPU acceleration for handling large astrophysical datasets.

## Requirements

The following Python packages are required to run the code. Install them via `pip`:

```bash
pip install -r requirements.txt
```

- `jax==0.4.30`
- `healpy==1.17.3`
- `matplotlib==3.9.0`
- `nifty8==8.5.3`
- `pandas==2.2.2`
- `numpy==1.26.4`
- `jaxbind==1.1.0`
- `ducc0==0.34.0`

For a detailed list, see the [requirements.txt](requirements.txt) file.

## File Structure

Here is an overview of the main files and their roles in this project:

- **`correlated_field_model.py`**: Defines correlated field models using JAX and NIFTy. This script models large-scale astrophysical structures, crucial for interpreting observational data.
- **`data_loader.py`**: Contains utilities to load and preprocess datasets, especially focusing on data relevant to neutrino energies and spatial distribution.
- **`healpix_energy_map.py`**: Implements HEALPix-based energy mapping, which allows for the spherical projection of neutrino energy data on the sky.
- **`main.py`**: The main driver script that ties together data loading, field modeling, and visualization tools, facilitating a seamless workflow from data input to analysis.
- **`neutrino_energy_histogram.py`**: Generates histograms from neutrino energy data, helping visualize energy distributions.
- **`wiener_filter.py`**: Implements Wiener filtering to reconstruct signals from noisy data, applying Bayesian techniques to enhance data quality.

## Getting Started

### Prerequisites

Ensure that you have Python 3.8+ installed and GPU/TPU support for JAX if you are working with large datasets and require accelerated computations.

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/your_repo.git
    cd your_repo
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Scripts

1. **Correlated Field Modeling**: Run `correlated_field_model.py` to build and analyze correlated fields based on observational data.
    ```bash
    python correlated_field_model.py
    ```

2. **Data Loading**: Use `data_loader.py` to load datasets into your environment.
    ```bash
    python data_loader.py
    ```

3. **Energy Mapping and Histogram Analysis**: 
   - Execute `healpix_energy_map.py` to create HEALPix projections of neutrino energies.
   - Use `neutrino_energy_histogram.py` to create energy histograms for observational insights.
   
4. **Wiener Filtering**: Use `wiener_filter.py` to apply Wiener filtering to data fields for noise reduction.
    ```bash
    python wiener_filter.py
    ```

### Example Usage

To reconstruct an energy field and visualize it:

```python
import correlated_field_model
import healpix_energy_map

# Example function calls
field = correlated_field_model.create_field()
map_data = healpix_energy_map.generate_map(field)
```

## Contributions

This repository is open to contributions. If you'd like to collaborate, please submit a pull request or open an issue.

## License

Distributed under the MIT License. See `LICENSE` for more information.
