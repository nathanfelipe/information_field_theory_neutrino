import glob
import nifty8 as jft
from jax import random
import nifty8.re as jft
import pandas as pd


class DataLoader:
    def __init__(self, file_path_pattern):
        self.file_path_pattern = file_path_pattern

    @staticmethod
    def load_single_file(self, file_path):
        # Load a single CSV file
        data = pd.read_csv(file_path, sep='\s+')
        data.columns = data.columns.str.strip()  # Clean up column names
        data.columns = ['#', 'MJD', 'log10_E_GeV', 'AngErr_deg', 'RA_deg', 'Dec_deg', 'Azimuth_deg', 'Zenith_deg']
        return data

    def load_multiple_files(self):
        # Load multiple CSV files based on the pattern
        column_names = ['MJD', 'log10_E_GeV', 'AngErr_deg', 'RA_deg', 'Dec_deg', 'Azimuth_deg', 'Zenith_deg']
        csv_files = glob.glob(self.file_path_pattern)
        data_list = []

        for file in csv_files:
            temp_data = pd.read_csv(file,  sep='\s+', header=None, skiprows=1,
                                    skipinitialspace=True, names=column_names, usecols=range(len(column_names)))
            data_list.append(temp_data)

        return pd.concat(data_list, ignore_index=True)

    @staticmethod
    def generate_synthetic_data():
        # Generate synthetic data (similar to the original code)
        print("Generating synthetic data...")
        seed = 42
        key = random.PRNGKey(seed)

        dims = (128, 128)
        distances = 1 / dims[0]
        grid = jft.correlated_field.make_grid(dims, distances=distances, harmonic_type="fourier")

        return grid, key