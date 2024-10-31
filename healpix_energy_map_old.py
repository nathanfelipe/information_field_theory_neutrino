import healpy as hp
import numpy as np
import matplotlib.pyplot as plt


class HealpixEnergyMap:
    def __init__(self, data, nside=32):
        self.data = data
        self.nside = nside
        self.energy = self.data['log10_E_GeV']  # Convert log10(E) to linear scale

    def convert_to_radians(self):
        # Convert angles from degrees to radians
        self.ra_radians = np.radians(self.data['RA_deg'])
        self.dec_radians = np.radians(self.data['Dec_deg'])
        self.zenith_radians = np.radians(self.data['Zenith_deg'])
        self.azimuth_radians = np.radians(self.data['Azimuth_deg'])

    def create_healpix_map(self, coord_type='equatorial'):
        # Convert angular coordinates to HEALPix pixel indices and accumulate energy
        self.convert_to_radians()

        if coord_type == 'equatorial':
            theta_eq = np.pi / 2 - self.dec_radians  # self.dec_radians + np.pi / 2
            phi_eq = self.ra_radians
            pix_idx_eq = hp.ang2pix(self.nside, theta_eq, phi_eq)
            energy_map = np.zeros(hp.nside2npix(self.nside))
            np.add.at(energy_map, pix_idx_eq, self.energy)
            return energy_map
        else:
            theta_az = self.zenith_radians
            phi_az = self.azimuth_radians  # - np.pi / 2 Double check this change later
            pix_idx_az = hp.ang2pix(self.nside, theta_az, phi_az)
            energy_map = np.zeros(hp.nside2npix(self.nside))
            np.add.at(energy_map, pix_idx_az, self.energy)
            return energy_map

    def plot_healpix_map(self, output_file):
        # Create a figure with two subplots (equatorial and azimuthal)
        energy_map_eq = self.create_healpix_map(coord_type='equatorial')
        energy_map_az = self.create_healpix_map(coord_type='azimuthal')  # also known as celestial

        fig = plt.figure(figsize=(10, 12))

        # First subplot: Equatorial coordinates
        ax1 = fig.add_subplot(2, 1, 1)
        hp.mollview(energy_map_eq, coord=['C'], rot=(0, 180, 0),
                    title="Accumulated Neutrino Energy Map (Equatorial Coordinates)",
                    unit='Energy (log_GeV)', cmap='viridis', sub=(2, 1, 1), norm="log")
        hp.graticule()

        # Second subplot: Azimuthal coordinates
        ax2 = fig.add_subplot(2, 1, 2)
        hp.mollview(energy_map_az, coord=['C'], rot=(0, 180, 0),
                    title="Accumulated Neutrino Energy Map (Celestial Coordinates)",
                    unit='Energy (log_GeV)', cmap='viridis', sub=(2, 1, 2), fig=fig, norm="log")
        hp.graticule()

        # Save the figure
        plt.savefig(output_file)
        plt.show()
