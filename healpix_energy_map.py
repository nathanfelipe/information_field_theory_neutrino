import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from numpy import ma
from matplotlib.colors import LogNorm


class HealpixEnergyMap:
    def __init__(self, data, nside=32):
        self.data = data
        self.nside = nside
        self.energy = 10 ** self.data['log10_E_GeV']  # Linear energy in GeV
        self.log_energy = self.data['log10_E_GeV']
        self.coord_type = "azimuthal"

    def create_healpix_map(self, coord_type, map_type="sum_energy"):
        if coord_type == 'equatorial':
            # Use degrees directly; latitude is Dec_deg
            theta = self.data['Dec_deg']  # Latitude in degrees [-90, +90]
            phi = self.data['RA_deg']     # Longitude in degrees [0, 360]
        elif coord_type == 'azimuthal':
            # Convert Zenith angle to latitude
            theta = -90 + self.data['Zenith_deg']  # Latitude in degrees [-90, +90]
            phi = self.data['Azimuth_deg']        # Azimuth in degrees [0, 360]
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

        # Use lonlat=True since angles are in degrees (longitude and latitude)
        pix_idx = hp.ang2pix(self.nside, phi, theta, lonlat=True)
        npix = hp.nside2npix(self.nside)

        if map_type == 'counts':
            counts_map = np.zeros(npix)
            np.add.at(counts_map, pix_idx, 1)
            counts_map = ma.masked_where(counts_map == 0, counts_map)
            return counts_map
        elif map_type == 'sum_energy':
            energy_map = np.zeros(npix)
            np.add.at(energy_map, pix_idx, self.energy)
            energy_map = ma.masked_where(energy_map == 0, energy_map)
            return energy_map
        elif map_type == 'average_energy':
            energy_sum_map = np.zeros(npix)
            counts_map = np.zeros(npix)
            np.add.at(energy_sum_map, pix_idx, self.energy)
            np.add.at(counts_map, pix_idx, 1)
            with np.errstate(divide='ignore', invalid='ignore'):
                average_energy_map = np.divide(energy_sum_map, counts_map)
            average_energy_map = ma.masked_where(counts_map == 0, average_energy_map)
            return average_energy_map
        else:
            raise ValueError(f"Unknown map_type: {map_type}")

    def plot_healpix_map(self, output_file, map_type='sum_energy'):
        # Create maps
        map_eq = self.create_healpix_map(coord_type='equatorial', map_type=map_type)
        map_az = self.create_healpix_map(coord_type='azimuthal', map_type=map_type)

        # Print min and max values for debugging
        print(f"Equatorial Map ({map_type}): min =", map_eq.min(), "max =", map_eq.max())
        print(f"Azimuthal Map ({map_type}): min =", map_az.min(), "max =", map_az.max())

        fig = plt.figure(figsize=(10, 12))

        # Determine plotting parameters based on map_type
        if map_type == 'counts':
            norm = 'log'
            unit = 'Counts'
            title_eq = "Neutrino Event Counts (Equatorial Coordinates)"
            title_az = "Neutrino Event Counts (Azimuthal Coordinates)"
        elif map_type == 'sum_energy':
            norm = 'log'
            unit = 'Energy (GeV)'
            title_eq = "Accumulated Neutrino Energy (Equatorial Coordinates)"
            title_az = "Accumulated Neutrino Energy (Azimuthal Coordinates)"
        elif map_type == 'average_energy':
            norm = None
            unit = 'Average Energy (GeV)'
            title_eq = "Average Neutrino Energy (Equatorial Coordinates)"
            title_az = "Average Neutrino Energy (Azimuthal Coordinates)"
        else:
            raise ValueError(f"Unknown map_type: {map_type}")

        # First subplot: Equatorial coordinates
        hp.mollview(map_eq, coord=['C'], # rot=(0, 180, 0),
                    title=title_eq, unit=unit, cmap='viridis',
                    sub=(2, 1, 1), norm=norm)
        hp.graticule()

        # Second subplot: Azimuthal coordinates
        hp.mollview(map_az, coord=['C'], # rot=(0, 180, 0),
                    title=title_az, unit=unit, cmap='viridis',
                    sub=(2, 1, 2), norm=norm)
        hp.graticule()

        # Save and show the figure
        plt.savefig(output_file)
        plt.show()

    def plot_individual_neutrinos(self, output_file, coord_type):
        if coord_type == 'equatorial':
            theta = self.data['Dec_deg']  # Latitude in degrees [-90, +90]
            phi = self.data['RA_deg']
            title = "Individual Neutrino Events (Equatorial Coordinates)"
        elif coord_type == 'azimuthal':
            theta = 90 - self.data['Zenith_deg']  # Convert Zenith angle to latitude
            phi = self.data['Azimuth_deg']
            title = "Individual Neutrino Events (Azimuthal Coordinates)"
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

        # Convert angles to radians for projection
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)

        # Convert spherical coordinates to Cartesian for plotting
        hp_proj = hp.projector.MollweideProj(rot=(0, 180, 0))

        x, y = hp_proj.proj(phi_rad, theta_rad, lonlat=True)

        # Create a scatter plot
        plt.figure(figsize=(10, 5))
        hp.mollview(np.zeros(hp.nside2npix(self.nside)), coord=['C'], rot=(0, 180, 0),
                    title=title, unit='', cmap='gray', min=0, max=1)
        hp.graticule()
        plt.scatter(x, y, c=self.energy, s=1, cmap='viridis', norm=LogNorm())
        plt.colorbar(label='Energy (GeV)')
        plt.savefig(output_file)
        plt.show()