import matplotlib.pyplot as plt


class NeutrinoEnergyHistogram:
    def __init__(self, data):
        self.data = data

    def plot_histogram(self, output_file):
        # Plotting the neutrino count based on energy
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['log10_E_GeV'], bins=50, edgecolor='black')
        plt.title('Neutrino Counts Based on Energy')
        plt.xlabel('log10(E/GeV)')
        plt.ylabel('Neutrino Count')
        plt.grid(True)

        # Save the figure locally as a PNG file
        plt.savefig(output_file)
        plt.show()
