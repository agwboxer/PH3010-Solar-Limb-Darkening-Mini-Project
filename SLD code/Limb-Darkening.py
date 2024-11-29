import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the generalized polynomial fit function
def func(x, *theta):
    return sum(theta[i] * x**i for i in range(len(theta)))

# Path to the directory containing the FITS files - Change to relevant file path
path = r'C:\Users\agwbo\Desktop\solar limb darkening\Example-data'

# Find all files with similar names
files = glob.glob(f"{path}\DJBtest5_*.fits")

# Initialise a list to store central intensities
central_intensities = []

# Process each file and gather central intensities
for file in files:
    # Open the FITS file
    f = fits.open(file)

    # Extract data
    data = f[0].data
    
    # Find the central region
    center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
    central_intensity = data[center_y, center_x]
    
    # Append intensity to the list
    central_intensities.append(central_intensity)

    print(f"Processed {file}, intensity: {central_intensity}")

# Create an x-axis for the number of files
x = np.arange(len(central_intensities))
y = np.array(central_intensities)

# Fit polynomials of different orders
for M in range(2, 6):  # Loop over polynomial orders
    print(f"\nFitting polynomial of order M = {M}:")
    
    # Set initial parameter values (guess)
    p0 = np.ones(M + 1)  # Initial guess for Mth order polynomial

    # Perform the fit
    theta_hat, cov = curve_fit(func, x, y, p0)

    # Print fitted parameters
    print("Fitted parameters:")
    for i, param in enumerate(theta_hat):
        print(f"theta_hat[{i}] = {param}")
    
    # Plot the data and the fit
    plt.plot(x, y, color='black', label='Measured Data')
    x_fit = np.linspace(0, len(x) - 1, 500)
    y_fit = func(x_fit, *theta_hat)
    plt.plot(x_fit, y_fit, '--', label=f'Polynomial Fit (M={M})')
    plt.xlabel("Position on Sun")
    plt.ylabel("Intensity (ADU)")
    plt.title(f"Intensity of Sun's Radiation (Polynomial Order M={M})")
    plt.legend()
    plt.show()
