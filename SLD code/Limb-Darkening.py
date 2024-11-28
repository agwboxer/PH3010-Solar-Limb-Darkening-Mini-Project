import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# Path to the directory containing the FITS files - Change to relevant file path
path = r'C:\Users\agwbo\Desktop\solar limb darkening\Example-data'

# Find all files with similar names
files = glob.glob(f"{path}\DJBtest5_*.fits")

# Initialize a list to store central intensities
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
    
    # Compute the mean intensity of the central region
    central_intensities.append(central_intensity)

    print(f"Processed {file}, intensity: {central_intensity}")

# Plot the mean intensities
plt.plot(central_intensities)
plt.xlabel("Position on sun")
plt.ylabel("Mean Intensity (ADU)")
plt.title("Intensity of Sun's Radiation as a function of position")
plt.show()
