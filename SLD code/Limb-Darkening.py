from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Open the FITS file and extract the image data
file_path = r'C:\Users\agwbo\Desktop\solar limb darkening\data\DJBtest5_00105.fits'
f = fits.open(file_path)

data = f[0].data

# Determine the center of the solar disk and its radius
centre_x, centre_y = data.shape[0] // 2, data.shape[1] // 2
radius = min(centre_x, centre_y)

# Generate a grid of pixel indices for plotting and calculations
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
xx, yy = np.meshgrid(x, y)

# Extract the pixel values of the central row of the image
central_row = data[centre_x, :]

# Calculate the gradient (rate of change) of the intensity across the central row
gradient = np.gradient(central_row)

# Detect a sharp rise: find where the gradient exceeds a specific threshold
sharp_rise_threshold = np.max(gradient) * 0.99999999999
sharp_rise_index = np.argmax(gradient > sharp_rise_threshold)

# Compute the background intensity from the region before the sharp rise
background_pixels = central_row[:sharp_rise_index]
background_threshold = np.mean(background_pixels)
print(f"Background Intensity (ADU): {background_threshold}")
print(f"Sharp rise detected at pixel index: {sharp_rise_index}")

# Filter the pixels that are above the background threshold
above_threshold_indices = np.where(central_row > background_threshold)[0]
above_threshold_values = central_row[above_threshold_indices]

# Estimate the radius of the solar disk based on the indices above the threshold
radius = (min(above_threshold_indices) + max(above_threshold_indices)) // 2

# Define the range of radial distances for computing the cosine of the angle (mu)
r = np.linspace(-radius, radius, data.shape[1])

# Calculate the normalized radial distance (r/R) and corresponding mu values (cosine of angle)
normalised_r = np.abs(r) / radius
mu = np.sqrt(1 - normalised_r**2)

# Restrict the mu values to the range of the above threshold indices
mu_restrictions = np.linspace(min(above_threshold_indices) - min(above_threshold_indices) // 3,
                              max(above_threshold_indices) - min(above_threshold_indices) // 3, len(above_threshold_indices)).astype(int)
mu_above_threshold = mu[mu_restrictions]

# Plot the mu values (cosine of the angle) as a function of pixel index
plt.figure(figsize=(12, 6))
plt.plot(above_threshold_indices, mu_above_threshold, label="Mu (cos$\theta$)", color='green')
plt.title(r"Values of $\mu$ over the Sun's Disk")
plt.xlabel("Pixel Index")
plt.ylabel(r"$\mu$ (cos $\theta$)")
plt.legend()
plt.show()

# Plot only the intensities above the background threshold
plt.figure(figsize=(12, 6))
plt.plot(above_threshold_indices, above_threshold_values, label='Intensity', color='blue')
plt.axhline(y=background_threshold, color='red', linestyle='--', label='Background Threshold')
plt.title("Intensity Over the Sun's Disk")
plt.xlabel("Pixel Index")
plt.ylabel("Intensity (ADU)")
plt.legend()
plt.show()

# Plot the full FITS image for reference
plt.imshow(data, cmap='gray')
plt.colorbar() 
plt.title("FITS Image")
plt.show()

# Define the limb darkening model (4th order polynomial)
def limb_darkening(mu, a0, a1, a2, a3, a4):
    return a0 + a1 * mu + a2 * mu**2 + a3 * mu**3 + a4 * mu**4

# Clean the data by removing NaN or infinite values from the mu and intensity arrays
valid_indices = np.isfinite(mu_above_threshold) & np.isfinite(above_threshold_values)
mu_above_threshold_clean = mu_above_threshold[valid_indices]
above_threshold_values_clean = above_threshold_values[valid_indices]

# Perform curve fitting using the limb darkening model
params, covariance = curve_fit(limb_darkening, mu_above_threshold_clean, above_threshold_values_clean)

# Extract the fitted parameters for the limb darkening model
a0_fit, a1_fit, a2_fit, a3_fit, a4_fit = params

# Plot the measured data and the fitted limb darkening curve
plt.scatter(mu_above_threshold_clean, above_threshold_values_clean, label="Measured Data", color="blue")
plt.plot(mu_above_threshold_clean, limb_darkening(np.abs(mu_above_threshold_clean), *params), label="Fitted Curve", color="red")
plt.title("Limb Darkening Fit (4th Order)")
plt.xlabel(r"$\mu = \cos\theta$")
plt.ylabel("Intensity (ADU)")
plt.legend()
plt.show()

# Output the fitted parameters for analysis
print(f"Fitted Limb Darkening Parameters:\n a0' = {a0_fit}\n a1' = {a1_fit}\n a2' = {a2_fit}\n a3' = {a3_fit}\n a4' = {a4_fit}")

# Close the FITS file to free resources
f.close()
