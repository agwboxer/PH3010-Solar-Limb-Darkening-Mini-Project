from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Open the FITS file and extract the image data
file_path = r'%filepath'
f = fits.open(file_path)

data = f[0].data

# Determine the center of the solar disk and its radius
centre_x, centre_y = data.shape[0] // 2, data.shape[1] // 2

# Generate a grid of pixel indices for plotting and calculations
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
xx, yy = np.meshgrid(x, y)

# Extract the pixel values of the central row of the image
central_row = data[centre_x, :]

# Calculate the gradient (rate of change) of the intensity across the central row
gradient = np.gradient(central_row)

# Detect a sharp rise: find where the gradient exceeds a specific threshold
sharp_rise_threshold = np.max(gradient) * 0.55
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
r = np.arange(-radius, radius)

# Calculate the normalized radial distance (r/R) and corresponding mu values (cosine of angle)
normalised_r = np.abs(r) / radius
mu = np.sqrt(1 - normalised_r**2)

x_start = min(above_threshold_indices)
x = np.arange(x_start, x_start + len(mu))

    # Plot the mu values (cosine of the angle) as a function of pixel index
plt.figure(figsize=(12, 6))
plt.plot(x, mu, label="Mu (cos$\theta$)", color='green')
plt.xlabel("Pixel Index")
plt.ylabel(r"$\mu$ (cos $\theta$)")
plt.legend()
plt.show()
if False:
    # Plot the full FITS image for reference
    plt.imshow(data, cmap='gray')
    plt.colorbar()
    plt.show()
    
    plt.scatter(mu_above_threshold, above_threshold_values, label="Measured Data", color="blue")
    plt.plot(mu_above_threshold, limb_darkening(mu_above_threshold, *params), label="Fitted Curve", color="red")
    plt.xlabel(r"$\mu = \cos\theta$")
    plt.ylabel("Intensity (ADU)")
    plt.legend()
    plt.show()


# Plot only the intensities above the background threshold
plt.figure(figsize=(12, 6))
plt.plot(above_threshold_indices, above_threshold_values, label='Intensity', color='blue')
plt.axhline(y=background_threshold, color='red', linestyle='--', label='Background Threshold')
plt.xlabel("Pixel Index")
plt.ylabel("Intensity (ADU)")
plt.legend()
plt.show()



# Define the limb darkening model (5th order polynomial)
def limb_darkening(mu, a0, a1, a2, a3, a4):
    return a0 + a1 * mu + 2 * a2 * mu**2 + 6 * a3 * mu**3 + 24 * a4 * mu**4

# Restrict mu to the indices corresponding to above_threshold_indices
mu_above_threshold = mu[above_threshold_indices]
# Clean the data by removing NaN or infinite values from the mu and intensity arrays


# Perform curve fitting using the 5th-order limb darkening model
params, covariance = curve_fit(limb_darkening, mu_above_threshold, above_threshold_values)

# Extract the fitted parameters for the limb darkening model
a0_fit, a1_fit, a2_fit, a3_fit, a4_fit = params

# Plot the measured data and the fitted limb darkening curve

# Output the fitted parameters for analysis
print(f"Fitted Limb Darkening Parameters:")
print(f"a0 = {a0_fit}")
print(f"a1 = {a1_fit}")
print(f"a2 = {a2_fit}")
print(f"a3 = {a3_fit}")
print(f"a4 = {a4_fit}")

# Calculate intensity using the fitted limb darkening model
fitted_intensity = limb_darkening(mu_above_threshold, *params)

# Calculate temperature from the fitted intensity using the Planck function
def temperature_from_intensity(intensity, wavelength):
    h = 6.62607015e-34  # Planck constant (Joule second)
    c = 3.0e8  # Speed of light (m/s)
    k = 1.380649e-23  # Boltzmann constant (Joule/Kelvin)
    return (h * c / (wavelength * k)) / np.log(1 + (2 * h * c**2) / (wavelength**5 * intensity))

# Wavelength in meters 
wavelength = 700e-9 # edit to according filter

# Calculate temperatures for the fitted intensities
temperatures = temperature_from_intensity(fitted_intensity, wavelength)

# Plot the limb-darkening model against mu
plt.figure(figsize=(12, 6))
plt.scatter(mu_above_threshold, above_threshold_values, label="Measured Data", color="blue")
plt.plot(mu_above_threshold, fitted_intensity, label="Fitted Limb-Darkening Model", color="red")
plt.xlabel(r"$\mu = \cos\theta$")
plt.ylabel("Intensity (ADU)")
plt.legend()
plt.show()

# Plot the temperature against mu
plt.figure(figsize=(12, 6))
plt.plot(mu_above_threshold, temperatures, label="Temperature", color="orange")
plt.xlabel(r"$\mu = \cos\theta$")
plt.ylabel("Temperature (K)")
plt.show()


# Close the FITS file to free resources
f.close()

