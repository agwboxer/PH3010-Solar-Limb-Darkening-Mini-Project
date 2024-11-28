#import the relevant modules
from astropy.io import fits
import matplotlib.pyplot as plt

# Open the file
f = fits.open(r'C:\Users\agwbo\Desktop\solar limb darkening\Example-data\DJBtest5_00100.fits')

# Take relevant data in the form of a 2D array
d = f[0].data

# This will plot the full image
plt.imshow(d, cmap='gray')  # Adding cmap='gray' can improve visualization for grayscale images
plt.colorbar()  # Add a colorbar for reference
plt.show()

# This is the ADU value on the representative pixel (580, 1000)
C = d[580, 1000]
print(f"The ADU value at pixel (580, 1000) is {C}")
