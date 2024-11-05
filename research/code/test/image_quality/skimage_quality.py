from skimage import io, measure

# Load images
image1 = io.imread("/Users/bhal/Downloads/IMG_8646.HEIC")
image2 = io.imread("/Users/bhal/Downloads/IMG_8646.HEIC")

# Calculate Mean Squared Error (MSE)
mse = measure.marching_cubes()

print("MSE:", mse)