from picarta import Picarta

api_token = "YOUR_API_TOKEN"
localizer = Picarta(api_token)

# Geolocate a local image
result = localizer.localize(img_path="/path/to/local/image.jpg")

print(result)

# Geolocate an image from URL with optional parameters for a specific location search
result = localizer.localize(
img_path="https://upload.wikimedia.org/wikipedia/commons/8/83/San_Gimignano_03.jpg",
top_k=10,
center_latitude=43.464, 
center_longitude=11.038,
radius=100)

print(result)