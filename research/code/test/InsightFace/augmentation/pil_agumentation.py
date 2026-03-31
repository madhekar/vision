import Augmentor

# 1. Instantiate a Pipeline object pointing to your image directory
p = Augmentor.Pipeline(source_directory="/home/madhekar/tmp")

# 2. Add operations to the Pipeline
# Rotate 90 degrees left or right with 70% probability to simulate yaw
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)

# Optional: Add other transformations
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.flip_left_right(probability=0.5)

# 3. Sample from the pipeline to generate new images
p.sample(5) # Generates 100 augmented images
