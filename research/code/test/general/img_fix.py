from PIL import Image

def repair_png(input_path, output_path):
    with Image.open(input_path) as img:
        img.load()  # Force raster data to load into memory
        img.save(output_path, "PNG")
    print(f"Saved repaired image to {output_path}")

repair_png("/Users/bhal/Downloads/Eu-ssd/Bhalchandra 3/IMG_9800.PNG", "repaired.png")
