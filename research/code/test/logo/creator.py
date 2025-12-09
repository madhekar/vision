from PIL import Image, ImageDraw, ImageFont


def create_logo_0(text_content, font_path):
    # Define the image size and the desired text/font
    img_width, img_height = 300, 150
    font_size = 50
    #text_content = "My Logo"
    #font_path = "arial.ttf" # Use a valid font file path, 'arial.ttf' is common on Windows

    try:
        fnt = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Could not load font {font_path}. Using default font.")
        fnt = ImageFont.load_default()

    # Create a new image in RGBA mode
    # color=(R, G, B, A). (0, 0, 0, 0) is fully transparent black.
    img = Image.new(mode="RGBA", size=(img_width, img_height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Calculate text position (simple centering)
    # Using textbbox for better positioning calculation in modern Pillow
    bbox = draw.textbbox((0, 0), text_content, font=fnt)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (img_width - text_width) / 2
    text_y = (img_height - text_height) / 2

    # Draw the text with a solid fill (fill=(R, G, B, A), A=255 is opaque)
    draw.text((text_x, text_y), text_content, font=fnt, fill=(0, 0, 0, 255))

    # Save the image as a PNG file, which supports transparency
    output_filename = "transparent_logo.png"
    img.save(output_filename, "PNG")

    print(f"Created transparent logo: {output_filename}")

def create_logo_1(text, font_path):
    # Create a new RGBA image (Red, Green, Blue, Alpha) with 50% transparency for background
    img = Image.new("RGBA", (200, 100), (255, 255, 255, 127)) # White with 50% opacity
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 30)
    draw.text((10, 30), text, font=font, fill=(0, 0, 0, 255)) # Black text, fully opaque
    img.save("zm_logo.png", "PNG") # Save as PNG to preserve transparency
    
