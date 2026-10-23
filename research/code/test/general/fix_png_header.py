PNG_HEADER = b'\x89PNG\r\n\x1a\n'

def fix_png_header(input_path, output_path):
    with open(input_path, "rb") as f:
        data = f.read()
    
    # Check if header is missing, if so, prepend it
    if not data.startswith(PNG_HEADER):
        # Strip any bad headers if you know the exact offset, or prepend directly:
        fixed_data = PNG_HEADER + data[8:] # Assumes the first 8 bytes were bad
        
        with open(output_path, "wb") as f:
            f.write(fixed_data)
        print("PNG Header restored.")
    else:
        print("Header looks normal.")

fix_png_header("/Users/bhal/Downloads/Eu-ssd/Bhalchandra 3/IMG_9800.PNG", "fixed_file.png")