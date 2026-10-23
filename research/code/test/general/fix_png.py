import struct
import zlib

def fix_png_crc(input_path, output_path):
    with open(input_path, "rb") as f:
        data = f.read()
    
    # PNG Magic bytes are 8 bytes long. IHDR starts next.
    # IHDR chunk data is located at bytes 12-28
    # Chunk type ("IHDR") + 13 bytes of data = 17 bytes
    ihdr_type_and_data = data[16:29] 
    
    # Recalculate the proper CRC-32 checksum
    correct_crc = zlib.crc32(ihdr_type_and_data) & 0xFFFFFFFF
    
    # Patch the file: Replace bytes 29-33 with the calculated CRC
    fixed_data = data[:29] + struct.pack(">I", correct_crc) + data[33:]
    
    with open(output_path, "wb") as f:
        f.write(fixed_data)
    print(f"Fixed CRC written to {output_path}!")

# Usage
fix_png_crc("/Users/bhal/Downloads/Eu-ssd/Bhalchandra 3/IMG_9800.PNG", "fixed_file.png")
