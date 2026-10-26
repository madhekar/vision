import struct
import zlib

def fix_png_structure(input_path: str, output_path: str):
    """
    Parses a corrupted PNG file, corrects invalid magic bytes, 
    recalculates bad CRC checksums, and outputs a repaired file.
    """
    PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
    
    with open(input_path, 'rb') as f:
        data = f.read()

    # Step 1: Force correct PNG Magic Bytes
    if not data.startswith(PNG_SIGNATURE):
        print("[!] Invalid PNG signature detected. Fixing headers...")
        # Strip corrupted leading bytes or prepend valid signature if completely missing
        if b'PNG' in data[:10]:
            idx = data.find(b'PNG') - 1
            data = PNG_SIGNATURE + data[idx+4:]
        else:
            data = PNG_SIGNATURE + data[8:]

    repaired_data = bytearray(data[:8])
    offset = 8
    length_data = len(data)

    # Step 2: Traverse and rebuild chunks
    while offset < length_data:
        if offset + 8 > length_data:
            break
            
        # Read length and chunk type
        chunk_length = struct.unpack('>I', data[offset:offset+4])[0]
        chunk_type = data[offset+4:offset+8]
        
        # Guard against truncated or corrupted length metrics
        if offset + 8 + chunk_length + 4 > length_data:
            print(f"[!] Warning: Chunk {chunk_type.decode(errors='ignore')} is truncated. Truncating file read.")
            # Auto-append a clean IEND if missing or cut off
            if chunk_type != b'IEND':
                repaired_data.extend(struct.pack('>I', 0) + b'IEND' + zlib.crc32(b'IEND').to_bytes(4, 'big'))
            break

        chunk_data = data[offset+8:offset+8+chunk_length]
        current_crc = struct.unpack('>I', data[offset+8+chunk_length:offset+12+chunk_length])[0]
        
        # Step 3: Verify and recalculate CRC
        calculated_crc = zlib.crc32(chunk_type + chunk_data) & 0xffffffff
        
        if current_crc != calculated_crc:
            print(f"[*] Fixing incorrect CRC for chunk {chunk_type.decode(errors='ignore')} "
                  f"(Expected: {hex(calculated_crc)}, Found: {hex(current_crc)})")
            current_crc = calculated_crc

        # Re-pack fixed structure
        repaired_data.extend(struct.pack('>I', chunk_length))
        repaired_data.extend(chunk_type)
        repaired_data.extend(chunk_data)
        repaired_data.extend(struct.pack('>I', current_crc))
        
        offset += 12 + chunk_length
        if chunk_type == b'IEND':
            break

    with open(output_path, 'wb') as f:
        f.write(repaired_data)
    print(f"[+] Repaired PNG safely written to: {output_path}")

# Example execution
fix_png_structure("/Users/bhal/Downloads/Eu-ssd/Bhalchandra 3/IMG_9800.PNG", "fixed_structure.PNG")
