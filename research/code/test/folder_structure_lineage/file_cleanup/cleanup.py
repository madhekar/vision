import os



def check_size(file_path):
    try:
        size = os.path.getsize(file_path)
        print(f"File size: {size} bytes")
        
        # Convert to Megabytes for readability
        print(f"File size: {size / (1024 * 1024):.2f} MB")
    except OSError as e:
        print(f"Error: {e}")


img1 = "/mnt/zmdata/home-media-app/data/input-data/img/GRANDCANYON/2bdfd5ed-baad-5106-8579-0f1c6ac12ded/IMG_20181227_113004.jpg" #File size: 262579 bytes
#"/mnt/zmdata/home-media-app/data/input-data/img/GRANDCANYON/2bdfd5ed-baad-5106-8579-0f1c6ac12ded/IMG_20181225_114314.jpg"      #File size:   88111 bytes
#"/mnt/zmdata/home-media-app/data/input-data/img/GRANDCANYON/1ff9c465-fa8b-5067-8af4-1b1a9484b55a/vcm_s_kf_repr_832x624.jpg"      #File size: 59057 bytes  #File size: 0.06 MB
#"/mnt/zmdata/home-media-app/data/input-data/img/GRANDCANYON/1b5b4aae-a35e-52a7-990b-af65dc370bcb/vcm_s_kf_m160_160x120.jpg"      # File size: 7086 bytes  

check_size(img1)