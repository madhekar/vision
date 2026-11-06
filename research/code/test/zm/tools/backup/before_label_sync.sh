
rsync -avzh -I --progress /mnt/zmdata/home-media-app/data/input-data/ madhekar@192.168.68.121:/mnt/zmdata/home-media-app/data/input-data/
rsync -avzh -I --progress /mnt/zmdata/home-media-app/data/app-data/ madhekar@192.168.68.121:/mnt/zmdata/home-media-app/data/app-data/
rsync -avzh -I --progress /mnt/zmdata/home-media-app/models/ madhekar@192.168.68.121:/mnt/zmdata/home-media-app/models/
rsync -avzh -I -f"+ */" -f"- *" /mnt/zmdata/home-media-app/data/raw-data/ madhekar@192.168.68.121:/mnt/zmdata/home-media-app/data/raw-data/
