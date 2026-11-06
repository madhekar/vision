
!#/bin/bash

sudo rsync -azrc -I --delete --progress /mnt/zmdata/home-media-app/data/input-data/ madhekar@192.168.68.121:/mnt/zmdata/home-media-app/data/input-data/
sudo rsync -azrc -I --delete --progress /mnt/zmdata/home-media-app/data/app-data/ madhekar@192.168.68.121:/mnt/zmdata/home-media-app/data/app-data/
sudo rsync -azrc -I --delete --progress /mnt/zmdata/home-media-app/models/ madhekar@192.168.68.121:/mnt/zmdata/home-media-app/models/
sudo rsync -azrc -I --delete -f"+ */" -f"- *" /mnt/zmdata/home-media-app/data/raw-data/ madhekar@192.168.68.121:/mnt/zmdata/home-media-app/data/raw-data/
